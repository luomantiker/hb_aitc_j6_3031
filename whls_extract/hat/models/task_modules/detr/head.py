# Copyright (c) Horizon Robotics. All rights reserved.

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.mlp_module import MlpModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["DetrHead"]


@OBJECT_REGISTRY.register
class DetrHead(nn.Module):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        transformer: transformer module.
        pos_embed: position encoding module.
        num_classes: Number of categories excluding the background.
        in_channels: Number of channels in the input feature map.
        max_per_img: Number of object queries, ie detection slot.
            The maximal number of objects DETR can detect in a single image.
            For COCO, we recommend 100 queries.
        int8_output: If True, output int8, otherwise output int32.
            Default: False.
        dequant_output: Whether to dequant output. Default: True.
        set_int16_qconfig: Whether to set int16 qconfig. Default: False.
        input_shape: shape used to construct masks for inference.
    """

    def __init__(
        self,
        transformer: nn.Module,
        pos_embed: nn.Module,
        num_classes: int = 80,
        in_channels: int = 2048,
        max_per_img: int = 100,
        int8_output: bool = False,
        dequant_output: bool = True,
        set_int16_qconfig: bool = False,
        input_shape: tuple = (800, 1332),
    ):
        super(DetrHead, self).__init__()
        num_pos_feats = pos_embed.num_pos_feats
        embed_dims = transformer.embed_dims
        assert num_pos_feats * 2 == embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_pos_feats. Found {embed_dims}"
            f" and {num_pos_feats}."
        )

        self.num_classes = num_classes
        self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_query = max_per_img
        self.activate = nn.ReLU(inplace=True)
        self.pos_embed = pos_embed
        self.transformer = transformer
        self.act_layer = nn.ReLU(inplace=True)

        self.int8_output = int8_output
        self.dequant_output = dequant_output
        self.set_int16_qconfig = set_int16_qconfig
        self.input_shape = input_shape

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        self.input_proj = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1
        )
        self.fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = MlpModule2d(
            self.embed_dims,
            self.embed_dims,
            self.embed_dims,
            act_layer=self.act_layer,
            drop_ratio=0.0,
        )
        self.fc_reg = nn.Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        self.pos_embed_quant = QuantStub(scale=None)
        self.query_embed_weight_quant = QuantStub(scale=None)

    @fx_wrap()
    def forward_single(self, x, img_meta):
        """Forward features of a single scale levle.

        Args:
            x: FPN feature maps of the specified stride.
            img_meta: Dict containing keys of different image size.
                batch_input_shape means image size after padding while
                img_shape means image size after data augment,
                but before padding.
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        if self.training:
            if "batch_input_shape" in img_meta:
                input_img_h, input_img_w = img_meta["batch_input_shape"][0]
            else:
                input_img_h, input_img_w = self.input_shape
            masks = torch.ones(
                (batch_size, input_img_h, input_img_w), device=x.device
            )
            for img_id in range(batch_size):
                if "before_pad_shape" in img_meta:
                    img_h, img_w = img_meta["before_pad_shape"][img_id]
                elif "img_shape" in img_meta:
                    _, img_h, img_w = img_meta["img_shape"][img_id]
                else:
                    img_h, img_w = self.input_shape
                masks[img_id, :img_h, :img_w] = 0
        else:
            input_img_h, input_img_w = self.input_shape
            masks = torch.zeros(
                (batch_size, input_img_h, input_img_w), device=x.device
            )

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = (
            F.interpolate(masks.unsqueeze(1), size=x.shape[-2:])
            .to(torch.bool)
            .squeeze(1)
        )
        # position encoding
        pos_embed = self.pos_embed(masks)  # [bs, embed_dim, h, w]
        pos_embed = self.pos_embed_quant(pos_embed)
        query_embedding_weight = self.query_embed_weight_quant(
            self.query_embedding.weight
        )
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(
            x, masks, query_embedding_weight, pos_embed
        )

        outputs_class = self.fc_cls(outs_dec)
        outputs_coord = self.fc_reg(self.activate(self.reg_ffn(outs_dec)))

        if self.dequant_output:
            outputs_class = self.dequant(outputs_class)
            outputs_coord = self.dequant(outputs_coord)

        return outputs_class, outputs_coord

    def forward(self, feats, img_meta):
        feats = feats[-1]
        outputs = self.forward_single(feats, img_meta)

        return outputs

    def fuse_model(self):
        pass

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.query_embedding.qconfig = None
        if self.set_int16_qconfig:

            self.reg_ffn.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.activate.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.fc_cls.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.fc_reg.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            if hasattr(self.transformer, "set_qconfig"):
                self.transformer.set_qconfig()
        else:
            from hat.utils import qconfig_manager

            # disable output quantization for last quanti layer.
            if not self.int8_output:
                self.fc_cls.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
                self.fc_reg.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
        self.query_embedding.qconfig = None

    def set_calibration_qconfig(self):
        if self.set_int16_qconfig:
            self.reg_ffn.qconfig = (
                horizon.quantization.get_default_calib_qconfig(dtype="qint16")
            )
            self.activate.qconfig = (
                horizon.quantization.get_default_calib_qconfig(dtype="qint16")
            )
            if hasattr(self.transformer, "set_calibration_qconfig"):
                self.transformer.set_calibration_qconfig()
