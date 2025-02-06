# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import horizon_plugin_pytorch.nn as hnn
import torch
from horizon_plugin_pytorch.qtensor import QTensor
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = [
    "ViewFusion",
    "ViewFusionIrInfer",
    "ViewFusion4DIrInfer",
]


@OBJECT_REGISTRY.register
class ViewFusion(nn.Module):
    """The basic structure of bev.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        view_transformer:
            View transformer module for transforming from img view to bev view.
        aux_heads: List of auxiliary heads for training.
        bev_encoder:
            Encoder for the feature of bev view.
            If set to None, bev feature is used for decoders directly.
        bev_decoders:
            Decoder for bev feature.
        bev_feat_index:
            Index for bev feats. Default 0.
        bev_transforms:
            Transfomrs for bev traning.
        bev_upscale:
            Upscale parameter for bec feature.
        compile_model:
            Whether in compile model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        view_transformer: nn.Module = None,
        temporal_fusion: nn.Module = None,
        aux_heads: List[nn.Module] = None,
        bev_encoder: Optional[nn.Module] = None,
        bev_decoders: List[nn.Module] = None,
        bev_feat_index: int = 0,
        bev_transforms: Optional[List] = None,
        bev_upscale: int = 2,
        compile_model: bool = False,
    ):
        super(ViewFusion, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.view_transformer = view_transformer
        self.temporal_fusion = temporal_fusion
        self.aux_heads = nn.ModuleList(aux_heads)
        self.bev_encoder = bev_encoder
        self.bev_decoders = nn.ModuleList(bev_decoders)
        self.bev_feat_index = bev_feat_index
        self.bev_transforms = bev_transforms
        if self.bev_transforms is None:
            self.bev_transforms = []

        self.bev_upscale = bev_upscale
        if self.bev_upscale > 1:
            self.resize = hnn.Interpolate(
                scale_factor=self.bev_upscale,
                align_corners=None,
                recompute_scale_factor=True,
            )
        self.compile_model = compile_model

    def img_encode(self, img: Tensor) -> Tensor:
        """Encode the input image and returns the encoded features.

        Args:
            img: The input image to be encoded.

        Returns:
            feats: The encoded features of the input image.
        """

        feats = self.backbone(img)
        feats = self.neck(feats)
        return feats

    def forward(self, data: Dict) -> Tuple[Dict, Dict]:
        """Perform the forward pass of the model.

        Args:
            data: A dictionary containing the input data,
                  including the image and other relevant information.

        Returns:
            preds: The predictions of the model.
            results: A dictionary containing the results of the model.
        """

        img = data["img"]
        feat = self.img_encode(img)[self.bev_feat_index]
        bev_feat, aux_data = self.view_transformer(
            feat, data, self.compile_model
        )

        aux_res = None
        for aux_head in self.aux_heads:
            result = aux_head(aux_data, data)
            aux_res = self._update_aux(aux_res, result)

        if self.temporal_fusion:
            bev_feat, prev_feat = self.temporal_fusion(
                bev_feat, data, self.compile_model
            )

        if self.bev_upscale > 1:
            bev_feat = self.resize(bev_feat)

        bev_feat, data = self._transform(bev_feat, data)
        if self.bev_encoder:
            bev_feat = self.bev_encoder(bev_feat, data)
        if not isinstance(bev_feat, Sequence):
            bev_feat = [bev_feat]
        preds = None
        results = {}
        for bev_decoder in self.bev_decoders:
            pred, result = bev_decoder(bev_feat, data)
            preds, results = self._update_res(pred, result, preds, results)
        if self.compile_model is True:
            if self.temporal_fusion:
                return preds, prev_feat
            else:
                return preds
        if aux_data is not None:
            return preds, results, aux_res
        return preds, results

    @fx_wrap()
    def _transform(self, bev_feat: Tensor, data: Dict) -> Tuple[Tensor, Dict]:
        """Apply transformations to the bev_feat and data.

        Args:
            bev_feat: The encoded or transformed features of the input image.
            data: A dictionary containing the input data.

        Returns:
            bev_feat: The transformed features.
            data: The updated data after applying transformations.
        """

        if self.training and not isinstance(bev_feat, QTensor):
            for transform in self.bev_transforms:
                bev_feat, data = transform(bev_feat, data)
        return bev_feat, data

    @fx_wrap()
    def _update_aux(self, aux_res, result):
        if aux_res is None:
            aux_res = result
        else:
            aux_res.update(result)
        return aux_res

    @fx_wrap()
    def _update_res(
        self, pred: List, result: Dict, preds: Dict, results: Dict
    ) -> Tuple[Dict, Dict]:
        """Update the predictions and results.

        Args:
            pred: The predicted values from the model.
            result: The results from the model.
            preds: The existing predictions.
            results: The existing results.

        Returns:
            preds: The updated predictions.
            results: The updated results.
        """

        if preds is None:
            preds = pred
        else:
            preds.extend(pred)

        if result:
            results.update(result)
        return preds, results

    def export_reference_points(
        self, data: Dict, feat_wh: Tuple[int, int]
    ) -> Dict:
        """Export refrence points.

        Args:
            data: A dictionary containing the input data.
            feat_wh: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """

        return self.view_transformer.export_reference_points(data, feat_wh)

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
            self.temporal_fusion,
            self.bev_encoder,
            *self.bev_decoders,
        ]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
            self.temporal_fusion,
            self.bev_encoder,
            *self.bev_decoders,
            *self.aux_heads,
        ]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class ViewFusionIrInfer(nn.Module):
    """
    The basic structure of ViewFusionIrInfer.

    Args:
        ir_model: The ir model.
        deploy_model: The deploy model to generate refpoints.
        model_convert_pipeline: Define the process of model convert.
        vt_input_hw: Feature map shape.
        bev_decoder_infers: bev_decoder_infers module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module = None,
        model_convert_pipeline: List[callable] = None,
        vt_input_hw: List[int] = None,
        bev_decoder_infers: List[nn.Module] = None,
    ):
        super(ViewFusionIrInfer, self).__init__()
        self.ir_model = ir_model
        self.deploy_model = deploy_model
        if model_convert_pipeline is not None:
            self.deploy_model = model_convert_pipeline(deploy_model)
            self.deploy_model.eval()
        self.bev_decoder_infers = bev_decoder_infers
        self.vt_input_hw = vt_input_hw

    def process_input(self, data):
        if self.deploy_model is None:
            return data
        ref_p = self.deploy_model.export_reference_points(
            data, self.vt_input_hw
        )
        data.update(ref_p)
        return data

    def process_output(self, outputs):
        return outputs

    def forward(self, data):
        inputs = self.process_input(data)
        hbir_outputs = self.ir_model(inputs)
        outputs = self.process_output(hbir_outputs)
        idx = 0
        results = {}
        for decoder_infer in self.bev_decoder_infers:
            num = decoder_infer.input_num()
            decoder_input = outputs[idx : idx + num]
            _, ret = decoder_infer(decoder_input, data)
            results.update(ret)
            idx = idx + num
        return None, results


@OBJECT_REGISTRY.register
class ViewFusion4DIrInfer(ViewFusionIrInfer):
    """
    The basic structure of ViewFusion4DIrInfer.

    Args:
        bev_size: The deploy model to generate refpoints.
        in_channels: Define the process of model convert.
        num_views: Feature map shape.
        kwargs: As same ViewFusionIrInfer docstring.
    """

    def __init__(
        self,
        bev_size: List,
        in_channels: int,
        num_views: int,
        **kwargs,
    ):
        super(ViewFusion4DIrInfer, self).__init__(**kwargs)
        prev_tensor = torch.zeros(
            (
                2,
                in_channels,
                int((bev_size[0] * 2) / bev_size[2]),
                int((bev_size[1] * 2) / bev_size[2]),
            )
        )
        self.prev_feats = nn.Parameter(prev_tensor, requires_grad=False)

        prev_p_tensor = torch.zeros(
            (
                2,
                int((bev_size[0] * 2) / bev_size[2]),
                int((bev_size[1] * 2) / bev_size[2]),
                2,
            )
        )
        self.prev_points = nn.Parameter(prev_p_tensor, requires_grad=False)
        self.num_views = num_views

    def process_input(self, data):
        ref_p = self.deploy_model.export_reference_points(
            data, self.vt_input_hw
        )

        for k, v in ref_p.items():
            ref_p[k] = v[: self.num_views]
        image = data["img"][: self.num_views]
        prev_points = (
            self.deploy_model.temporal_fusion.export_reference_points(
                data["img"], data
            )
        )
        self.prev_points.data = torch.cat(
            [prev_points["prev_points"], self.prev_points[0:1]]
        )

        result = {
            "img": image,
            "prev_feats": self.prev_feats,
            "prev_points": self.prev_points,
        }
        result.update(ref_p)
        return result

    def process_output(self, outputs):
        prev_feat = outputs[-1]
        self.prev_feats.data = torch.cat([prev_feat, self.prev_feats[0:1]])
        return outputs[0]
