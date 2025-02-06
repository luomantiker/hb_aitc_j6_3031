# Copyright (c) Horizon Robotics. All rights reserved.

import math
from distutils.version import LooseVersion
from typing import Dict, List, Tuple

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.detr3d.head import FFN
from hat.models.task_modules.detr.transformer import MultiheadAttention
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["PETRHead", "PETRTransformer", "PETRDecoder"]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x : The tensor to do the
                inverse.
        eps : EPS avoid numerical
                overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
                function of sigmoid, has same
                shape with input.
    """

    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pos2posemb3d(
    pos: Tensor, num_pos_feats: int = 128, temperature: int = 10000
) -> Tensor:
    """Convert the position tensor into a positional embedding in 3D space.

    Args:
        pos : The position tensor.
        num_pos_feats : The number of positional embedding features.
        temperature : The temperature parameter.

    Returns:
        The positional embedding tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class PETRDecoderLayer(nn.Module):
    """PETR decoder layer module.

    Args:
        num_heads: Number of heads for self attention.
        embed_dims: Embeding dims for output.
        feedforward_channels: Feedforward channels of ffn.
        ffn_drop: Drop prob.
        dropout: Dropout for self attention.
        num_views: Number of views for input.
    """

    def __init__(
        self,
        num_heads: int = 8,
        embed_dims: int = 256,
        feedforward_channels: int = 512,
        ffn_drop: float = 0.1,
        dropout: float = 0.1,
        num_views: int = 6,
    ):
        super(PETRDecoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.dropout1_add = FloatFunctional()
        self.dropout2_add = FloatFunctional()

        self.pos_add1 = FloatFunctional()
        self.pos_add2 = FloatFunctional()
        self.pos_add3 = FloatFunctional()

        self.norm1 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )
        self.norm2 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )
        self.norm3 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )

        self.self_attns = MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=dropout,
            bias=True,
        )

        self.corss_attn = MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=dropout,
            bias=True,
        )

        self.ffn = FFN(
            embed_dims=self.embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=ffn_drop,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            query : The query tensor.
            key : The key tensor.
            value : The value tensor.
            query_pos : The query positional tensor.
            key_pos : The key positional tensor.

        Returns:
            The output tensor.
        """
        tgt = query
        query_pos_embed = self.pos_add1.add(query, query_pos)
        tgt2, _ = self.self_attns(
            query=query_pos_embed, key=query_pos_embed, value=query
        )
        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        query_pos_embed = self.pos_add2.add(tgt, query_pos)
        key_pos_embed = self.pos_add3.add(key, key_pos)
        tgt2, _ = self.corss_attn(
            query=query_pos_embed,
            key=key_pos_embed,
            value=value,
        )
        tgt = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        tgt = self.ffn(tgt)
        tgt = self.norm3(tgt)
        return tgt

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        self.ffn.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        self.corss_attn.matmul.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )

        self.self_attns.matmul.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )

        self.self_attns.softmax.min_sub_out = -18.0
        self.corss_attn.softmax.min_sub_out = -18.0


@OBJECT_REGISTRY.register
class PETRDecoder(nn.Module):
    """PETR decoder module.

    Args:
        num_layer: Number of layers.
    """

    def __init__(self, num_layer: int = 6, **kwargs):
        super(PETRDecoder, self).__init__()

        self.decode_layers = nn.ModuleList()

        for _ in range(num_layer):
            self.decode_layers.append(PETRDecoderLayer(**kwargs))

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
    ) -> List[Tensor]:
        """Forward pass of the module.

        Args:
            query : The query tensor.
            key : The key tensor.
            value : The value tensor.
            query_pos : The query positional tensor.
            key_pos : The key positional tensor.

        Returns:
            The output tensors for each decode layer.
        """

        outs = []
        for decode_layer in self.decode_layers:
            query = decode_layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
            )
            outs.append(query)
        return outs

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        for decode_layer in self.decode_layers:
            decode_layer.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        for decode_layer in self.decode_layers:
            decode_layer.set_qconfig()


@OBJECT_REGISTRY.register
class PETRTransformer(nn.Module):
    """Petr Transformer module.

    Args:
        decoder: Decoder module for PETR.
    """

    def __init__(
        self,
        decoder: nn.Module,
    ):
        super(PETRTransformer, self).__init__()
        self.decoder = decoder
        self.tgt_quant = QuantStub()

    def forward(
        self,
        feats: Tensor,
        query_embed: Tensor,
        pos_embed: Tensor,
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            feats : The input feature tensor.
            query_embed : The query embedding tensor.
            pos_embed : The positional embedding tensor.

        Returns:
            The output tensor.
        """
        target = torch.zeros_like(query_embed)
        if LooseVersion(horizon.__version__) < LooseVersion("2.3.4"):
            target = self.tgt_quant(target)
        feats = self.decoder(
            query=target,
            key=feats,
            value=feats,
            query_pos=query_embed,
            key_pos=pos_embed,
        )
        return feats

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        self.decoder.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        self.decoder.set_qconfig()


@OBJECT_REGISTRY.register
class PETRHead(nn.Module):
    """Petr Head module.

    Args:
        transformer: Transformer module for Detr3d.
        num_query: Number of query.
        query_align: Align number for query.
        embed_dims: Embeding channels.
        in_channels: Input channels.
        num_cls_fcs: Number of classification layer.
        num_reg_fcs: Number of classification layer.
        reg_out_channels: Number of regression outoput channels.
        cls_out_channels: Numbger of classification output channels,
        position_range: Positon ranges
        bev_range: BEV ranges.
        num_views: Number of views for input.
        depth_num: Number of max depth.
        depth_start: start of depth.
        positional_encoding: PE module.
        int8_output: Whether output is int8.
        dequant_output: Whether dequant output.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_query: int = 900,
        query_align: int = 8,
        embed_dims: int = 256,
        in_channels: int = 2048,
        num_cls_fcs: int = 2,
        num_reg_fcs: int = 2,
        reg_out_channels: int = 10,
        cls_out_channels: int = 10,
        position_range: Tuple[float] = None,
        bev_range: Tuple[float] = None,
        num_views: int = 6,
        depth_num: int = 64,
        depth_start: int = 1,
        positional_encoding: nn.Module = None,
        int8_output: bool = False,
        dequant_output: bool = True,
    ):
        super(PETRHead, self).__init__()
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs

        self.num_query_h = int(np.ceil(num_query / query_align))
        self.num_query_w = query_align

        self.num_query = self.num_query_h * self.num_query_w
        self.embed_dims = embed_dims
        self.in_channels = in_channels

        self.reg_out_channels = reg_out_channels
        self.cls_out_channels = cls_out_channels

        self.num_views = num_views

        self.depth_num = depth_num
        self.depth_start = depth_start

        self.positional_encoding = positional_encoding
        self.transformer = transformer
        self.bev_range = bev_range
        self.position_range = position_range

        self.int8_output = int8_output
        self.dequant_output = dequant_output

        self._init_layers()
        self._init_weight()

    def _init_weight(self) -> None:
        """Initialize the weights."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_branch[-1], std=0.01, bias=bias_cls)

        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    def _init_cls_branch(self) -> None:
        """Initialize the classification branch."""
        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(
                nn.Conv2d(
                    in_channels=self.embed_dims,
                    out_channels=self.embed_dims,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                )
            )
            cls_branch.append(
                LayerNorm2d(normalized_shape=[self.embed_dims, 1, 1], dim=1)
            )
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(
            nn.Conv2d(
                in_channels=self.embed_dims,
                out_channels=self.cls_out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
        )
        return nn.Sequential(*cls_branch)

    def _init_reg_branch(self) -> None:
        """Initialize the regression branch."""
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(
                ConvModule2d(
                    in_channels=self.embed_dims,
                    out_channels=self.embed_dims,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        reg_branch.append(
            nn.Conv2d(
                in_channels=self.embed_dims,
                out_channels=self.reg_out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
        )
        return nn.Sequential(*reg_branch)

    def _init_encoder(self) -> None:
        """Initialize layers of the encoder."""
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(
                self.embed_dims * 3 // 2,
                self.embed_dims * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.embed_dims * 4,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(
                self.depth_num * 3,
                self.embed_dims * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.embed_dims * 4,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.query_embedding = nn.Sequential(
            nn.Conv2d(
                self.embed_dims * 3 // 2,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.embed_dims,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.qurey_quant = QuantStub()
        self.pos_quant = QuantStub()
        self.cls_dequant = DeQuantStub()
        self.reg_dequant = DeQuantStub()

        self.input_project = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1
        )

        self.cls_branch = self._init_cls_branch()
        self.reg_branch = self._init_reg_branch()
        self._init_encoder()

        self.reference_points = nn.Embedding(self.num_query, 3)

        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

        self.query_embed_quant = QuantStub(scale=None)
        self.pos_embed_quant = QuantStub(scale=None)

    def position_embeding(
        self, feat_hw: Tuple[int, int], meta: Dict
    ) -> Tensor:
        """Perform position embedding for the input feature map.

        Args:
            feat : The input feature tensor.
            meta : A dictionary containing additional information,
                   such as the shape of the image tensor.
        Returns:
            The position embedding tensor.
        """
        _, _, img_H, img_W = meta["img"].shape
        H, W = feat_hw
        img = meta["img"]

        coords_h = torch.arange(H, device=img.device).float() * img_H / H
        coords_w = torch.arange(W, device=img.device).float() * img_W / W
        index = torch.arange(
            start=0, end=self.depth_num, step=1, device=img.device
        ).float()
        bin_size = (self.position_range[3] - self.depth_start) / (
            self.depth_num * (self.depth_num + 1)
        )
        coords_d = self.depth_start + bin_size * index * (index + 1)
        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, coords_d)
        ).permute(1, 2, 3, 0)
        coords_shape = coords.shape[:3]

        coords = torch.cat(
            [coords, torch.ones((*coords_shape, 1), device=img.device)], -1
        )
        eps = 1e-5
        coords[..., :2] = coords[..., :2] * torch.clamp(
            coords[..., 2:3], min=eps
        )
        homography = meta["ego2img"]
        coords3d = []
        for homo in homography:
            homo = np.linalg.inv(homo.cpu().numpy())
            homo = torch.tensor(homo, device=img.device)

            coord3d = torch.matmul(
                coords.double(), homo.permute(1, 0).double()
            ).float()
            coords3d.append(coord3d)
        coords3d = torch.stack(coords3d)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0]
        )
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1]
        )
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2]
        )

        coords3d = (
            coords3d.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(-1, 3 * self.depth_num, H, W)
        )
        coords3d = inverse_sigmoid(coords3d)
        return coords3d

    @fx_wrap()
    def _build_transformer_input(
        self, feats: List[Tensor], meta: Dict, compile_model: bool
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build the input for the transformer module.

        Args:
            feats : The list of feature tensors.
            meta : The metadata dictionary.
        Returns:
            The input tensors for the transformer module.
        """
        feat = feats[-1]
        n, _, h, w = feat.shape
        bs = n // self.num_views

        if compile_model is True:
            pos_embed = meta["pos_embed"]
        else:
            pos_embed = self._gen_pe(meta, feat.shape[2:])

        feat = self.input_project(feat)

        reference_points = self.reference_points.weight
        query_embed = pos2posemb3d(
            reference_points, num_pos_feats=self.embed_dims // 2
        )
        query_embed = (
            query_embed.permute(1, 0)
            .contiguous()
            .view(1, -1, self.num_query_h, self.num_query_w)
            .repeat(bs, 1, 1, 1)
        )
        query_embed = self.query_embedding(query_embed)
        feat = (
            feat.view(bs, self.num_views, -1, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, -1, self.num_views * h, w)
        )

        query_embed = self.qurey_quant(query_embed)
        pos_embed = self.pos_quant(pos_embed)

        reference_points = reference_points.view(
            1, self.num_query_h, self.num_query_w, -1
        ).repeat(bs, 1, 1, 1)
        reference_points = inverse_sigmoid(reference_points)
        return feat, query_embed, pos_embed, reference_points

    def _gen_pe(self, meta: Dict, feat_hw: Tuple[int, int]) -> Tensor:
        """Generate the position embeddings.

        Args:
           meta: The metadata.
           feat_hw: The feature height and width.
        Returns:
           The generated position embeddings.
        """
        h, w = feat_hw

        device = meta["img"].device
        coords3d = self.position_embeding(feat_hw, meta)
        bs = coords3d.shape[0] // self.num_views
        masks = torch.zeros((bs, self.num_views, h, w), device=device).to(
            torch.bool
        )
        pos_embed = self.position_encoder(coords3d)
        sin_embed = self.positional_encoding(masks)
        sin_embed = sin_embed.flatten(0, 1)
        sin_embed = self.adapt_pos3d(sin_embed)
        pos_embed = pos_embed + sin_embed
        _, c, h, w = pos_embed.shape
        pos_embed = (
            pos_embed.view(-1, self.num_views, c, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, c, self.num_views * h, w)
        )
        return pos_embed

    def forward(
        self, feats: List[Tensor], meta: Dict, compile_model: bool = False
    ) -> Tuple[Tensor]:
        """Represent the forward pass of the module.

        Args:
            feats : The list of feature tensors.
            meta : The metadata dictionary.
            compile_model: Whether in compile model.
        Returns:
            The output result list
        """
        (
            feat,
            query_embed,
            pos_embed,
            reference_points,
        ) = self._build_transformer_input(feats, meta, compile_model)

        feats = self.transformer(feat, query_embed, pos_embed)
        return self._build_res_list(feats), reference_points

    def export_reference_points(self, meta: Dict, feat_hw: Tuple[int, int]):
        """Export the reference points.

        Args:
            meta: Additional metadata.
            feat_hw: The feature height and width.

        Returns:
            A dictionary containing the position embeddings
            and reference points.
        """
        pos_embed = self._gen_pe(meta, feat_hw)
        reference_points = self.reference_points.weight
        reference_points = reference_points.view(
            1, self.num_query_h, self.num_query_w, -1
        )
        reference_points = inverse_sigmoid(reference_points)
        output = {"pos_embed": pos_embed, "reference_points": reference_points}
        return output

    @fx_wrap()
    def _build_res_list(
        self,
        feats: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Build the list of output tensors.

        Args:
            feats : The list of feature tensors.

        Returns:
            The list of output tensors for classification
            and regression branches.
        """
        cls_list = []
        reg_list = []
        for feat in feats:
            cls = self.cls_branch(feat)
            cls = self.cls_dequant(cls)
            reg = self.reg_branch(feat)
            reg = self.reg_dequant(reg)
            cls_list.append(cls)
            reg_list.append(reg)
        return cls_list, reg_list

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        for i in range(self.num_reg_fcs):
            self.reg_branch[i].fuse_model()
        self.transformer.fuse_model()

    def set_calibration_qconfig(self):
        """Set the calibration quantization configuration."""
        self.adapt_pos3d.qconfig = None
        self.position_encoder.qconfig = None
        self.query_embedding.qconfig = None

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.adapt_pos3d.qconfig = None
        self.position_encoder.qconfig = None
        self.query_embedding.qconfig = None
        self.reference_points.qconfig = None
        self.query_embed.qconfig = None
        self.transformer.set_qconfig()
        # disable output quantization for last quanti layer.

        if not self.int8_output:
            self.cls_branch[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.reg_branch[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
