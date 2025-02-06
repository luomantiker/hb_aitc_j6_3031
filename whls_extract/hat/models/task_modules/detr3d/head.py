# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional, Tuple

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    QuantStub,
)
from torch import Tensor
from torch.quantization import DeQuantStub, QConfig

from hat.core.nus_box3d_utils import adjust_coords
from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import (
    bias_init_with_prob,
    constant_init,
    normal_init,
    xavier_init,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

try:
    from hbdk.torch_script.placeholder import placeholder
except ImportError:
    placeholder = None

__all__ = ["Detr3dHead", "Detr3dTransformer", "Detr3dDecoder"]

grid_qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=qinfo("qint16").min,
        quant_max=qinfo("qint16").max,
        dtype="qint16",
        saturate=True,
    ),
    weight=None,
)


class FFN(nn.Module):
    """Detr3d FFN module.

    Args:
        embed_dims: Embeding dims for output.
        feedforward_channels: Feedforward channels of ffn.
        ffn_drop=0.1: Drip prob.
    """

    def __init__(self, embed_dims=256, feedforward_channels=512, ffn_drop=0.1):
        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            ConvModule2d(
                in_channels=embed_dims,
                out_channels=feedforward_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.Conv2d(
                in_channels=feedforward_channels,
                out_channels=embed_dims,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
        )
        self.shortcut_add = FloatFunctional()
        self.dropout = nn.Dropout2d(ffn_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        Args:
            x: The input tensor.

        Returns:
            The output of the forward pass.
        """
        shortcut = x
        x = self.ffn(x)
        return self.shortcut_add.add(self.dropout(x), shortcut)

    def init_weights(self) -> None:
        """Initialize the weights."""
        for mod in self.ffn:
            if isinstance(mod, ConvModule2d):
                xavier_init(mod[0], distribution="uniform", bias=0.0)

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        from horizon_plugin_pytorch import quantization

        self.ffn[0].fuse_model()
        torch.quantization.fuse_modules(
            self,
            ["ffn.1", "shortcut_add"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class FeatureSampler(nn.Module):
    """Detr3d featureSampler module.

    Args:
        num_views: Number of views.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.
    """

    def __init__(
        self,
        num_views: int = 6,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scales=None,
    ):
        super(FeatureSampler, self).__init__()

        if grid_quant_scales is None:
            grid_quant_scales = [1 / 128, 1 / 128, 1 / 128, 1 / 128]
        self.quant_stubs = nn.ModuleList()

        self.grid_samples = nn.ModuleList()
        self.cat = FloatFunctional()
        self.num_views = num_views
        self.masks_quant = QuantStub(scale=None)
        for i in range(len(grid_quant_scales)):
            self.quant_stubs.append(QuantStub(scale=grid_quant_scales[i]))
            self.grid_samples.append(
                hnn.GridSample(
                    mode=mode,
                    padding_mode=padding_mode,
                )
            )

    def _get_homography(
        self, homography: Tensor, feat_hw: Tuple[int, int], meta: Dict
    ) -> Tensor:
        """Modify the given homography matrix based on the feature and image shapes.

        Args:
            homography: Homography matrix.
            feats: Feature tensor.
            meta: Metadata dictionary.

        Returns:
            homopgrahpy: Modified homography matrix.
        """
        orig_hw = meta["img"][0].shape[1:]
        scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        view = np.eye(4)
        view[0, 0] = scales[1]
        view[1, 1] = scales[0]
        view = torch.tensor(view).to(device=homography.device)
        homography = torch.matmul(view.double(), homography.double())
        return homography

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        for quant_stub in self.quant_stubs:
            quant_stub.qconfig = grid_qconfig

    def _get_points(
        self,
        feat_hw: Tuple[int, int],
        homography: Tensor,
        reference_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get the new coordinates of the reference points after applying homography.

        Args:
            feat_hw: view transformer input shape
                     for generationg reference points.
            homography: The homography matrix.
            reference_points: The reference points.

        Returns:
            The new coordinates of the reference points
            and a mask indicating which points are within
            the feature tensor bounds.
        """
        new_coords = []
        for homo in homography:
            # Apply homography to reference points
            new_coord = torch.matmul(
                reference_points.double(),
                homo.permute(1, 0).double(),
            ).float()
            new_coords.append(new_coord)
        new_coords = torch.concat(new_coords, dim=0)
        # Clamp z-coordinate to a minimum value
        new_coords[..., 2] = torch.clamp(new_coords[..., 2], min=0.05)
        # Calculate x and y coordinates
        X = new_coords[..., 0] / new_coords[..., 2]
        Y = new_coords[..., 1] / new_coords[..., 2]
        new_coords = torch.stack((X, Y), dim=-1)

        H, W = feat_hw

        # Create a mask for points within feature tensor bounds
        mask = (
            (new_coords[..., 0] >= 0)
            & (new_coords[..., 0] < W)
            & (new_coords[..., 1] >= 0)
            & (new_coords[..., 1] < H)
        )
        return new_coords, mask

    def _gen_3d_points(
        self, bev_range: Tuple[int, int, int], reference_points: Tensor
    ) -> Tensor:
        """Generate reference_points for 3d.

        Args:
            bev_range : The BEV (Bird's Eye View) range.
            reference_points: The reference points tensor.

        Returns:
            The reference point for 3d.
        """
        with torch.no_grad():
            reference_points_3d = reference_points.sigmoid()
            reference_points_3d = reference_points_3d.permute(0, 2, 3, 1)
            reference_points_3d[..., 0] = (
                reference_points_3d[..., 0] * (bev_range[3] - bev_range[0])
                + bev_range[0]
            )
            reference_points_3d[..., 1] = (
                reference_points_3d[..., 1] * (bev_range[4] - bev_range[1])
                + bev_range[1]
            )
            reference_points_3d[..., 2] = (
                reference_points_3d[..., 2] * (bev_range[5] - bev_range[2])
                + bev_range[2]
            )
            reference_points_3d = torch.cat(
                (
                    reference_points_3d,
                    torch.ones(
                        reference_points_3d[..., :1].shape,
                        device=reference_points_3d.device,
                    ),
                ),
                -1,
            ).double()
        return reference_points_3d

    def _gen_coords(
        self,
        feats: List[Tensor],
        reference_points: Tensor,
        bev_range: Tuple[float],
        homography: Tensor,
        meta: Dict,
    ) -> Tuple[List[Tensor], Tensor]:
        """Generate coordinates and masks based on the given inputs.

        Args:
            feats: A list of input features.
            reference_points: The reference points.
            bev_range: The range of the bird's eye view.
            homography: The homography matrix.
            meta: Additional metadata.

        Returns:
            The generated coordinates and masks.
        """
        reference_points_3d = self._gen_3d_points(bev_range, reference_points)
        H = reference_points.shape[2]
        W = reference_points.shape[3]
        masks = []
        coords = []
        for feat in feats:
            with torch.no_grad():
                homography_tmp = self._get_homography(
                    homography, feat.shape[2:], meta
                )
                new_coords, mask = self._get_points(
                    feat.shape[2:], homography_tmp, reference_points_3d
                )
                new_coords = adjust_coords(new_coords, (W, H))
            coords.append(new_coords)
            masks.append(mask)
        masks = torch.stack(masks, dim=1).float()
        N = feats[0].shape[0]
        bs = N // self.num_views
        masks = masks.view(bs, -1, H, W).float()
        return coords, masks

    def export_reference_points(
        self, meta, feats_hw, bev_range, reference_points
    ) -> Tuple[List[Tensor], Tensor]:
        """Export the reference points.

        Args:
            meta: Additional metadata.
            feats_hw: A list of feature heights and widths.
            bev_range: The range of the bird's eye view.
            reference_points: The reference points.
        Returns:
            The exported coordinates and masks.
        """

        reference_points_3d = self._gen_3d_points(bev_range, reference_points)
        H = reference_points.shape[2]
        W = reference_points.shape[3]
        masks = []
        coords = []
        homography = meta["ego2img"]
        for feat_hw in feats_hw:
            homography_tmp = self._get_homography(homography, feat_hw, meta)
            new_coords, mask = self._get_points(
                feat_hw, homography_tmp, reference_points_3d
            )
            new_coords = adjust_coords(new_coords, (W, H))
            coords.append(new_coords)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        masks = masks.view(-1, self.num_views * len(feats_hw), H, W).float()
        return coords, masks

    def _get_from_meta(self, meta: Dict) -> Tuple[List[Tensor], Tensor]:
        """Retrieve the coordinates and masks from the meta dictionary.

        Args:
            meta: The meta dictionary.

        Returns:
            A tuple containing the coordinates and masks.
        """
        coords = []
        for k, v in meta.items():
            if k.startswith("coords"):
                if placeholder is not None and isinstance(v, placeholder):
                    v = v.sample
                coords.append(v)
        masks = meta["masks"]
        return coords, masks

    @fx_wrap()
    def forward(
        self,
        feats: Tensor,
        reference_points: Tensor,
        bev_range: List[float],
        meta: Dict,
        compile_model,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the module.

        Args:
            feats : The feature tensor.
            reference_points : The reference points tensor.
            bev_range : The BEV (Bird's Eye View) range.
            meta : The metadata dictionary.

        Returns:
            The BEV features tensor.
            The masks tensor.
        """
        if compile_model is False:
            homography = meta["ego2img"]
            coords, masks = self._gen_coords(
                feats, reference_points, bev_range, homography, meta
            )
        else:
            coords, masks = self._get_from_meta(meta)

        bev_feats = []
        for quant_stub, grid_sample, feat, coord in zip(
            self.quant_stubs, self.grid_samples, feats, coords
        ):
            bev_feat = grid_sample(feat, quant_stub(coord))
            bev_feats.append(bev_feat)
        bev_feats = self.cat.cat(bev_feats, dim=1)
        masks = self.masks_quant(masks)
        return bev_feats, masks


class Detr3dCrossAtten(nn.Module):
    """Detr3d Cross attention module.

    Args:
        embed_dims: Embeding dims for output.
        num_levels: Number of levels for multiscale inputs.
        num_views: Number of views for input.
        num_points: Number of points for corss attention.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_levels: int = 4,
        num_views: int = 6,
        num_points: int = 5,
    ):
        super(Detr3dCrossAtten, self).__init__()
        self.num_views = num_views
        self.num_levels = num_levels
        self.num_points = num_points
        self.attention = ConvModule2d(
            in_channels=embed_dims,
            out_channels=num_views * num_levels * num_points,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.out_proj = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.position_encoder = nn.Sequential(
            ConvModule2d(
                in_channels=3,
                out_channels=embed_dims,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1),
            nn.ReLU(inplace=True),
            ConvModule2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.attn_mul = FloatFunctional()
        self.attn_sum = FloatFunctional()
        self.shortcut_add = FloatFunctional()
        self.pos_add = FloatFunctional()
        self.masks_mul = FloatFunctional()

    def init_weights(self) -> None:
        """Initialize the weights."""
        constant_init(self.attention[0], val=0.0, bias=0.0)
        xavier_init(self.out_proj[0], distribution="uniform", bias=0.0)
        for mod in self.position_encoder:
            if isinstance(mod, ConvModule2d):
                xavier_init(mod[0], distribution="uniform", bias=0.0)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            query : The query tensor.
            value : The value tensor.
            reference_points : The reference points tensor.
            masks : The masks tensor.

        Returns:
            The output tensor.
        """
        residual = query
        N, C, H, W = value.shape
        C = C // self.num_levels
        bs = N // self.num_views
        attention = self.attention(query)
        value = value.view(bs, -1, C, H, W)
        attention = self.sigmoid(attention)
        masks = masks.view(bs, -1, H, W)
        attention = self.masks_mul.mul(attention, masks)
        attention = attention.view(bs, -1, 1, H, W)
        output = self.attn_mul.mul(attention, value)
        output = self.attn_sum.sum(output, dim=1, keepdim=True)
        output = output.view(bs, C, H, W)
        output = self.out_proj(output)
        pos_enc = self.position_encoder(reference_points)
        output = self.pos_add.add(output, pos_enc)
        output = self.shortcut_add.add(output, residual)
        return output


class Detr3dDecoderLayer(nn.Module):
    """Detr3d decoder layer module.

    Args:
        num_heads: Number of heads for self attention.
        embed_dims: Embeding dims for output.
        feedforward_channels: Feedforward channels of ffn.
        ffn_drop: Drop prob.
        dropout: Dropout for self attention.
        num_levels: Number of levels for multiscale inputs.
        num_views: Number of views for input.
        num_points: Number of points for corss attention.
    """

    def __init__(
        self,
        num_heads: int = 8,
        embed_dims: int = 256,
        feedforward_channels: int = 512,
        ffn_drop: float = 0.1,
        dropout: float = 0.1,
        num_levels: int = 4,
        num_views: int = 6,
        num_points: int = 5,
    ):
        super(Detr3dDecoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_views = num_views
        self.num_heads = num_heads
        self.num_points = num_points

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.dropout1_add = FloatFunctional()
        self.dropout2_add = FloatFunctional()

        self.pos_add1 = FloatFunctional()
        self.pos_add2 = FloatFunctional()
        self.pos_add3 = FloatFunctional()

        self.self_attns = MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm1 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )

        self.pos_add2 = FloatFunctional()
        self.cross_attn = Detr3dCrossAtten(
            embed_dims=self.embed_dims,
            num_levels=self.num_levels,
            num_views=self.num_views,
            num_points=self.num_points,
        )
        self.norm2 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )
        self.ffn = FFN(
            embed_dims=self.embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=ffn_drop,
        )
        self.norm3 = LayerNorm2d(
            normalized_shape=[self.embed_dims, 1, 1], dim=1
        )

    def init_weights(self):
        if hasattr(self.ffn, "init_weights"):
            self.ffn.init_weights()
        if hasattr(self.cross_attn, "init_weights"):
            self.cross_attn.init_weights()

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            query : The query tensor.
            value : The value tensor.
            query_pos : The positional encoding of the query tensor.
            reference_points : The reference points tensor.
            masks : The masks tensor.

        Returns:
            The output tensor.
        """
        tgt = query
        query_pos_embed = self.pos_add1.add(query, query_pos)
        tgt2, _ = self.self_attns(
            query=query_pos_embed, key=query_pos_embed, value=tgt
        )
        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        tgt = self.pos_add2.add(tgt, query_pos)
        tgt2 = self.cross_attn(
            query=tgt,
            value=value,
            reference_points=reference_points,
            masks=masks,
        )
        tgt = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        tgt = self.ffn(tgt)
        tgt = self.norm3(tgt)
        return tgt

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""

        self.ffn.fuse_model()
        self.self_attns.matmul.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )
        self.cross_attn.attn_mul.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )


@OBJECT_REGISTRY.register
class Detr3dDecoder(nn.Module):
    """Detr3d decoder module.

    Args:
        num_layer: Number of layers.
    """

    def __init__(self, num_layer: int = 6, **kwargs):
        super(Detr3dDecoder, self).__init__()

        self.decode_layers = nn.ModuleList()

        for _ in range(num_layer):
            self.decode_layers.append(Detr3dDecoderLayer(**kwargs))

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        masks: Tensor,
    ) -> List[Tensor]:
        """Forward pass of the module.

        Args:
            query : The query tensor.
            value : The value tensor.
            query_pos : The positional encoding of the query tensor.
            reference_points : The reference points tensor.
            masks : The masks tensor.

        Returns:
            The list of output tensors from each decoding layer.
        """
        outs = []
        for decode_layer in self.decode_layers:
            query = decode_layer(
                query=query,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points,
                masks=masks,
            )
            outs.append(query)
        return outs

    def init_weights(self):
        for decode_layer in self.decode_layers:
            if hasattr(decode_layer, "init_weights"):
                decode_layer.init_weights()

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""

        for decode_layer in self.decode_layers:
            decode_layer.fuse_model()


@OBJECT_REGISTRY.register
class Detr3dTransformer(nn.Module):
    """Detr3d Transfomer module.

    Args:
        decoder: Decoder modules.
        embed_dims: Embeding dims for output.,
        num_views: Number of views for input,
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scales: Quanti scale for grid sample.
        homography: Homegraphy for view transformation.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dims: int = 256,
        num_views: int = 6,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scales: List[float] = None,
        homography: Optional[torch.Tensor] = None,
    ):
        super(Detr3dTransformer, self).__init__()
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_views = num_views
        self.mode = mode
        self.padding_mode = padding_mode
        self.homography = homography
        self.grid_quant_scales = grid_quant_scales
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the layers of the module."""
        self.reference_points = ConvModule2d(
            in_channels=self.embed_dims,
            out_channels=3,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.feature_sample = FeatureSampler(
            num_views=self.num_views,
            mode=self.mode,
            padding_mode=self.padding_mode,
            grid_quant_scales=self.grid_quant_scales,
        )
        self.pos_embed_quant = QuantStub(scale=None)
        self.refer_quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()

    def init_weights(self) -> None:
        """Initialize the weights."""
        # for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)
        xavier_init(self.reference_points[0], distribution="uniform", bias=0.0)
        if hasattr(self.decoder, "init_weights"):
            self.decoder.init_weights()

    def forward(
        self,
        feats: List[Tensor],
        query_embed: Tensor,
        pos_embed: Tensor,
        meta: Dict,
        bev_range: List[float],
        compile_model: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the module.

        Args:
            feats : The feature tensor.
            query_embed : The query embedding tensor.
            pos_embed : The positional embedding tensor.
            meta : The metadata dictionary.
            bev_range : The BEV (Bird's Eye View) range.
            compile_model : A flag indicating whether to
                            use pre-compiled homography matrix
                            or use it from metadata.

        Returns:
            The output tensor and the reference points tensor.
        """
        reference_points = self.reference_points(pos_embed)
        feats, masks = self.feature_sample(
            feats, reference_points, bev_range, meta, compile_model
        )
        pos_embed = self.pos_embed_quant(pos_embed)
        bs = feats.shape[0] // self.num_views
        reference_points = reference_points.repeat(bs, 1, 1, 1)
        feats = self.decoder(
            query=query_embed,
            value=feats,
            query_pos=pos_embed,
            reference_points=self.refer_quant(reference_points),
            masks=masks,
        )
        return feats, reference_points

    def export_reference_points(self, pos_embed, meta, feats_hw, bev_range):
        reference_points = self.reference_points(pos_embed)
        coords, masks = self.feature_sample.export_reference_points(
            meta, feats_hw, bev_range, reference_points
        )
        outputs = {}
        for i in range(len(coords)):
            outputs[f"coords{i}"] = coords[i].detach()
        outputs["masks"] = masks.detach()
        outputs["reference_points"] = (
            reference_points.detach().permute(0, 2, 3, 1).contiguous()
        )
        return outputs

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        self.feature_sample.set_qconfig()
        self.reference_points.qconfig = None

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""

        self.decoder.fuse_model()


@OBJECT_REGISTRY.register
class Detr3dHead(nn.Module):
    """Detr3d Head module.

    Args:
        transformer: Transformer module for Detr3d.
        num_query: Number of query.
        query_align: Align number for query.
        embed_dims: embeding channels.
        num_cls_fcs: Number of classification layer.
        num_reg_fcs: Number of classification layer.
        reg_out_channels: Number of regression outoput channels.
        cls_out_channels: Numbger of classification output channels,
        bev_range: BEV range.
        num_levels: Nunmber of levels for multiscale inputs.
        int8_output: Whether output is int8.
        dequant_output: Whether dequant output.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_query: int = 900,
        query_align: int = 8,
        embed_dims: int = 256,
        num_cls_fcs: int = 2,
        num_reg_fcs: int = 2,
        reg_out_channels: int = 10,
        cls_out_channels: int = 10,
        bev_range: Tuple[float] = None,
        num_levels: int = 4,
        int8_output: bool = False,
        dequant_output: bool = True,
    ):
        super(Detr3dHead, self).__init__()
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs

        self.num_query_h = int(np.ceil(num_query / query_align))
        self.num_query_w = query_align

        self.num_query = self.num_query_h * self.num_query_w
        self.embed_dims = embed_dims
        self.reg_out_channels = reg_out_channels
        self.cls_out_channels = cls_out_channels
        self.num_levels = num_levels
        self.transformer = transformer
        self.bev_range = bev_range

        self.int8_output = int8_output
        self.dequant_output = dequant_output

        self._init_layers()
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights."""
        # for p in self.parameters():
        #   if p.dim() > 1:
        #       nn.init.xavier_uniform_(p)
        bias_cls = bias_init_with_prob(0.01)
        for mod in self.cls_branch[:-1]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        normal_init(self.cls_branch[-1][0], std=0.01, bias=bias_cls)

        for mod in self.reg_branch:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        if hasattr(self.transformer, "init_weights"):
            self.transformer.init_weights()

    def _init_cls_branch(self) -> None:
        """Initialize the classification branch."""
        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(
                ConvModule2d(
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
            ConvModule2d(
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
            ConvModule2d(
                in_channels=self.embed_dims,
                out_channels=self.reg_out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
        )
        return nn.Sequential(*reg_branch)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_dequant = DeQuantStub()
        self.reg_dequant = DeQuantStub()
        self.cls_branch = self._init_cls_branch()
        self.reg_branch = self._init_reg_branch()

        self.pos_embed = nn.Embedding(self.num_query, self.embed_dims)

        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

        self.query_embed_quant = QuantStub(scale=None)
        self.pos_embed_quant = QuantStub(scale=None)

    def export_reference_points(self, data, feat_hw):
        pos_embed = self.pos_embed.weight
        pos_embed = (
            pos_embed.permute(1, 0)
            .contiguous()
            .view(1, self.embed_dims, self.num_query_h, self.num_query_w)
        )
        return self.transformer.export_reference_points(
            pos_embed, data, feat_hw, self.bev_range
        )

    def forward(
        self, feats: List[Tensor], meta: Dict, compile_model: bool = False
    ) -> List[Tensor]:
        """Forward pass of the module.

        Args:
            feats : The feature tensor.
            meta : The metadata dictionary.
            compile_model: Whether in compile model.
        Returns:
            The list of output tensors and the reference points tensor.
        """
        query_embed = self.query_embed.weight.to(device=feats[0].device)
        query_embed = (
            query_embed.permute(1, 0)
            .contiguous()
            .view(1, self.embed_dims, self.num_query_h, self.num_query_w)
        )
        query_embed = self.query_embed_quant(query_embed)
        pos_embed = self.pos_embed.weight.to(device=feats[0].device)
        pos_embed = (
            pos_embed.permute(1, 0)
            .contiguous()
            .view(1, self.embed_dims, self.num_query_h, self.num_query_w)
        )
        start = len(feats) - self.num_levels
        feats = feats[start:]
        feats, reference_points = self.transformer(
            feats,
            query_embed,
            pos_embed,
            meta,
            self.bev_range,
            compile_model,
        )
        reference_points = reference_points.permute(0, 2, 3, 1).contiguous()
        return self.build_res_list(feats), reference_points

    @fx_wrap()
    def build_res_list(
        self,
        feats: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Build the list of output tensors.

        Args:
            feats : The list of feature tensors.
            reference_points : The reference points tensor.

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

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.transformer.set_qconfig()
        self.pos_embed.qconfig = None
        self.query_embed.qconfig = None
        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.cls_branch[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
            self.reg_branch[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
