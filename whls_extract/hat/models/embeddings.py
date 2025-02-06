# Copyright (c) Horizon Robotics. All rights reserved.

import math
from typing import Dict

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = [
    "PositionEmbeddingSine",
    "PositionEmbeddingLearned",
    "SinePositionalEncoding3D",
]


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        bn_kwargs: Dict,
        pe_channel: int = 3,
        is_with_relu: bool = False,
    ):
        """Positional embedding conv-2d layer.

        Args:
            bn_kwargs (Dict):
                for Bn layer. No Bn layer if bn_kwargs=None.
            pe_channel (int, optional):
                channel of input map. Defaults to 3.
            is_with_relu (bool, optional):
                whether to use the activation function relu. Defaults to False.
        """
        super().__init__()
        self.quant = QuantStub(scale=None)
        self.is_with_relu = is_with_relu
        # two structures of embedding layer upon with_relu or not,
        # based on experimental result, is_with_relu-False by default.
        if not is_with_relu:
            self.embedding = ConvModule2d(
                in_channels=pe_channel,
                out_channels=pe_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(pe_channel, **bn_kwargs),
                act_layer=None,
            )
        else:
            self.embedding = ExtSequential(
                [
                    ConvModule2d(
                        in_channels=pe_channel,
                        out_channels=pe_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        norm_layer=nn.BatchNorm2d(pe_channel, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                    ConvModule2d(
                        in_channels=pe_channel,
                        out_channels=pe_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        norm_layer=nn.BatchNorm2d(pe_channel, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                ]
            )

    def forward(self, coordinate_map: torch.Tensor) -> torch.Tensor:
        """Forward func for positional embedding layer.

        Args:
            coordinate_map (torch.Tensor): encoding map for embedding.

        Returns:
            torch.Tensor: position embedding.

        """

        coordinate_map_qat = self.quant(coordinate_map)
        position_embedding = self.embedding(coordinate_map_qat)

        return position_embedding

    def fuse_model(self):
        if not self.is_with_relu:
            self.embedding.fuse_model()
        else:
            getattr(self.embedding, "0").fuse_model()
            getattr(self.embedding, "1").fuse_model()


@OBJECT_REGISTRY.register
class PositionEmbeddingSine(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_pos_feats: The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature: The temperature used for scaling
            the position embedding. Default 10000.
        normalize: Whether to normalize the position
            embedding. Default False.
        scale: A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default 2*pi.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
        offset: float = 0.0,
    ):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.offset = offset
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
        )
        self.dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    @fx_wrap()
    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, -1:, :] + eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, -1:] + eps)
                * self.scale
            )

        dim_t = self.dim_t.to(mask.device)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


@OBJECT_REGISTRY.register
class PositionEmbeddingLearned(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_pos_feats: The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed: The dictionary size of row embeddings.
            Default 50.
        col_num_embed: The dictionary size of col embeddings.
            Default 50.
    """

    def __init__(self, num_pos_feats=256, row_num_embed=50, col_num_embed=50):
        super(PositionEmbeddingLearned, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_pos_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_pos_feats)
        self.num_pos_feats = num_pos_feats
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


@OBJECT_REGISTRY.register
class SinePositionalEncoding3D(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(
        self,
        num_feats,
        temperature=10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        super(SinePositionalEncoding3D, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, n, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, n, num_feats*3, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (
                (n_embed + self.offset)
                / (n_embed[:, -1:, :, :] + self.eps)
                * self.scale
            )
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, :, -1:, :] + self.eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, :, -1:] + self.eps)
                * self.scale
            )
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos = (
            torch.cat((pos_n, pos_y, pos_x), dim=4)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        return pos
