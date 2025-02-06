# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import Dict, List, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.detr3d.head import FFN
from hat.models.task_modules.view_fusion.view_transformer import (
    ViewTransformer,
)
from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from hat.utils.model_helpers import fx_wrap


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute position embedding, learned.

    Args:
        num_pos_feats: List containing the number of features for row,
                       column, and frame embeddings, respectively.
        num_pos: Number of positions to embed.

    Example:
    ```
        import torch
        model = PositionEmbeddingLearned(num_pos_feats=[53, 53, 54],
                                         num_pos=50)
        input_patch = torch.rand(16, 3, 128, 128)
        num_views = 4
        output_pos = model(input_patch, num_views)
        print(output_pos.shape)  # should print (128, 128, 16, 160)
    ```
    """

    def __init__(self, num_pos_feats: List[int] = None, num_pos: int = 50):
        super().__init__()
        self.row_embed = nn.Embedding(num_pos, num_pos_feats[0])
        self.col_embed = nn.Embedding(num_pos, num_pos_feats[1])
        self.frame_embed = nn.Embedding(num_pos, num_pos_feats[2])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the embedding layers."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.frame_embed.weight)

    def forward(self, patch: Tensor, num_views: int) -> Tensor:
        """
        Forward pass of the PositionEmbeddingLearned module.

        Args:
        - patch: Input tensor.
        - num_views: Number of views.

        Returns:
        - pos: Output tensor of shape (height, width, batch_size,
                                       num_pos_feats[0]+num_pos_feats[1]+num_pos_feats[2]).
        """

        _, _, h, w = patch.shape
        f = num_views
        i = torch.arange(h, device=patch.device)
        j = torch.arange(w, device=patch.device)
        k = torch.arange(f, device=patch.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.frame_embed(k)
        pos = torch.cat(
            [
                x_emb.unsqueeze(1).unsqueeze(1).repeat(1, w, f, 1),
                y_emb.unsqueeze(1).unsqueeze(0).repeat(h, 1, f, 1),
                z_emb.unsqueeze(0).unsqueeze(0).repeat(h, w, 1, 1),
            ],
            dim=-1,
        ).permute(2, 3, 0, 1)
        return pos


class PositionEmbeddingLearned2D(nn.Module):
    """
    Absolute position embedding, learned.

    Args:
        embed_dims: Number of dimensions for the position embeddings.
        num_pos: Number of positional embeddings in the row
                 and column dimensions, respectively.

    Example:
    ```
        import torch
        model = PositionEmbeddingLearned2D(embed_dims=256, num_pos=[64, 64])
        input_patch = torch.rand(16, 3, 128, 128)
        output_pos = model(input_patch)
        print(output_pos.shape)  # should print (1, 256, 128, 128)
    ```
    """

    def __init__(self, embed_dims: int = 256, num_pos: List[int] = None):
        super().__init__()
        self.row_embed = nn.Embedding(num_pos[0], embed_dims)
        self.col_embed = nn.Embedding(num_pos[1], embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the embedding layers."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, patch: Tensor) -> Tensor:
        """
        Forward pass of the PositionEmbeddingLearned2D module.

        Args:
            patch: Input tensor.

        Returns:
            pos: Output tensor.
        """
        hw, _ = patch.shape
        hw = torch.tensor(hw)
        h = w = torch.sqrt(hw).int()
        i = torch.arange(h, device=patch.device)
        j = torch.arange(w, device=patch.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = x_emb.unsqueeze(1).repeat(1, w, 1) + y_emb.unsqueeze(0).repeat(
            h, 1, 1
        )
        return pos.permute(2, 0, 1).contiguous().unsqueeze(0)


def gen_sineembed_for_position(
    pos_tensor: Tensor, height_range: Tuple[float, float], dim: int
) -> Tensor:
    """
    Generate sine embeddings for the position tensor.

    Args:
        pos_tensor: Position tensori.
        height_range: Range of height values to be encoded.
        dim: Number of dimensions for the sine encodings.

    Returns:
    - pos_h: Sine embeddings for the position tensor.

    Example:
    ```
        import torch
        pos_tensor = torch.rand(16, 3, 128, 128)
        height_range = (0, 1)
        dim = 256
        pos_h = gen_sineembed_for_position(pos_tensor, height_range, dim)
        print(pos_h.shape)  # should print (16, 256, 128, 128)
    ```
    """
    scale = 2 * math.pi / (height_range[1] - height_range[0])
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 100 ** (2 * (dim_t // 2) / dim)
    h_embed = (pos_tensor - height_range[0]) * scale
    dim_t = dim_t.view(1, -1, 1, 1)
    pos_h = h_embed / dim_t
    pos_h = torch.concat(
        (pos_h[:, 0::2, :, :].sin(), pos_h[:, 1::2, :, :].cos()), dim=1
    )
    return pos_h


class MLP(nn.Module):
    """
    An MLP module used in the Detr3D model.

    Args:
        input_channels: Number of input channels.
        output_channels: Number of output channels.
        feedforward_channels: Number of channels in
                              the intermediate feedforward
                              layers.
    """

    def __init__(
        self,
        input_channels: int = 256,
        output_channels: int = 256,
        feedforward_channels: int = 512,
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            ConvModule2d(
                in_channels=input_channels,
                out_channels=feedforward_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.Conv2d(
                in_channels=feedforward_channels,
                out_channels=output_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
        )
        self._init_weight()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x: Input tensor.

        Returns:
            x : Output tensor.
        """
        x = self.mlp(x)
        return x

    def _init_weight(self) -> None:
        """Initialize the weights of the MLP's convolutional layers."""
        for mod in self.mlp:
            if isinstance(mod, ConvModule2d):
                xavier_init(mod[0], distribution="uniform", bias=0.0)

    def fuse_model(self) -> None:
        """Fuse the first ConvModule2d in the MLP for model quantization."""
        self.mlp[0].fuse_model()


class Encoderlayer(nn.Module):
    """
    Encoder layer of a transformer model.

    Args:
        embed_dims: Embedding dimensions.
        num_heads: Number of attention heads.
        feedforward_channels: Number of channels in the feedforward layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
    ):
        super(Encoderlayer, self).__init__()
        self.norm1 = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)
        self.norm2 = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=dropout,
        )
        self.self_attns = MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
        )
        self.pos_add = FloatFunctional()
        self.dropout1_add = FloatFunctional()
        self.dropout2_add = FloatFunctional()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Forward pass of the Encoderlayer.

        Args:
            x: Input tensor.
            pos: Position tensor.

        Returns:
        - tgt2: Output tensor.
        """
        x = self.norm1(x)
        q = k = self.pos_add.add(x, pos)
        tgt, _ = self.self_attns(query=q, key=k, value=x)
        tgt = self.dropout1_add.add(x, self.dropout1(tgt))
        tgt2 = self.norm2(tgt)
        tgt2 = self.ffn(tgt2)
        tgt2 = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        return tgt2


class Encoder(nn.Module):
    """
    Encoder module of a transformer model.

    Args:
        num_layers: Number of encoder layers.
        embed_dims: Embedding dimensions.
        num_heads: Number of attention heads.
        feedforward_channels: Number of channels in the feedforward layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int = 1,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList()
        for _ in range(num_layers):
            self.encoders.append(
                Encoderlayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    dropout=dropout,
                )
            )
        self.norm = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Forward pass of the Encoder module.

        Args:
            x: Input tensor.
            pos: Position tensor.

        Returns:
            x: Output tensor.
        """
        for encoder in self.encoders:
            x = encoder(x, pos=pos)
        x = self.norm(x)
        return x


class Decoderlayer(nn.Module):
    """
    Decoder layer of a transformer model.

    Args:
        embed_dims: Embedding dimensions.
        num_heads: Number of attention heads.
        feedforward_channels: Number of channels in the feedforward layers.
        dropout: Dropout rate.
        num_views: Number of views.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        num_views: int = 6,
    ):
        super(Decoderlayer, self).__init__()
        self.embed_dims = embed_dims
        self.num_views = num_views
        self.cross_attns = MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
        )

        self.Qadd = FloatFunctional()
        self.Qadd2 = FloatFunctional()
        self.Kadd = FloatFunctional()
        self.tgt_cat = FloatFunctional()

        self.norm1 = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)
        self.norm2 = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=dropout,
        )

        self.dropout1_add = FloatFunctional()
        self.dropout2_add = FloatFunctional()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.proj = nn.Conv2d(
            in_channels=embed_dims * 2,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
        )

    def forward(
        self,
        feat: Tensor,
        tgt: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        ref_h_embed: Tensor,
    ):
        """
        Forward pass of the Decoderlayer.

        Args:
            feat: Input feature tensor..
            tgt: Target tensor.
            query_pos: Position embedding for the target tensor.
            key_pos: Position embedding for the input feature tensor.
            ref_h_embed: Position embedding for the reference height.
        Returns:
            tgt: Output tensor.
        """
        n, c, h, w = feat.shape
        bs = n // self.num_views
        feat = feat.view(-1, self.num_views, c, h, w)
        key_pos = key_pos.view(-1, self.num_views, c, h, w)

        feat = feat.permute(0, 2, 1, 3, 4).contiguous().view(bs, c, -1, w)
        key_pos = (
            key_pos.permute(0, 2, 1, 3, 4).contiguous().view(bs, c, -1, w)
        )
        query = self.Qadd.add(tgt, query_pos)

        query = self.Qadd2.add(query, ref_h_embed)
        key = self.Kadd.add(feat, key_pos)
        tgt2, _ = self.cross_attns(query=query, key=key, value=feat)

        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        tgt2 = self.ffn(tgt)
        tgt = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        return tgt


class Decoder(nn.Module):
    """
    Decoder module of a transformer model.

    Args:
        num_layers: Number of decoder layers.
        embed_dims: Embedding dimensions.
        num_heads: Number of attention heads.
        feedforward_channels: Number of channels in the feedforward layers.
        dropout: Dropout rate.
        num_views: Number of views.
    """

    def __init__(
        self,
        num_layers: int = 2,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        num_views: int = 6,
    ):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.decoders.append(
                Decoderlayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    dropout=dropout,
                    num_views=num_views,
                )
            )
        self.query_trans_pos = MLP(
            input_channels=embed_dims,
            output_channels=embed_dims,
            feedforward_channels=embed_dims,
        )
        self.mul = FloatFunctional()

    def forward(
        self,
        x: Tensor,
        tgt: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        ref_h_embed: Tensor,
    ) -> Tensor:
        """
        Forward pass of the Decoder module.

        Args:
            x: Input tensor.
            tgt: Target tensor.
            query_pos: Position embedding for the target tensor.
            key_pos: Position embedding for the input tensor.
            ref_h_embed: Position embedding for the reference height.

        Returns:
            tgt: Output tensor.
        """
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                pos_transformation = self.query_trans_pos(tgt)
                ref_h_embed = self.mul.mul(ref_h_embed, pos_transformation)
            ref_h_embed = ref_h_embed + query_pos
            tgt = decoder(
                x,
                tgt=tgt,
                query_pos=query_pos,
                key_pos=key_pos,
                ref_h_embed=ref_h_embed,
            )
        return tgt


@OBJECT_REGISTRY.register
class CFTTransformer(ViewTransformer):
    """
    Cross-View Fusion Transformer model for computer vision tasks.

    Args:
        embed_dims: Embedding dimensions.
        position_range: Range of position values.
        num_heads: Number of attention heads.
        feedforward_channels: Number of channels in the feedforward layers.
        dropout: Dropout rate.
        encoder_layers: Number of encoder layers.
        decoder_layers: Number of decoder layers.
        num_pos: Number of positions to embed.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        position_range: List[float] = None,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
        num_pos: int = 16,
        **kwargs
    ):
        super(CFTTransformer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.position_range = position_range

        num_queries = self.grid_size[0] * self.grid_size[1]
        self.query_embed = nn.Embedding(num_queries, self.embed_dims)
        self.pos_embedding = PositionEmbeddingLearned(
            num_pos_feats=[100, 100, 56], num_pos=num_pos
        )
        self.query_pos_embed = PositionEmbeddingLearned2D(
            embed_dims=256, num_pos=self.grid_size
        )
        self.ref_h_head = MLP(
            input_channels=embed_dims,
            output_channels=1,
            feedforward_channels=embed_dims,
        )

        self.encoder = Encoder(
            num_layers=encoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_layers=decoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
            num_views=self.num_views,
        )
        self.tgt_quant = QuantStub()
        self.query_pos_quant = QuantStub()
        self.key_pos_quant = QuantStub()
        self.ref_h_quant = QuantStub()

    @fx_wrap()
    def _position_embed(
        self, feats: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute position embeddings.

        Args:
            feats: Input feature tensor of shape
                   (batch_size, channels, height, width).

        Returns:
            query_pos: Query position embedding tensor.
            key_pos: Key position embedding tensor.
            ref_h_embed: Position embedding of reference height tensor.
            ref_h: Reference height tensor.
        """
        n, c, h, w = feats.shape

        query_pos = self.query_pos_embed(self.query_embed.weight)
        key_pos = self.pos_embedding(feats, self.num_views)

        height_range = [self.position_range[2], self.position_range[5]]
        ref_h = self.ref_h_head(query_pos)
        ref_h = (
            ref_h.sigmoid() * (height_range[1] - height_range[0])
            + height_range[0]
        )
        ref_h_embed = gen_sineembed_for_position(
            ref_h, height_range, self.embed_dims
        )
        return query_pos, key_pos, ref_h_embed, ref_h

    def forward(
        self, feats: Tensor, data: Tensor, compile_model: bool
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the CFTTransformer.

        Args:
            feats: Input feature tensor.
            data: Dictionary containing the input data.
            compile_model: Flag indicating whether the model is being compiled.

        Returns:
            feats: Output feature tensor.
            ref_h: Reference height tensor.
        """
        query_pos, key_pos, ref_h_embed, ref_h = self._position_embed(feats)

        bs = feats.shape[0] // self.num_views
        key_pos = key_pos.repeat(bs, 1, 1, 1)
        tgt = (
            self.query_embed.weight.view(
                self.grid_size[0], self.grid_size[1], -1
            )
            .repeat(bs, 1, 1, 1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        key_pos = self.key_pos_quant(key_pos)
        feats = self.encoder(feats, pos=key_pos)
        tgt = self.tgt_quant(tgt)
        query_pos = self.query_pos_quant(query_pos)
        ref_h_embed = self.ref_h_quant(ref_h_embed)
        feats = self.decoder(
            feats,
            tgt=tgt,
            query_pos=query_pos,
            key_pos=key_pos,
            ref_h_embed=ref_h_embed,
        )
        return feats, ref_h

    def set_qconfig(self) -> Tensor:
        """Set the quantization configuration for the model."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.query_pos_embed.qconfig = None
        self.pos_embedding.qconfig = None
        self.ref_h_head.qconfig = None
        self.query_embed.qconfig = None

    def export_reference_points(
        self, meta: Dict, feat_hw: Tuple[int, int]
    ) -> Dict:
        """Export refrence points.

        Args:
            meta: A dictionary containing the input data.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """
        return {}


@OBJECT_REGISTRY.register
class CFTAuxHead(nn.Module):
    """
    Auxiliary head module for the CFTTransformer.

    Args:
        out_size_factor: Output size factor.
        min_radius: Minimum radius of the heatmaps.
        upscale: Upscale factor for resizing the features.
        loss: Loss function.
    """

    def __init__(
        self,
        out_size_factor: int = 1,
        min_radius: int = 3,
        upscale=4.0,
        loss: nn.Module = None,
    ):
        super(CFTAuxHead, self).__init__()
        self.out_size_factor = out_size_factor
        self.loss = loss
        self.min_radius = min_radius
        self.upscale = upscale
        self.resize = hnn.Interpolate(
            scale_factor=self.upscale,
            align_corners=None,
            recompute_scale_factor=True,
        )
        self.dequant = DeQuantStub()

    def _get_gts(self, meta: Dict) -> Dict:
        """
        Compute the heatmap target for a single feature.

        Args:
            feat: Input feature tensor.
            gt_bboxes: List of ground truth bounding boxes.

        Returns:
            heatmap: Heatmap tensor.

        """
        return meta["bev_bboxes_labels"]

    def get_targets_single(
        self, feat: Tensor, gt_bboxes: List[np.array]
    ) -> Tensor:
        """
        Compute the heatmap target for a single feature.

        Args:
            feat : Input feature tensor.
            gt_bboxes : List of ground truth bounding boxes.

        Returns:
            heatmap : Heatmap tensor.
        """
        feat_size = feat.shape[1:]
        heatmap = torch.zeros((feat_size[0], feat_size[1]), device=feat.device)
        for bbox in gt_bboxes:
            width, length = bbox[3:5] / self.out_size_factor
            if width > 0 and length > 0:
                w, h = width // 2, length // 2
                w, h = max(w, self.min_radius), max(h, self.min_radius)
                x, y = bbox[:2] / self.out_size_factor
                z = bbox[2]
                hi = torch.tensor([z]).to(device=feat.device)
                center = torch.tensor(
                    [x, y], dtype=torch.float32, device=feat.device
                )
                center_int = center.to(torch.int32)
                xmin = torch.clamp(
                    center_int[0] + torch.tensor([-w], device=feat.device),
                    min=0,
                ).int()
                xmax = torch.clamp(
                    center_int[0] + torch.tensor([w + 1], device=feat.device),
                    max=feat_size[0],
                ).int()
                ymin = torch.clamp(
                    center_int[1] + torch.tensor([-h], device=feat.device),
                    min=0,
                ).int()
                ymax = torch.clamp(
                    center_int[1] + torch.tensor([h + 1], device=feat.device),
                    max=feat_size[1],
                ).int()
                heatmap[xmin:xmax, ymin:ymax] = hi

        return heatmap

    @fx_wrap()
    def forward(self, feat: Tensor, meta: Dict) -> Dict:
        """
        Forward pass of the CFTAuxHead.

        Args:
            feat: Input feature tensor.
            meta: Dictionary containing the input metadata.

        Returns:
            Dictionary containing the loss value.
        """
        feat = self.dequant(feat)
        feat = self.resize(feat)
        gt_bboxes = meta["bev_bboxes_labels"]
        hight_heatmaps = multi_apply(self.get_targets_single, feat, gt_bboxes)
        hight_heatmaps = torch.stack(hight_heatmaps)
        loss = self.loss(feat, hight_heatmaps)
        return {"ref_h_loss": loss}

    def set_qconfig(self) -> None:
        """Set the quantization configuration for the model."""
        self.qconfig = None
