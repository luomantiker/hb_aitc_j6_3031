import copy
from distutils.version import LooseVersion
from typing import Optional

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]


@OBJECT_REGISTRY.register
class Transformer(nn.Module):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        embed_dims: The feature dimension.
        num_heads: Parallel attention heads.
        num_encoder_layers: Number of `TransformerEncoderLayer`.
        num_decoder_layers: Number of `TransformerDecoderLayer`.
        feedforward_channels: The hidden dimension for FFNs used in both
            encoder and decoder.
        dropout: Probability of an element to be zeroed. Default 0.1.
        act_layer: Activation module for FFNs used in both encoder
            and decoder. Default ReLU.
        normalize_before: Whether the normalization layer is ordered
            first in the encoder and decoder. Default False.
        return_intermediate_dec: Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False. If True, the returned
            `hs` has shape [num_decoder_layers, bs, num_query, embed_dims].
            If False, the returned `hs` will have shape [1, bs, num_query,
            embed_dims].
    """

    def __init__(
        self,
        embed_dims: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        act_layer: nn.Module = nn.ReLU,
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            embed_dims,
            num_heads,
            feedforward_channels,
            dropout,
            act_layer,
            normalize_before,
        )
        encoder_norm = (
            LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)
            if normalize_before
            else None
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            embed_dims,
            num_heads,
            feedforward_channels,
            dropout,
            act_layer,
            normalize_before,
        )
        decoder_norm = LayerNorm2d(normalized_shape=[embed_dims, 1, 1], dim=1)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.tgt_quant = QuantStub(scale=None)

        # self.init_weights()

        self.embed_dims = embed_dims
        self.num_heads = num_heads

    def init_weights(self):
        """Initialize the transformer weights."""
        # follow the official DETR to init parameters
        for name, p in self.named_parameters():
            if "attn" not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x: Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask: The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed: The query embedding for decoder, with shape
                [num_query, c].
            pos_embed: The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple, containing the following tensor:
                out_dec: decoder output. If return_intermediate_dec is True,
                output has shape [num_dec_layers, bs, num_query, embed_dims],
                else has shape [1, bs, num_query, embed_dims].
                memory: Output results from encoder, with shape
                [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        query_embed = (
            query_embed.transpose(0, 1)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
            .contiguous()
            .view(bs, query_embed.shape[1], 2, query_embed.shape[0] // 2)
        )  # [num_query, dim] -> [bs, dim, 2, num_query/2]
        mask = mask.flatten(1)  # [bs, h, w] -> [bs, h*w]

        tgt = torch.zeros_like(query_embed)  # [bs, dim, 2, num_query/2]
        # tgt = torch.zeros(query_embed.size(), device=query_embed.device)
        if LooseVersion(horizon.__version__) < LooseVersion("2.3.4"):
            tgt = self.tgt_quant(tgt)
        memory = self.encoder(x, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )  # [nb_dec, bs, dim, 2, num_query/2]
        hs = (
            hs.contiguous()
            .view(
                hs.shape[0],
                hs.shape[1],
                hs.shape[2],
                hs.shape[3] * hs.shape[4],
            )
            .permute(0, 1, 3, 2)
        )  # [nb_dec, bs, num_query, embed_dim]
        return hs, memory

    def fuse_model(self):
        pass

    def set_qconfig(self):
        for module in [self.encoder, self.decoder]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def set_calibration_qconfig(self):
        for module in [self.encoder, self.decoder]:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def set_qconfig(self):
        for module in self.layers:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def set_calibration_qconfig(self):
        for module in self.layers:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm=None, return_intermediate=False
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.intermediate_cat = quantized.FloatFunctional()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        for idx, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )  # [bs, dim, 2, num_query/2]
            if self.return_intermediate:
                if idx == 0:
                    intermediate = self.norm(output).unsqueeze(0)
                else:
                    intermediate = self.intermediate_cat.cat(
                        [intermediate, self.norm(output).unsqueeze(0)]
                    )

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate = intermediate[:-1, ...]
                intermediate = self.intermediate_cat.cat(
                    [intermediate, output.unsqueeze(0)]
                )

        if self.return_intermediate:
            return intermediate

        return output.unsqueeze(0)

    def set_qconfig(self):
        for module in self.layers:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def set_calibration_qconfig(self):
        for module in self.layers:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, 1)

        self.norm1 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.norm2 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pos_add = quantized.FloatFunctional()
        self.dropout1_add = quantized.FloatFunctional()
        self.dropout2_add = quantized.FloatFunctional()

        self.activation = activation(inplace=True)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else self.pos_add.add(tensor, pos)

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)  # [bs, c, h, w]
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[
            0
        ]  # [bs, embed_dim, h, w]
        src = self.dropout1_add.add(src, self.dropout1(src2))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.dropout2_add.add(src, self.dropout2(src2))
        src = self.norm2(src)  # [bs, c, h, w]
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = self.dropout1_add.add(src, self.dropout1(src2))
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = self.dropout2_add.add(src, self.dropout2(src2))
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        modules_list = [
            self.dropout1_add,
            self.norm1,
            self.linear1,
            self.activation,
            self.linear2,
            self.dropout2_add,
            self.norm2,
        ]
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )

    def set_calibration_qconfig(self):
        modules_list = [
            self.dropout1_add,
            self.norm1,
            self.linear1,
            self.activation,
            self.linear2,
            self.dropout2_add,
            self.norm2,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_calib_qconfig(
                dtype="qint16"
            )


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, 1)

        self.norm1 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.norm2 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.norm3 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pos_add = quantized.FloatFunctional()
        self.dropout1_add = quantized.FloatFunctional()
        self.dropout2_add = quantized.FloatFunctional()
        self.dropout3_add = quantized.FloatFunctional()

        self.activation = activation(inplace=True)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else self.pos_add.add(tensor, pos)

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(
            tgt, query_pos
        )  # [bs, dim, 2, num_query/2]
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[
            0
        ]  # [bs, dim, 2, num_query/2]
        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[
            0
        ]  # [bs, dim, 2, num_query/2]
        tgt = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.dropout3_add.add(tgt, self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        return tgt  # [bs, dim, 2, num_query/2]

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.dropout2_add.add(tgt, self.dropout2(tgt2))
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = self.dropout3_add.add(tgt, self.dropout3(tgt2))
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        modules_list = [
            self.dropout1_add,
            self.norm1,
            self.dropout2_add,
            self.norm2,
            self.linear1,
            self.activation,
            self.linear2,
            self.dropout3_add,
            self.norm3,
            self.self_attn.matmul,
            self.self_attn.attn_matmul,
            self.self_attn.out_proj,
            self.multihead_attn.matmul,
            self.multihead_attn.attn_matmul,
            self.multihead_attn.out_proj,
        ]
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )

    def set_calibration_qconfig(self):
        modules_list = [
            self.dropout1_add,
            self.norm1,
            self.dropout2_add,
            self.norm2,
            self.linear1,
            self.activation,
            self.linear2,
            self.dropout3_add,
            self.norm3,
            self.self_attn.matmul,
            self.self_attn.attn_matmul,
            self.self_attn.out_proj,
            self.multihead_attn.matmul,
            self.multihead_attn.attn_matmul,
            self.multihead_attn.out_proj,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_calib_qconfig(
                dtype="qint16"
            )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
