import logging
from typing import Optional

import torch
from torch import Tensor, nn

from horizon_plugin_pytorch.qtensor import QTensor
from .functional_modules import FloatFunctional
from .linear import Linear
from .softmax import SegmentLUTSoftmax
from .stubs import QuantStub

logger = logging.getLogger(__name__)


class MultiheadAttention(nn.Module):
    r"""Qat version."""

    _FLOAT_MODULE = nn.MultiheadAttention

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim=None,
        vdim=None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        qconfig=None,
    ):
        assert add_bias_kv is False, "Only support add_bias_kv is false"
        assert add_zero_attn is False, "Only support add_zero_attn is false"

        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"
        assert qconfig.weight is not None, "qconfig.weight must be provided"

        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )
        assert (
            self._qkv_same_embed_dim
        ), "Only support q k v with same embed dim"

        self.num_heads = num_heads
        self.bias = bias
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qconfig = qconfig

        # define q k v projection layers
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, qconfig=qconfig)
        self.k_proj = Linear(embed_dim, self.kdim, bias=bias, qconfig=qconfig)
        self.v_proj = Linear(embed_dim, self.vdim, bias=bias, qconfig=qconfig)

        self.out_proj = Linear(
            embed_dim, embed_dim, bias=bias, qconfig=qconfig
        )

        self.attention_drop = nn.Dropout(p=dropout)

        self.softmax = SegmentLUTSoftmax(dim=-1, qconfig=qconfig)

        self.matmul = FloatFunctional(qconfig=qconfig)
        self.add = FloatFunctional(qconfig=qconfig)
        self.mask_add = FloatFunctional(qconfig=qconfig)
        self.attn_matmul = FloatFunctional(qconfig=qconfig)
        self.attn_mask_quant = QuantStub(qconfig=qconfig)
        self.attn_weights_sum = FloatFunctional(qconfig=qconfig)

    def forward(
        self,
        query: QTensor,
        key: QTensor,
        value: QTensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if self.batch_first:
            query, key, value = [
                x.transpose(1, 0) for x in (query, key, value)
            ]

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert (
            embed_dim == self.embed_dim
        ), f"was expecting embedding dimension of {self.embed_dim}, \
            but got {embed_dim}"
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"
        assert not is_causal, "is_causal is not supported by {}".format(
            self.__class__.__name__
        )

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                logger.warning(
                    "Byte tensor for attn_mask is deprecated. "
                    "Use bool tensor instead.",
                    extra={"call_times_context": ("message")},
                )
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert (
                    attn_mask.is_floating_point()
                    or attn_mask.dtype == torch.bool
                ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(  # noqa: E501
                    attn_mask.dtype
                )
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape},"
                        "but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape},"
                        "but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # prep key padding mask
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            logger.warning(
                "Byte tensor for key_padding_mask is deprecated."
                "Use bool tensor instead.",
                extra={"call_times_context": ("message")},
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(k.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(v.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, \
                but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask, float("-100")
                )

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask = new_attn_mask.masked_fill(attn_mask, float("-100"))
            attn_mask = new_attn_mask
        if attn_mask is not None and not isinstance(attn_mask, QTensor):
            attn_mask = self.attn_mask_quant(attn_mask)

        # q = q * self.scale
        # attention = (q @ k.transpose(-2, -1))
        q = q.mul(self.head_dim ** -0.5)
        k = k.transpose(-2, -1).contiguous()
        attention = self.matmul.matmul(q, k)

        if attn_mask is not None:
            # attention = attention + mask
            attention = self.mask_add.add(attention, attn_mask)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attention_drop(attention)
        # output = (attention @ v)
        attn_output = self.attn_matmul.matmul(attention, v)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attention.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = self.attn_weights_sum.sum(
                    attn_output_weights,
                    dim=1,
                    keepdim=True,
                ).squeeze(1)
                attn_output_weights = attn_output_weights.mul(
                    1 / self.num_heads
                )
        else:
            attn_output_weights = None

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_mod = cls(
            embed_dim=mod.embed_dim,
            num_heads=mod.num_heads,
            dropout=mod.dropout,
            bias=True if mod.in_proj_bias is not None else False,
            add_bias_kv=True if mod.bias_k is not None else False,
            add_zero_attn=mod.add_zero_attn,
            kdim=mod.kdim,
            vdim=mod.vdim,
            batch_first=mod.batch_first,
            device=mod.out_proj.weight.device,
            dtype=mod.out_proj.weight.dtype,
            qconfig=qconfig,
        )
        if mod._qkv_same_embed_dim:
            with torch.no_grad():
                w_q, w_k, w_v = mod.in_proj_weight.chunk(3)
                qat_mod.q_proj.weight.copy_(w_q)
                qat_mod.k_proj.weight.copy_(w_k)
                qat_mod.v_proj.weight.copy_(w_v)
                if mod.in_proj_bias is not None:
                    b_q, b_k, b_v = mod.in_proj_bias.chunk(3)
                    qat_mod.q_proj.bias.copy_(b_q)
                    qat_mod.k_proj.bias.copy_(b_k)
                    qat_mod.v_proj.bias.copy_(b_v)
        with torch.no_grad():
            qat_mod.out_proj.weight.copy_(mod.out_proj.weight)
            if mod.in_proj_bias is not None:
                qat_mod.out_proj.bias.copy_(mod.out_proj.bias)
        return qat_mod
