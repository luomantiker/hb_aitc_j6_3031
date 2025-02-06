import logging
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.quantization import QuantStub

from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)


@replace_torch_nn_module(nn.MultiheadAttention)
class MultiheadAttention(nn.Module):
    """This is a quant-friendly impl of `torch.nn.MultiheadAttention`.

    All arguments and functionality are same as `torch.nn.MultiheadAttention`.
    Except that `is_causal` is not supported now.
    """

    _new_in_version = "v2.3.4"

    @typechecked
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.add_zero_attn = add_zero_attn
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, bias=bias, **factory_kwargs
        )
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, bias=bias, **factory_kwargs
        )
        self.matmul = FloatFunctional()  # for Q @ K
        self.add = FloatFunctional()
        self.mask_merge_add = FloatFunctional()
        self.mask_add = FloatFunctional()
        self.attn_mask_quant = QuantStub()
        self.attn_weights_mean = FloatFunctional()
        self.softmax = nn.Softmax(dim=-1)
        self.attention_drop = nn.Dropout(p=dropout)
        self.attn_matmul = FloatFunctional()
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_k_quant = QuantStub()
            self.bias_v_quant = QuantStub()
            self.bias_k_cat = FloatFunctional()
            self.bias_v_cat = FloatFunctional()
        else:
            self.bias_k = self.bias_v = None

        if add_zero_attn:
            self.zero_attn_quant = QuantStub()
            self.zero_attn_k_cat = FloatFunctional()
            self.zero_attn_v_cat = FloatFunctional()

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)

        if self.bias:
            constant_(self.q_proj.bias, 0.0)
            constant_(self.k_proj.bias, 0.0)
            constant_(self.v_proj.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This impl handle state_dict compatibility of QAT and Quantized model.
        q_state_dict_from_old_version = False

        sum_prefix = "{}attn_weights_sum.".format(prefix)
        mean_prefix = "{}attn_weights_mean.".format(prefix)

        scaled_buffer_names = (
            "activation_post_process.scale",
            "activation_post_process.activation_post_process.min_val",
            "activation_post_process.activation_post_process.max_val",
            "scale",  # for quantized FloatFunctional
        )
        for n in scaled_buffer_names:
            if sum_prefix + n in state_dict:
                q_state_dict_from_old_version = True
                state_dict[mean_prefix + n] = (
                    state_dict.pop(sum_prefix + n) / self.num_heads
                )

        keys = tuple(state_dict.keys())
        for k in keys:
            if k.startswith(sum_prefix):
                q_state_dict_from_old_version = True
                n = k[len(sum_prefix) :]
                state_dict[mean_prefix + n] = state_dict.pop(k)

        if q_state_dict_from_old_version:
            self.mask_merge_add = None
            logger.warning(
                "MultiheadAttention loaded from old version "
                "state_dict (generated before {}).\n"
                "1. It will be dangerous to carry on calib or QAT training"
                " from old state_dict, because observer state may be broken.\n"
                "2. Because the scale of `mask_merge_add` is missing, if "
                "key_padding_mask and attn_mask are both provided, they must "
                "be bool type.".format(self._new_in_version)
            )

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert not is_causal, "is_causal is not supported now"
        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint."
            )

        assert (
            query.size(-1) == self.embed_dim
        ), "Expecting query embedding dimension of {}, but got {}".format(
            self.embed_dim, query.size(-1)
        )
        assert (
            key.size(-1) == self.kdim
        ), "Expecting key embedding dimension of {}, but got {}".format(
            self.kdim, key.size(-1)
        )
        assert (
            value.size(-1) == self.vdim
        ), "Expecting value embedding dimension of {}, but got {}".format(
            self.vdim, value.size(-1)
        )

        is_batched = query.ndim == 3
        if is_batched:
            bsz = query.size(0 if self.batch_first else 1)
            tgt_len = query.size(1 if self.batch_first else 0)
            src_len = key.size(1 if self.batch_first else 0)
            assert key.dim() == 3 and value.dim() == 3, (
                "For batched (3-D) `query`, expected `key` and `value` to"
                " be 3-D but found {}-D and {}-D tensors respectively".format(
                    key.dim(), value.dim()
                )
            )
        else:
            bsz = 1
            tgt_len = query.size(0)
            src_len = key.size(0)
            assert key.dim() == 2 and value.dim() == 2, (
                "For unbatched (2-D) `query`, expected `key` and `value` to"
                " be 2-D but found {}-D and {}-D tensors respectively".format(
                    key.dim(), value.dim()
                )
            )
        if key_padding_mask is not None:
            if is_batched:
                expected_shape = (bsz, src_len)
            else:
                expected_shape = (src_len,)
            assert key_padding_mask.shape == expected_shape, (
                "Expected `key_padding_mask` shape to be"
                " {} but got {}".format(expected_shape, key_padding_mask.shape)
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be "
                "`None`, 2-D or 3-D but found {}-D tensor instead".format(
                    attn_mask.dim()
                )
            )
            if attn_mask.dim() == 2:
                expected_shape = (tgt_len, src_len)
            else:
                expected_shape = (bsz * self.num_heads, tgt_len, src_len)
            assert (
                attn_mask.shape == expected_shape
            ), "Expected `attn_mask` shape to be" " {} but got {}".format(
                expected_shape, attn_mask.shape
            )

        # convert inputs to batch middle
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (
                    x.transpose(1, 0) for x in (query, key, value)
                )

        query = query.reshape(tgt_len, bsz, self.embed_dim)
        key = key.reshape(src_len, bsz, self.kdim)
        value = value.reshape(src_len, bsz, self.vdim)

        assert (
            key.shape[:-1] == value.shape[:-1]
        ), "key's sequence and batch dims {} do not match value's {}".format(
            key.shape[:-1], value.shape[:-1]
        )

        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            bias_k = self.bias_k_quant(self.bias_k)
            bias_v = self.bias_v_quant(self.bias_v)
            k = self.bias_k_cat.cat([k, bias_k.repeat(1, bsz, 1)])
            v = self.bias_v_cat.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )
        k = k.reshape(
            k.size(0), bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)
        v = v.reshape(
            v.size(0), bsz * self.num_heads, self.head_dim
        ).transpose(0, 1)

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            zero_attn = self.zero_attn_quant(
                torch.zeros(zero_attn_shape, device=k.device)
            )
            k = self.zero_attn_k_cat.cat([k, zero_attn], dim=1)
            v = self.zero_attn_v_cat.cat([v, zero_attn], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.reshape(1, 1, tgt_len, src_len)
            else:
                attn_mask = attn_mask.reshape(
                    bsz, self.num_heads, tgt_len, src_len
                )
            if attn_mask.dtype is torch.bool:
                # use logical_or if both mask is bool
                if (
                    key_padding_mask is not None
                    and key_padding_mask.dtype is torch.bool
                ):
                    attn_mask = torch.logical_or(
                        attn_mask, key_padding_mask.reshape(bsz, 1, 1, src_len)
                    )
                    key_padding_mask = None

                attn_mask = torch.zeros_like(
                    attn_mask, dtype=torch.float
                ).masked_fill(attn_mask, -100.0)
                attn_mask = self.attn_mask_quant(attn_mask)

        if key_padding_mask is not None:
            if key_padding_mask.dtype is torch.bool:
                key_padding_mask = torch.zeros_like(
                    key_padding_mask, dtype=torch.float
                ).masked_fill(key_padding_mask, -100.0)
                # share mask_quant because value range is the same
                key_padding_mask = self.attn_mask_quant(key_padding_mask)

            key_padding_mask = key_padding_mask.reshape(bsz, 1, 1, src_len)

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                if self.mask_merge_add is None:
                    raise RuntimeError(
                        "When MultiheadAttention loaded from old version "
                        "state_dict (generated before {}), if key_padding_mask"
                        " and attn_mask are both provided, they must "
                        "be bool type.".format(self._new_in_version)
                    )
                attn_mask = self.mask_merge_add.add(
                    attn_mask, key_padding_mask
                )

        # bsz * self.num_heads, tgt_len, self.head_dim
        q_scaled = q.mul(self.head_dim ** -0.5)

        # bsz * self.num_heads, tgt_len, src_len
        attn_output_weights = self.matmul.matmul(q_scaled, k.transpose(-2, -1))
        if attn_mask is not None:
            attn_mask = attn_mask.expand(
                bsz, self.num_heads, tgt_len, src_len
            ).reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_output_weights = self.mask_add.add(
                attn_output_weights, attn_mask
            )
        attn_output_weights = self.softmax(attn_output_weights)
        if self.dropout > 0:
            attn_output_weights = self.attention_drop(attn_output_weights)

        # bsz * self.num_heads, tgt_len, self.head_dim
        attn_output = self.attn_matmul.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(
            tgt_len, bsz, self.embed_dim
        )
        # tgt_len, bsz, self.embed_dim
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = self.attn_weights_mean.mean(
                    attn_output_weights, dim=1
                )
        else:
            attn_output_weights = None

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            if attn_output_weights is not None:
                attn_output_weights = attn_output_weights.squeeze(1)
        elif self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights

    @classmethod
    def from_torch(cls, mod: nn.MultiheadAttention):
        new_mod = cls(
            mod.embed_dim,
            mod.num_heads,
            mod.dropout,
            mod.in_proj_bias is not None,
            mod.bias_k is not None,
            mod.add_zero_attn,
            mod.kdim,
            mod.vdim,
            mod.batch_first,
        )

        with torch.no_grad():
            if mod._qkv_same_embed_dim:
                w_q, w_k, w_v = mod.in_proj_weight.chunk(3)
                new_mod.q_proj.weight.copy_(w_q)
                new_mod.k_proj.weight.copy_(w_k)
                new_mod.v_proj.weight.copy_(w_v)
            else:
                new_mod.q_proj.weight.copy_(mod.q_proj_weight)
                new_mod.k_proj.weight.copy_(mod.k_proj_weight)
                new_mod.v_proj.weight.copy_(mod.v_proj_weight)

            new_mod.out_proj.weight.copy_(mod.out_proj.weight)

            if new_mod.bias:
                b_q, b_k, b_v = mod.in_proj_bias.chunk(3)
                new_mod.q_proj.bias.copy_(b_q)
                new_mod.k_proj.bias.copy_(b_k)
                new_mod.v_proj.bias.copy_(b_v)
                new_mod.out_proj.bias.copy_(mod.out_proj.bias)

            if new_mod.bias_k is not None:
                new_mod.bias_k.copy_(mod.bias_k)
                new_mod.bias_v.copy_(mod.bias_v)

        new_mod.to(mod.out_proj.weight.device)
        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod
