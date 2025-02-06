import numbers
from typing import List, Union

import torch
from torch import Size, Tensor, nn

from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.memory_opt import MemoryOptManager, MemoryOptSwitch
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.checkpoint import checkpoint
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from ..layer_norm import LayerNorm as HorizonLayerNorm
from .functional_modules import FloatFunctional
from .segment_lut import SegmentLUT
from .stubs import QuantStub

layernorm_checkpoint_switch = MemoryOptSwitch(
    "LayerNormCkpt", 0, [0, 1, 2], levels=[0, None, 2]
)
MemoryOptManager.register_switch(layernorm_checkpoint_switch)


class MultiDimMean(nn.Module):
    def __init__(self, dims, qconfig):
        super(MultiDimMean, self).__init__()
        self.dims = dims

        from .avg_pool2d import AvgPool2d

        if len(dims) == 1:
            self.pre_mean = FloatFunctional(qconfig)
            self.avg_pooling = None
            self.post_mean = None
        else:
            self.pre_mean = None
            self.avg_pooling = AvgPool2d(3, qconfig=qconfig)
            if len(dims) > 2:
                self.post_mean = FloatFunctional(qconfig)
            else:
                self.post_mean = None

    def forward(self, x):
        if self.pre_mean:
            x = self.pre_mean.mean(x, dim=self.dims[0], keepdim=True)
        if self.avg_pooling:
            self.avg_pooling.kernel_size = x.shape[-2:]
            if x.ndim == 3:
                n, c, l = x.shape  # noqa: E741
                x = self.avg_pooling(x.reshape([n, 1, c, l])).reshape(n, 1, 1)
            elif x.ndim == 5:
                n, c, d, h, w = x.shape  # noqa: E741
                x = self.avg_pooling(x.reshape([n * c, d, h, w])).reshape(
                    n, c, d, 1, 1
                )
            else:
                x = self.avg_pooling(x)
        if self.post_mean:
            x = self.post_mean.mean(x, dim=-3, keepdim=True)
        return x


class LayerNorm(nn.LayerNorm):
    r"""Qat version."""

    _version: int = 2
    _FLOAT_MODULE = (nn.LayerNorm, HorizonLayerNorm)

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        dim=None,
        sqrt_kwargs=None,
        qconfig=None,
    ) -> None:
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        assert isinstance(
            normalized_shape, (list, tuple, Size)
        ), "normalized_shape muse be a list or intergral or tuple or torch.Size"  # noqa: E501
        assert (
            len(normalized_shape) < 4
        ), "Only support layernorm on W or HW or CHW."
        for v in normalized_shape:
            assert isinstance(
                v, numbers.Integral
            ), "elements of normalized_shape must be integral"
        assert isinstance(eps, float), "param eps must be a float"
        assert isinstance(
            elementwise_affine, bool
        ), "param elementwise_affine must be a bool"
        assert isinstance(
            dim, (type(None), numbers.Integral)
        ), "param dim must be None or a integral"
        if dim is None:
            assert len(normalized_shape) in (
                1,
                2,
                3,
            ), "Only support layernorm on W or HW or CHW."

        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"
        assert (
            qconfig.weight is not None
        ), "qconfig.activation must be provided"

        super(LayerNorm, self).__init__(
            normalized_shape,
            eps,
            elementwise_affine,
            device=device,
            dtype=dtype,
        )

        self.dims = (
            tuple(reversed(range(-1, -len(normalized_shape) - 1, -1)))
            if dim is None
            else (dim,)
        )

        self.qconfig = qconfig

        from horizon_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)

        self.input_mean = MultiDimMean(self.dims, qconfig=int16_qconfig)
        self.sub = FloatFunctional(qconfig=int16_qconfig)
        self.mul = FloatFunctional(int16_qconfig)
        self.var_mean = MultiDimMean(self.dims, qconfig=int16_qconfig)

        default_sqrt_kwargs = {
            "simulated_func": lambda x: torch.rsqrt(x + self.eps),
            "is_centrosymmetric": True,
            "inverse_func": lambda x: torch.reciprocal(torch.pow(x, 2)),
            "qconfig": int16_qconfig,
            "auto_divide_strategy": "curvature",
        }
        if sqrt_kwargs is not None:
            assert isinstance(sqrt_kwargs, dict), "sqrt_kwargs must be a dict"
            default_sqrt_kwargs.update(sqrt_kwargs)

        self.rsqrt = SegmentLUT(**default_sqrt_kwargs)

        if self.elementwise_affine:
            self.out_mul = FloatFunctional(qconfig=int16_qconfig)
            self.weight_quant = QuantStub(qconfig=int16_qconfig)
            self.weight_mul = FloatFunctional(qconfig=int16_qconfig)
            self.bias_quant = QuantStub(qconfig=int16_qconfig)
            self.bias_add = FloatFunctional(qconfig=qconfig)
        else:
            self.out_mul = FloatFunctional(qconfig=qconfig)

        self.forward_mapping = {
            0: self._forward_ckpt_l0,
            1: self._forward_ckpt_l1,
            2: self._forward_ckpt_l2,
        }

    def _forward_ckpt_l0(self, input: QTensor):
        mu = self.input_mean(input)
        diff = self.sub.sub(input, mu)
        diff_square = self.mul.mul(diff, diff)
        var = self.var_mean(diff_square)
        dev_rec = self.rsqrt(var)
        ret = self.out_mul.mul(diff, dev_rec)

        if self.elementwise_affine:
            ret = self.weight_mul.mul(ret, self.weight_quant(self.weight))
            ret = self.bias_add.add(ret, self.bias_quant(self.bias))

        return ret

    def _forward_ckpt_l1(self, input: QTensor):
        mu = self.input_mean(input)
        diff = self.sub.sub(input, mu)
        diff_square = self.mul.mul(diff, diff)
        var = self.var_mean(diff_square)
        dev_rec = self.rsqrt.forward_wo_fq(var)
        ret = checkpoint(self._forward_ckpt_l1_tail, diff, dev_rec)

        if self.elementwise_affine:
            ret = self.weight_mul.mul(ret, self.weight_quant(self.weight))
            ret = self.bias_add.add(ret, self.bias_quant(self.bias))

        return ret

    def _forward_ckpt_l1_tail(self, diff: QTensor, dev_rec: Tensor):
        dev_rec = self.rsqrt.activation_post_process(dev_rec)
        ret = self.out_mul.mul(diff, dev_rec)

        return ret

    def _forward_ckpt_l2(self, input: QTensor):
        mu = self.input_mean(input)
        diff = self.sub.sub(input, mu)
        ret = checkpoint(self._forward_ckpt_l2_tail, diff)

        if self.elementwise_affine:
            ret = self.weight_mul.mul(ret, self.weight_quant(self.weight))
            ret = self.bias_add.add(ret, self.bias_quant(self.bias))

        return ret

    def _forward_ckpt_l2_tail(self, diff: QTensor):
        diff_square = self.mul.mul(diff, diff)
        var = self.var_mean(diff_square)
        dev_rec = self.rsqrt(var)
        ret = self.out_mul.mul(diff, dev_rec)

        return ret

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: QTensor) -> QTensor:
        if not input.is_quantized:
            input = QTensor(
                input.as_subclass(torch.Tensor).float(),
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        ret = self.forward_mapping[layernorm_checkpoint_switch.value](input)

        return ret

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # For compatibility, when loading old version state_dict, set self.mul
        # to int8
        if (
            get_version(self, prefix, local_metadata) < 2
            and len(self.dims) > 1
            and isinstance(self.mul, FloatFunctional)
        ):
            self.mul = FloatFunctional(self.qconfig)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) in cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + (c.__name__ for c in cls._FLOAT_MODULE)
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_mod = cls(
            normalized_shape=mod.normalized_shape,
            eps=mod.eps,
            elementwise_affine=mod.elementwise_affine,
            device=mod.weight.device if mod.weight is not None else None,
            dtype=mod.weight.dtype if mod.weight is not None else None,
            dim=mod.dim if hasattr(mod, "dim") else None,
            sqrt_kwargs=mod.sqrt_kwargs
            if hasattr(mod, "sqrt_kwargs")
            else None,
            qconfig=qconfig,
        )
        if mod.elementwise_affine:
            with torch.no_grad():
                qat_mod.weight.copy_(mod.weight)
                qat_mod.bias.copy_(mod.bias)
        return qat_mod
