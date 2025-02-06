import torch
from torch import nn

from horizon_plugin_pytorch.qtensor import QTensor
from .layernorm import LayerNorm


class _InstanceNorm(nn.Module):
    r"""Qat version."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        super().__init__()
        assert (
            track_running_stats is False
        ), "InstanceNorm only supports `track_running_stats` == `False`."

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig

        factory_kwargs = {"device": device, "dtype": dtype}

        normalized_dims = [1] * self._get_normalized_ndim()

        self.ins_norm = LayerNorm(
            normalized_dims,
            eps=eps,
            elementwise_affine=affine,
            qconfig=qconfig,
        )

        affine_shape = [
            num_features,
        ] + normalized_dims
        self.ins_norm.weight = nn.Parameter(
            torch.ones(affine_shape, **factory_kwargs)
        )
        self.ins_norm.bias = nn.Parameter(
            torch.zeros(affine_shape, **factory_kwargs)
        )

    def forward(self, input: QTensor) -> QTensor:
        return self.ins_norm(input)

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
            num_features=mod.num_features,
            eps=mod.eps,
            affine=mod.affine,
            device=mod.weight.device if mod.weight is not None else None,
            dtype=mod.weight.dtype if mod.weight is not None else None,
            track_running_stats=mod.track_running_stats,
            qconfig=qconfig,
        )

        if mod.affine:
            affine_shape = qat_mod.ins_norm.weight.shape
            with torch.no_grad():
                qat_mod.ins_norm.weight.copy_(mod.weight.reshape(affine_shape))
                qat_mod.ins_norm.bias.copy_(mod.bias.reshape(affine_shape))
        return qat_mod


class InstanceNorm1d(_InstanceNorm):
    _FLOAT_MODULE = nn.InstanceNorm1d

    def _get_normalized_ndim(self):
        return 1


class InstanceNorm2d(_InstanceNorm):
    _FLOAT_MODULE = nn.InstanceNorm2d

    def _get_normalized_ndim(self):
        return 2


class InstanceNorm3d(_InstanceNorm):
    _FLOAT_MODULE = nn.InstanceNorm3d

    def _get_normalized_ndim(self):
        return 3
