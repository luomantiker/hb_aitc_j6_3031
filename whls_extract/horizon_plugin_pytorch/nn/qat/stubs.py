import logging

import torch
from torch import nn

from horizon_plugin_pytorch.qtensor import QTensor
from .qat_meta import is_float

logger = logging.getLogger(__name__)


class QuantStub(nn.Module):
    r"""Quantize stub module.

    Args:
        scale: Pass a number to use as fixed scale.
        zero_point: Pass a number to use as fixed zero_point.
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    _FLOAT_MODULE = torch.quantization.QuantStub

    def __init__(self, scale=None, zero_point=None, qconfig=None):
        super(QuantStub, self).__init__()
        assert qconfig, "qconfig must be provided for QAT module"
        self.scale = scale
        self.zero_point = zero_point
        self.qconfig = qconfig
        self.activation_post_process = self.qconfig.activation()
        if is_float(self.activation_post_process):
            self.scale = None
        if self.scale is not None:
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(
                self.scale, self.zero_point
            )

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
        if (
            self.scale is not None
            and not self.activation_post_process._observer_enabled
        ):
            loaded_scale = state_dict.get(
                prefix + "activation_post_process.scale", None
            )
            if loaded_scale is not None:
                current_scale = self.activation_post_process.scale
                if not torch.equal(
                    loaded_scale.to(device=current_scale.device),
                    current_scale,
                ):
                    raise ValueError(
                        "Loaded scale {} is conflict "
                        "with setted scale {}".format(
                            loaded_scale, current_scale
                        )
                    )

        super(QuantStub, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        """Refine this docstring in the future.

        When training with one net, but bpu inference with multi-subnet,
        qtensor is allowed as input
        """
        if isinstance(x, QTensor):
            if self.scale is not None:
                assert torch.all(
                    x.q_scale() == self.scale
                ), "input scale must be the same as op's"
            if self.activation_post_process.dtype != x.dtype:
                logger.warning(
                    f"QuantStub out dtype {self.activation_post_process.dtype}"
                    f" will be changed to {x.dtype}.",
                    extra={"call_times_context": ("message")},
                )
                self.activation_post_process.reset_dtype(x.dtype, False)
            self.activation_post_process.disable_observer()
            self.activation_post_process.disable_fake_quant()
            self.activation_post_process.set_qparams(x.q_scale())
            return self.activation_post_process(x.as_subclass(torch.Tensor))
        return self.activation_post_process(x)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        from horizon_plugin_pytorch.quantization import QuantStub

        cls._FLOAT_MODULE = (torch.quantization.QuantStub, QuantStub)

        assert type(mod) in cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + [modc.__name__ for modc in cls._FLOAT_MODULE]
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_stub = cls(
            scale=getattr(mod, "scale", None),
            zero_point=getattr(mod, "zero_point", None),
            qconfig=qconfig,
        )
        return qat_stub


class DeQuantStub(torch.quantization.DeQuantStub):
    r"""Dequantize stub module."""

    _FLOAT_MODULE = torch.quantization.DeQuantStub

    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        """Convert QTensor to Tensor."""
        if isinstance(x, QTensor):
            return x.dequantize()
        else:
            return x

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
        return cls()
