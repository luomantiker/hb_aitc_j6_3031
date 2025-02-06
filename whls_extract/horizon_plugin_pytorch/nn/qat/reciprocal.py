import torch
from torch import nn

from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.qtensor import QTensor
from ..reciprocal import Reciprocal as HorizonReciprocal
from .qat_meta import is_float
from .segment_lut import SegmentLUT


class Reciprocal(nn.Module):
    """qat version of Reciprocal module."""

    _FLOAT_MODULE = HorizonReciprocal

    def __init__(self, max_value=None, lut_kwargs=None, qconfig=None):
        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig activation must be provided"

        super(Reciprocal, self).__init__()

        self.max_value = None if max_value is None else abs(max_value)
        self.qconfig = qconfig

        if self.max_value is None:
            simulated_func = torch.reciprocal
        else:

            def simulated_func(x):
                return torch.reciprocal(x).clamp(
                    -self.max_value, self.max_value
                )

        default_lut_kwargs = {
            "dividing_points": None,
            "input_range": None,
            "auto_divide_strategy": "curvature",
            "gradients": [0, None],
        }
        if lut_kwargs is not None:
            assert isinstance(lut_kwargs, dict), "lut_kwargs must be a dict"
            assert all(x in default_lut_kwargs.keys() for x in lut_kwargs), (
                f"Only support setting {list(default_lut_kwargs.keys())} "
                + f"but get {list(lut_kwargs.keys())}"
            )
            default_lut_kwargs.update(lut_kwargs)
        self.reciprocal = SegmentLUT(
            simulated_func=simulated_func,
            is_centrosymmetric=True,
            inverse_func=torch.reciprocal,
            qconfig=qconfig,
            **default_lut_kwargs,
        )
        self.enable_clip = not is_float(
            self.reciprocal.activation_post_process
        )

    def _clip_input(self, input):
        # For int8 or int16 reciprocal, clip the values near 0
        # For fp16, do not modify input, overflow should be handled in
        # output FakeCast
        if (
            self.enable_clip
            and isinstance(input, QTensor)
            and not input.is_quantized
        ):
            # is qat
            input_data = input.as_subclass(torch.Tensor)
            lower_bound = input.q_scale().detach()[0] * (
                1 if input.dtype == qint8 else 32
            )
            self.reciprocal.input_range = [lower_bound, None]
            clamped_data = torch.where(
                input_data >= 0,
                input_data.clamp_min(lower_bound),
                input_data.clamp_max(-lower_bound),
            )

            input = QTensor(
                clamped_data,
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )

        return input

    def forward_wo_fq(self, input: QTensor):
        input = self._clip_input(input)
        return self.reciprocal.forward_wo_fq(input)

    def forward(self, input: QTensor):
        # only work in training
        from horizon_plugin_pytorch.fx.jit_scheme import Tracer
        from horizon_plugin_pytorch.quantization import hbdk4 as hb4

        if (
            not torch.onnx.is_in_onnx_export()
            and not Tracer.is_tracing()
            and not hb4.is_exporting()
        ):
            input = self._clip_input(input)
        return self.reciprocal(input)

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
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_reciprocal = cls(max_value=mod.max_value, qconfig=qconfig)
        return qat_reciprocal
