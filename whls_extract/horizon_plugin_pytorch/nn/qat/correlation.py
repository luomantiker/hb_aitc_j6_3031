import torch

from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.correlation import (
    Correlation as FloatCorrelation,
)
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import scale_quanti


class Correlation(FloatCorrelation):

    _FLOAT_MODULE = FloatCorrelation

    def __init__(
        self,
        kernel_size: int = 1,
        max_displacement: int = 1,
        stride1: int = 1,
        stride2: int = 1,
        pad_size: int = 0,
        is_multiply: bool = True,
        qconfig=None,
    ):
        super(Correlation, self).__init__(
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            pad_size,
            is_multiply,
        )
        assert (
            qconfig and qconfig.activation is not None
        ), "qconfig and qconfig activation must be provided for QAT module"
        self.qconfig = qconfig
        self.activation_post_process = self.qconfig.activation()
        from horizon_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        self.inter_post_process = replace_qconfig_dtype(
            self.qconfig, qint16
        ).activation()
        self.inter_post_process.disable_fake_quant()
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def _fake_quanti_inter_out(self, interout, s):
        if s is None:  # float16 and float32
            return interout
        # hardware only support right shift requantize of interout
        # must guarantee inter_scale = scale1 * scale2 * 2^n (0 <= n <= 24)

        # do another scale_quanti to guarantee interout = qm * inter_scale
        interout = scale_quanti(
            interout,
            s,
            torch.zeros_like(s, dtype=torch.long),
            -1,
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            True,
            False,
        )
        interout = self.inter_post_process(interout).dequantize()
        if self.kernel_size == 1:
            # if kernel_size == 1, only do fake_quanti once
            return interout
        ss = self.inter_post_process.scale
        shift = torch.clamp(torch.ceil(torch.log2(ss / s)), 0, 24)
        ss = s * (2.0 ** shift)
        self.inter_post_process.set_qparams(ss)
        # (quantize(interout, s) >> shift) * ss may be more consistented
        # with quantized process, but can not backward
        interout = scale_quanti(
            interout,
            ss,
            torch.zeros_like(ss, dtype=torch.long),
            -1,
            -32768,
            32767,
            True,
            False,
        )
        return interout

    @typechecked
    def forward(self, data1: QTensor, data2: QTensor) -> QTensor:
        """Refine this docstring in the future.

        Args:
            data1: QTensor
            data2: QTensor

        Return:
            out: QTensor
        """
        data1, data2 = pre_process(self.activation_pre_process, data1, data2)
        out = super(Correlation, self).forward(data1, data2)
        if not torch.onnx.is_in_onnx_export():
            inter_s = data1.q_scale() * data2.q_scale()
            # do another scale_quanti to guarantee out * n = q * (s1 * s2)
            out = (
                scale_quanti(
                    out * self.sumelems,
                    inter_s,
                    torch.zeros_like(inter_s, dtype=torch.long),
                    -1,
                    torch.iinfo(torch.int32).min,
                    torch.iinfo(torch.int32).max,
                    True,
                    False,
                )
                / self.sumelems
            )
        out = self.activation_post_process(out)
        return out

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparmas_dict.

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
        qat_mod = cls(
            mod.kernel_size,
            mod.max_displacement,
            mod.stride1,
            mod.stride2,
            mod.pad_size,
            mod.is_multiply,
            mod.qconfig,
        )
        return qat_mod
