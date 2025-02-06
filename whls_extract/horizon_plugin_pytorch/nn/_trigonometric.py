import logging

import numpy as np
import torch

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from horizon_plugin_pytorch.nn.qat.qat_meta import is_float
from horizon_plugin_pytorch.qtensor import QTensor
from .segment_lut import SegmentLUT

logger = logging.getLogger(__name__)


def is_qat(x: QTensor) -> bool:
    return isinstance(x, QTensor) and not x.is_quantized


def log_warning(func_name):
    logger.warning(
        f"torch.{func_name} is detected in the model, calibration should "
        "be performed before QAT, if you have finished "
        "calibration, please ignore this warning."
    )


@replace_torch_op("tan")
class Tan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(torch.tan, True, None, None, "curvature")
        log_warning("tan")

    def qat_forward(self, x):
        scale = x.q_scale()
        dtype = x.dtype
        data = x.as_subclass(torch.Tensor)
        if not torch.all(-np.pi / 2 < data) and torch.all(data < np.pi / 2):
            msg = "tan is not supported for QAT input not in (-pi/2, pi/2)"
            logger.error(msg)
            raise RuntimeError(msg)
        eps = scale if dtype == qint8 else 32 * scale
        data = torch.clamp(data, -np.pi + eps, np.pi - eps)
        return self.func(QTensor(data, scale, dtype, x.q_per_channel_axis()))

    def forward(self, x):
        if is_qat(x) and not torch.onnx.is_in_onnx_export():
            return self.qat_forward(x)
        return self.func(x)


@replace_torch_op("sinh")
class Sinh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(torch.sinh, True, None, None, "curvature")
        log_warning("sinh")

    def forward(self, x):
        return self.func(x)


@replace_torch_op("erf")
class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(torch.erf, True)
        log_warning("erf")

    def forward(self, x):
        return self.func(x)


@replace_torch_op("cosh")
class Cosh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(torch.cosh)
        log_warning("cosh")

    def forward(self, x):
        return self.func(x)


@replace_torch_op("atanh")
class Atanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(torch.atanh, True, None, (-1, 1), "curvature")
        log_warning("atanh")

    def qat_forward(self, x):
        scale = x.q_scale()
        dtype = x.dtype
        data = x.as_subclass(torch.Tensor)
        eps = scale if dtype == qint8 else 32 * scale
        data = torch.clamp(data, -1 + eps, 1 - eps)
        return self.func(QTensor(data, scale, dtype, x.q_per_channel_axis()))

    def forward(self, x):
        if is_qat(x) and not torch.onnx.is_in_onnx_export():
            return self.qat_forward(x)
        return self.func(x)


@replace_torch_op("asinh")
class Asinh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(
            torch.asinh, True, None, None, "curvature", torch.sinh
        )
        log_warning("asinh")

    def forward(self, x):
        return self.func(x)


@replace_torch_op("asin")
@replace_torch_op("arcsin")
class Asin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(
            torch.asin,
            True,
            None,
            input_range=None,
            auto_divide_strategy="curvature",
            inverse_func=torch.sin,
        )

    def qat_forward(self, input):
        input_data = input.as_subclass(torch.Tensor)
        assert input.dtype in [
            qint8,
            qint16,
        ], "Asin only support qint8 or qint16 inputs!"
        # make sure qat input in [-1, 1]
        upper_bound = (
            1 / input.q_scale().detach()
        ).floor() * input.q_scale().detach()
        clamped_data = input_data.clamp(-upper_bound, upper_bound)

        return self.func(
            QTensor(
                clamped_data,
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        )

    def forward(self, x):
        if hasattr(self.func, "activation_post_process") and is_float(
            self.func.activation_post_process
        ):
            out = self.func(x)
            return out
        if is_qat(x) and not torch.onnx.is_in_onnx_export():
            return self.qat_forward(x)
        return self.func(x)


@replace_torch_op("acos")
class Acos(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = SegmentLUT(
            torch.acos,
            False,
            None,
            input_range=None,
            auto_divide_strategy="curvature",
            inverse_func=torch.cos,
        )

    def qat_forward(self, input):
        input_data = input.as_subclass(torch.Tensor)
        assert input.dtype in [
            qint8,
            qint16,
        ], "Acos only support qint8 or qint16 inputs!"
        # make sure qat input in [-1, 1]
        upper_bound = (
            1 / input.q_scale().detach()
        ).floor() * input.q_scale().detach()
        clamped_data = input_data.clamp(-upper_bound, upper_bound)

        return self.func(
            QTensor(
                clamped_data,
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        )

    def forward(self, x):
        if hasattr(self.func, "activation_post_process") and is_float(
            self.func.activation_post_process
        ):
            out = self.func(x)
            return out
        if is_qat(x) and not torch.onnx.is_in_onnx_export():
            return self.qat_forward(x)
        return self.func(x)


@replace_torch_op("acosh")
class Acosh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.func = SegmentLUT(
            torch.acosh, False, None, (1, np.inf), "curvature"
        )

    def forward(self, x):
        return self.func(x)
