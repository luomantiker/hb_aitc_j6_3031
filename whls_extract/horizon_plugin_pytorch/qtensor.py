import copy
import enum
import logging
import sys
from distutils.version import LooseVersion
from numbers import Real

import torch
from torch import Tensor
from torch.jit.annotations import BroadcastingList2, List, Optional
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair
from torch.overrides import handle_torch_function, has_torch_function

from horizon_plugin_pytorch.dtype import QuantDType, qinfo, qint8, qint16
from horizon_plugin_pytorch.utils import deprecated_interface_warning

__all__ = ["QTensor"]

logger = logging.getLogger(__name__)


def _unsupported(func, types, args, kwargs):
    msg = "function {} is not implemented for QTensor. ".format(func)
    msg += "Please check whether has unsupported ops in the model."
    logger.error(msg)
    raise NotImplementedError(msg)


def _qtensor_to_float(data):
    if isinstance(data, QTensor):
        return data.dequantize()
    elif isinstance(data, (list, tuple)):
        return type(data)(_qtensor_to_float(d) for d in data)
    elif isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = _qtensor_to_float(v)
        return ret
    else:
        return data


def _call_on_float(func, types, args, kwargs):
    """Call func on dequantized Tensor."""
    tensor_args = _qtensor_to_float(args)
    tensor_kwargs = _qtensor_to_float(kwargs)
    if tensor_kwargs is None:
        tensor_kwargs = {}
    return func(*tensor_args, **tensor_kwargs)


def _qtensor_to_tensor(data):
    if isinstance(data, QTensor):
        return data.as_subclass(Tensor)
    elif isinstance(data, (list, tuple)):
        return type(data)(_qtensor_to_tensor(d) for d in data)
    elif isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = _qtensor_to_tensor(v)
        return ret
    else:
        return data


def _qtensor_to_base(data):
    if isinstance(data, QTensor):
        return data._base
    elif isinstance(data, (list, tuple)):
        return type(data)(_qtensor_to_base(d) for d in data)
    elif isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            ret[k] = _qtensor_to_base(v)
        return ret
    else:
        return data


def _call_as_tensor(func, types, args, kwargs):
    """Call func as on Tensor."""
    types = (torch.Tensor for t in types)
    return torch.Tensor.__torch_function__(func, types, args, kwargs)


def _call_on_base(func, types, args, kwargs):
    """Directly call func on the underlying Tensor."""
    tensor_args = _qtensor_to_base(args)
    tensor_kwargs = _qtensor_to_base(kwargs)
    if tensor_kwargs is None:
        tensor_kwargs = {}
    return func(*tensor_args, **tensor_kwargs)


def _compare_call_on_tensor(func, types, args, kwargs):
    input1 = args[0]
    input2 = args[1]

    if isinstance(input1, (float, int)):
        return func(
            input1,
            input2.as_subclass(Tensor),
        )

    if isinstance(input2, (float, int)):
        return func(
            input1.as_subclass(Tensor),
            input2,
        )

    assert (
        input1.dtype == input2.dtype
    ), "expeted same dtype, but get one {} and another {}".format(
        input1.dtype, input2.dtype
    )

    common_scale = torch.maximum(input1.q_scale(), input2.q_scale())
    quantized_input1 = QTensor(
        input1.as_subclass(Tensor), common_scale, input1.dtype
    ).int_repr()
    quantized_input2 = QTensor(
        input2.as_subclass(Tensor), common_scale, input2.dtype
    ).int_repr()
    res = func(
        quantized_input1.as_subclass(Tensor),
        quantized_input2.as_subclass(Tensor),
    )

    return res


def _wrap_ret(func, types, args, kwargs):
    """Call func as on Tensor and wrap return value as QTensor use input scale and dtype."""  # noqa: E501
    ret = _call_as_tensor(func, types, args, kwargs)
    if isinstance(args[0], QTensor):
        return QTensor(
            ret, args[0].q_scale(), args[0].dtype, args[0].q_per_channel_axis()
        )
    else:
        return ret


def _wrap_ret_to_bool(func, types, args, kwargs):
    """Call func as on Tensor and wrap return value as QTensor use input scale and dtype."""  # noqa: E501
    ret = _call_as_tensor(func, types, args, kwargs)
    return ret


def _wrap_rets(func, types, args, kwargs):
    """Call func as on Tensor and wrap return values as QTensor use input scale and dtype."""  # noqa: E501
    rets = _call_as_tensor(func, types, args, kwargs)
    return type(rets)(
        QTensor(
            ret, args[0].q_scale(), args[0].dtype, args[0].q_per_channel_axis()
        )
        for ret in rets
    )


def _wrap_first(func, types, args, kwargs):
    """Refine this docstring in the future.

    Call func as on Tensor and wrap the first return value
    as QTensor use input scale and dtype
    """
    rets = _call_as_tensor(func, types, args, kwargs)
    return type(rets)(
        (
            QTensor(
                rets[0],
                args[0].q_scale(),
                args[0].dtype,
                args[0].q_per_channel_axis(),
            ),
        )
        + rets[1:]
    )


def _assert_scale_close(func, types, args, kwargs):
    """Call func as on Tensor and check the scale of inputs."""
    assert torch.allclose(args[0].q_scale(), args[1].q_scale())
    rets = _call_as_tensor(func, types, args, kwargs)
    return rets


def _qtensor_avg_pool2d(
    input,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[float]] = None,
    padding: BroadcastingList2[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[bool] = None,
):
    if input.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            avg_pool2d as quantized_avg_pool2d,
        )

        kernel_size = _pair(kernel_size)
        stride = kernel_size if stride is None else _pair(stride)
        padding = _pair(padding)

        ret, scale = quantized_avg_pool2d(
            input.as_subclass(Tensor),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )

        return QTensor(ret, scale, input.dtype, input.q_per_channel_axis())
    else:
        from horizon_plugin_pytorch.nn.qat.functional import (
            avg_pool2d as qat_avg_pool2d,
        )

        return qat_avg_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            None,
            True,
        )


def _qtensor_pad(
    input,
    pad: List[int],
    mode: str = "constant",
    value: Optional[float] = None,
):
    if value is None:
        value = 0.0
    if input.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            pad as quantized_pad,
        )

        res = quantized_pad(
            input.as_subclass(Tensor),
            pad,
            mode,
            value,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )
    else:
        if mode == "constant":
            from horizon_plugin_pytorch.nn.quantized.functional import quantize

            value = float(
                quantize(
                    torch.tensor([float(value)], device=input.device),
                    input.q_scale(),
                    input.q_zero_point(),
                    -1,
                    input.dtype,
                )[0]
                * input.q_scale(),
            )

        res = torch.nn.functional.pad(
            input.as_subclass(Tensor),
            pad,
            mode,
            value,
        )

    scale = input.q_scale()
    return QTensor(res, scale, input.dtype, input.q_per_channel_axis())


def _qtensor_masked_fill(
    input,
    mask: Tensor,
    value: float,
):
    assert (
        mask.dtype == torch.bool
    ), "mask is expected to be BoolTensor, but got {} instead.".format(
        mask.dtype
    )
    assert input.q_scale().numel() == 1, (
        "only per-tensor scale is supported, "
        + "and expecting scale shape to be (1,), "
        + "but got {} instead".format(input.q_scale().shape)
    )
    if input.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            masked_fill as quantized_masked_fill,
        )

        res = quantized_masked_fill(
            input.as_subclass(Tensor),
            mask,
            value,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )
    else:
        from horizon_plugin_pytorch.nn.quantized.functional import quantize

        filled_value = float(
            quantize(
                torch.tensor([float(value)], device=input.device),
                input.q_scale(),
                input.q_zero_point(),
                -1,
                input.dtype,
            )[0]
            * input.q_scale()
        )

        res = torch.masked_fill(
            input.as_subclass(Tensor),
            mask,
            filled_value,
        )

    return QTensor(
        res, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


def _qtensor_where(
    condition: Tensor,
    input,
    other,
):
    raise RuntimeError(
        "torch.where or Tensor.where is not directly supported,"
        " please use horizon_plugin_pytorch.nn.Where"
    )


def _qtensor_remainder(*args, **kwargs):
    raise RuntimeError(
        "torch.remainder or Tensor.remainder is not directly supported,"
        " please use horizon_plugin_pytorch.nn.Remainder"
    )


def _qtensor_ones_like(input, **kwargs):
    tensor = torch.ones_like(input.as_subclass(Tensor), **kwargs)
    if tensor.dtype == input.as_subclass(Tensor).dtype:
        return QTensor(
            tensor,
            torch.full_like(input.q_scale(), 1),
            input.dtype,
        )
    else:
        return tensor


def _qtensor_ones(*args, **kwargs):
    tensor = torch.ones(*args, **kwargs)
    return QTensor(
        tensor,
        torch.ones([1], dtype=torch.float32, device=tensor.device),
        qint8,
    )


def _qtensor_new_ones(input, *args, **kwargs):
    tensor = input.as_subclass(Tensor).new_ones(*args, **kwargs)
    if tensor.dtype == input.as_subclass(Tensor).dtype:
        return QTensor(
            tensor,
            torch.full_like(input.q_scale(), 1),
            input.dtype,
        )
    else:
        return tensor


def _qtensor_zeros_like(input, **kwargs):
    tensor = torch.zeros_like(input.as_subclass(Tensor), **kwargs)
    if tensor.dtype == input.as_subclass(Tensor).dtype:
        return QTensor(
            tensor,
            torch.full_like(input.q_scale(), torch.finfo(torch.float32).eps),
            input.dtype,
        )
    else:
        return tensor


def _qtensor_new_zeros(input, *args, **kwargs):
    tensor = input.as_subclass(Tensor).new_zeros(*args, **kwargs)
    if tensor.dtype == input.as_subclass(Tensor).dtype:
        return QTensor(
            tensor,
            torch.full_like(input.q_scale(), torch.finfo(torch.float32).eps),
            input.dtype,
        )
    else:
        return tensor


def _qtensor_affine_grid(
    theta, size: List[int], align_corners: Optional[bool] = None
):
    device = theta.device
    INT16_MAX = (1 << 15) - 1  # noqa: N806

    if theta.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            matmul,
            requantize,
        )

        N, C, H, W = size  # noqa: N806

        x = (
            torch.linspace(
                -INT16_MAX,
                INT16_MAX,
                W,
                dtype=torch.int16,
                device=device,
            )
            .unsqueeze(0)
            .expand(H, W)
        )
        y = (
            torch.linspace(
                -INT16_MAX,
                INT16_MAX,
                H,
                dtype=torch.int16,
                device=device,
            )
            .unsqueeze(-1)
            .expand(H, W)
        )
        ones = torch.full((H, W), INT16_MAX, dtype=torch.int16, device=device)

        if not align_corners:
            x = (x * ((W - 1) / W)).round().to(dtype=torch.int16)
            y = (y * ((H - 1) / H)).round().to(dtype=torch.int16)

        base_grid = (
            torch.stack([x, y, ones], dim=-1).unsqueeze(0).expand(N, H, W, 3)
        )
        base_grid_scale = torch.tensor(
            [1 / INT16_MAX], dtype=torch.float32, device=device
        )
        grid_scale = torch.tensor(
            [2 / INT16_MAX], dtype=torch.float32, device=device
        )

        theta = theta.reshape(N, 1, 2, 3)

        if theta.dtype != "qint16":
            theta = QTensor(
                requantize(
                    theta.as_subclass(Tensor),
                    theta.q_scale(),
                    theta.q_zero_point(),
                    theta.dtype,
                    theta.q_scale() / (1 << 8),
                    theta.q_zero_point(),
                    "qint16",
                ),
                theta.q_scale() / (1 << 8),
                "qint16",
            )

        grid = matmul(
            base_grid.reshape(N, 1, H * W, 3),
            theta.as_subclass(Tensor),
            False,
            True,
            base_grid_scale,
            theta.q_zero_point(),
            "qint16",
            theta.q_scale(),
            theta.q_zero_point(),
            "qint16",
            grid_scale,
            theta.q_zero_point(),
            "qint16",
        ).reshape(N, H, W, 2)

        return QTensor(grid, grid_scale, "qint16")

    else:
        from horizon_plugin_pytorch.nn.qat.functional import scale_quanti

        grid = F.affine_grid(theta.as_subclass(Tensor), size, align_corners)
        scale = torch.tensor(
            [2 / INT16_MAX], dtype=torch.float32, device=device
        )
        zero_point = torch.zeros(1, dtype=torch.long, device=device)

        info = qinfo("qint16")

        return QTensor(
            scale_quanti(
                grid,
                scale,
                zero_point,
                -1,
                info.min,
                info.max,
                True,
                False,
            ),
            scale,
            "qint16",
        )


def _qtensor_mul_scalar_guard(input, other, *, out=None):
    return isinstance(other, (int, float))


def _qtensor_mul_scalar(input, other, *, out=None):
    assert out is None
    assert isinstance(other, (int, float))
    other_scale = abs(other)
    other_data = 0 if other == 0 else other / other_scale

    if input.is_quantized:
        # if 0 or 1, directly return QTensor, avoid extra 'mul' on HW
        if other_data == 0:
            r = torch.zeros_like(input.as_subclass(Tensor))
        elif other_data == 1:
            r = input.as_subclass(Tensor)
        else:
            from horizon_plugin_pytorch.nn.quantized.functional import mul

            r = mul(
                input.as_subclass(Tensor),
                torch.tensor([other_data]).to(input.as_subclass(Tensor)),
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                torch.tensor([other_scale], dtype=torch.float32).to(
                    input.device
                ),
                input.q_zero_point(),
                input.dtype,
                input.q_scale() * other_scale,
                input.q_zero_point(),
                input.dtype,
            )
        return QTensor(
            r,
            (input.q_scale() * other_scale).clamp_min(
                torch.finfo(torch.float32).eps
            ),
            input.dtype,
            input.q_per_channel_axis(),
        )
    else:
        return QTensor(
            input.as_subclass(Tensor) * other,
            (input.q_scale() * other_scale).clamp_min(
                torch.finfo(torch.float32).eps
            ),
            input.dtype,
            input.q_per_channel_axis(),
        )


def _qtensor_clamp(input, min=None, max=None, *, out=None):
    # min and max could be Number, Tensor and None
    # if min or max is constant Tensor, float and qat results may diff much
    # in the case min < max < input or input < min < max because clamp to input
    # range operation in fake quant of min and max
    info = qinfo(input.dtype)
    if input.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            quantize,
            requantize,
        )

        if isinstance(min, QTensor):
            min = requantize(
                min.int_repr(),
                min.q_scale(),
                min.q_zero_point(),
                min.dtype,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
            )
        elif isinstance(min, Tensor):
            min = quantize(
                min,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                input.dtype,
            )
        elif isinstance(min, Real):
            # keep min type
            min = quantize(
                torch.tensor([float(min)]),
                input.q_scale().cpu(),
                input.q_zero_point().cpu(),
                input.q_per_channel_axis(),
                input.dtype,
            ).item()
        else:
            assert min is None, "Only support min type: Number, Tensor, None"

        if isinstance(max, QTensor):
            max = requantize(
                max.int_repr(),
                max.q_scale(),
                max.q_zero_point(),
                max.dtype,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
            )
        elif isinstance(max, Tensor):
            max = quantize(
                max,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                input.dtype,
            )
        elif isinstance(max, Real):
            # keep max type
            max = quantize(
                torch.tensor([float(max)]),
                input.q_scale().cpu(),
                input.q_zero_point().cpu(),
                input.q_per_channel_axis(),
                input.dtype,
            ).item()
        else:
            assert max is None, "Only support max type: Number, Tensor, None"

        r = torch.clamp(input.int_repr(), min, max)
        return QTensor(
            r, input.q_scale(), input.dtype, input.q_per_channel_axis()
        )
    else:
        from horizon_plugin_pytorch.nn.qat.functional import scale_quanti

        if isinstance(min, QTensor):
            min = min.as_subclass(Tensor)
        elif isinstance(min, Tensor):

            min = scale_quanti(
                min,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
            )
        elif isinstance(min, Real):
            # keep min type
            min = scale_quanti(
                torch.tensor([float(min)]),
                input.q_scale().cpu(),
                input.q_zero_point().cpu(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
            ).item()
        else:
            assert min is None, "Only support min type: Number, Tensor, None"

        if isinstance(max, QTensor):
            max = max.as_subclass(Tensor)
        elif isinstance(max, Tensor):

            max = scale_quanti(
                max,
                input.q_scale(),
                input.q_zero_point(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
            )
        elif isinstance(max, Real):
            # keep max type
            max = scale_quanti(
                torch.tensor([float(max)]),
                input.q_scale().cpu(),
                input.q_zero_point().cpu(),
                input.q_per_channel_axis(),
                info.min,
                info.max,
                True,
                False,
            ).item()
        else:
            assert max is None, "Only support max type: Number, Tensor, None"

        return QTensor(
            torch.clamp(input.as_subclass(Tensor), min, max),
            input.q_scale(),
            input.dtype,
            input.q_per_channel_axis(),
        )


def _qtensor_channel_shuffle(input, groups: int):
    from horizon_plugin_pytorch.nn.functional import channel_shuffle

    return channel_shuffle(input, groups)


def _qtensor_pixel_shuffle(input, upscale_factor: int):
    out = F.pixel_shuffle(input.as_subclass(Tensor), upscale_factor)
    return QTensor(
        out, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


def _qtensor_pixel_unshuffle(input, downscale_factor: int):
    out = F.pixel_unshuffle(input.as_subclass(Tensor), downscale_factor)
    return QTensor(
        out, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


def _qtensor_topk(
    input,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
):
    assert input.q_scale().numel() == 1, "only support per-tensor scale!"
    from horizon_plugin_pytorch.nn.quantized.functional import topk

    # The quantized implementation is compatible with qat input
    output, indices = topk(input.as_subclass(Tensor), k, dim, largest, sorted)
    return (
        QTensor(
            output, input.q_scale(), input.dtype, input.q_per_channel_axis()
        ),
        indices,
    )


def _qtensor_requires_grad_(input):
    torch.Tensor.__torch_function__(
        torch.Tensor.requires_grad_, (torch.Tensor,), (input,), {}
    )
    return input


def _qtensor_grad_get_(input):
    grad = torch.Tensor.__torch_function__(
        torch.Tensor.grad.__get__, (torch.Tensor,), (input,), {}
    )
    # Construct QTensor on a Tensor requires grad makes QTensor non-leaf,
    # so maybe get grad from its base.
    if grad is None and input._base.requires_grad:
        grad = input._base.grad
    return grad


def _qtensor_abs(input, overflow_mode: str = "saturate"):
    assert input.dtype in [
        qint8,
        qint16,
    ], "only support input dtype qint8 or qint16!"
    from horizon_plugin_pytorch.nn.quantized import functional

    input_data = input.as_subclass(Tensor)
    if input.is_quantized:
        input_data = functional.abs(input_data, overflow_mode)
    # qat
    else:
        if overflow_mode == "saturate":
            input_data = torch.clamp_min(
                input_data, -input.dtype.max * input.q_scale()
            )

        elif overflow_mode == "trunc":
            pass

        else:
            raise ValueError(
                "Unsupported overflow mode! "
                "Only 'saturate' or 'trunc' allowed!"
            )
        input_data = torch.abs(input_data)

    return QTensor(
        input_data,
        input.q_scale(),
        input.dtype,
        input.q_per_channel_axis(),
    )


def _qtensor_relu(input, inplace: bool = False):
    out = F.relu(input.as_subclass(Tensor), inplace)
    return QTensor(
        out, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


def _qtensor_setitem(*args, **kwargs):
    raise RuntimeError(
        "Tensor.__setitem__ is not directly supported,"
        " please use horizon_plugin_pytorch.nn.SetItem"
    )


def _qtensor_masked_scatter(*args, **kwargs):
    raise RuntimeError(
        "Tensor.masked_scatter is not directly supported,"
        " please use horizon_plugin_pytorch.nn.SetItem"
    )


def _refuse_all_guard(*args, **kwargs):
    return False


def _qtensor_max(input, *args, **kwargs):
    rets = input.as_subclass(Tensor).max(*args, **kwargs)

    if isinstance(rets, tuple):
        return type(rets)(
            (
                QTensor(
                    rets[0],
                    input.q_scale(),
                    input.dtype,
                    input.q_per_channel_axis(),
                ),
                rets[1],
            )
        )
    else:
        return QTensor(
            rets,
            input.q_scale(),
            input.dtype,
            input.q_per_channel_axis(),
        )


def _qtensor_min(input, *args, **kwargs):
    rets = input.as_subclass(Tensor).min(*args, **kwargs)

    if isinstance(rets, tuple):
        return type(rets)(
            (
                QTensor(
                    rets[0],
                    input.q_scale(),
                    input.dtype,
                    input.q_per_channel_axis(),
                ),
                rets[1],
            )
        )
    else:
        return QTensor(
            rets,
            input.q_scale(),
            input.dtype,
            input.q_per_channel_axis(),
        )


def _qtensor_to(input, *args, **kwargs):
    tensor_args = _qtensor_to_base(args)
    ret: Tensor = input.dequantize().to(*tensor_args, **kwargs)
    if ret.is_floating_point():
        return QTensor(
            ret, input.q_scale(), input.dtype, input.q_per_channel_axis()
        )
    else:
        return ret


def _qtensor_sign(input, *args, **kwargs):
    ret = torch.sign(input.as_subclass(Tensor))
    return QTensor(
        ret.to(torch.int8) if input.is_quantized else ret,
        torch.ones(1, device=input.device),
        qint8,
    )


def _qtensor_new_tensor(input, *args, **kwargs):
    assert not isinstance(
        args[0], Tensor
    ), "new_tensor arg must be array_like constant values, like [1., 2.]."
    return input.as_subclass(torch.Tensor).new_tensor(*args, **kwargs)


def _qtensor_float(input, *args, **kwargs):
    if input.is_quantized:
        # quantized qtensor, do nothing
        return input
    else:
        return QTensor(
            input.as_subclass(Tensor).float(),
            input.q_scale(),
            input.dtype,
            input.q_per_channel_axis(),
        )


_CALL_AS_TENSOR = [
    torch._C._TensorBase.argmax,
    torch._C._TensorBase.argmin,
    torch._C._TensorBase.device.__get__,
    torch._C._TensorBase.dim,
    torch._C._TensorBase.get_device,
    torch._C._TensorBase.is_cuda.__get__,
    torch._C._TensorBase.is_contiguous,
    torch._C._TensorBase.numel,
    torch._C._TensorBase.shape.__get__,
    torch._C._TensorBase.size,
    torch._C._TensorBase.ndim.__get__,
    torch._C._TensorBase.is_mkldnn.__get__,
    torch._C._TensorBase.is_complex,
    torch.argmax,
    torch.argmin,
    torch.argsort,
    torch.all,
    Tensor.backward,
    Tensor.grad_fn.__get__,
    torch.is_same_size,
    torch._C._TensorBase.requires_grad.__get__,
    torch._C._TensorBase.requires_grad.__set__,
    torch._C._TensorBase.grad_fn.__set__,
    torch.Tensor.all,
    torch.Tensor.argsort,
    torch.Tensor.retain_grad,
    torch.Tensor.double,
    torch.Tensor._backward_hooks.__get__,
    torch.Tensor._backward_hooks.__set__,
    Tensor.__len__,
]

if LooseVersion(torch.__version__.split("+")[0]) >= LooseVersion("1.13.0"):
    _CALL_AS_TENSOR.append(torch._C._TensorBase.is_mps.__get__)


_CALL_ON_FLOAT = [
    torch.Tensor.long,
    torch.Tensor.int,
]


_COMPARE_CALL_ON_TENSOR = [
    torch._C._TensorBase.eq,
    torch._C._TensorBase.gt,
    torch._C._TensorBase.greater,
    torch._C._TensorBase.ge,
    torch._C._TensorBase.greater_equal,
    torch._C._TensorBase.lt,
    torch._C._TensorBase.less,
    torch._C._TensorBase.le,
    torch._C._TensorBase.less_equal,
    torch.eq,
    # torch.equal return a bool!!!
    torch.gt,
    torch.greater,
    torch.ge,
    torch.greater_equal,
    torch.less,
    torch.le,
    torch.less_equal,
    torch.lt,
]


# In dictionary order
_WRAP_RET = [
    torch._C._TensorBase.__getitem__,
    torch._C._TensorBase.contiguous,
    torch._C._TensorBase.detach,
    torch._C._TensorBase.expand,
    torch._C._TensorBase.flatten,
    torch._C._TensorBase.permute,
    torch._C._TensorBase.repeat,
    torch._C._TensorBase.reshape,
    torch._C._TensorBase.roll,
    torch._C._TensorBase.squeeze,
    torch._C._TensorBase.tile,
    torch._C._TensorBase.transpose,
    torch._C._TensorBase.unsqueeze,
    torch._C._TensorBase.view,
    torch._C._TensorBase.clone,
    torch._C._TensorBase.gather,
    torch._C._TensorBase.T.__get__,
    torch._C._TensorBase.masked_select,
    torch.Tensor.cpu,
    torch.Tensor.cuda,
    torch.Tensor.neg,
    torch.Tensor.negative,
    torch.Tensor.bfloat16,
    torch.Tensor.index_select,
    torch.Tensor.expand_as,
    torch.Tensor.repeat_interleave,
    torch.Tensor.flip,
    torch.Tensor.tril,
    torch.Tensor.triu,
    torch.Tensor.t,
    torch.repeat_interleave,
    torch.flatten,
    torch.permute,
    torch.reshape,
    torch.roll,
    torch.squeeze,
    torch.t,
    torch.tile,
    torch.transpose,
    torch.unsqueeze,
    torch.clone,
    torch.gather,
    torch.neg,
    torch.negative,
    torch.index_select,
    torch.flip,
    torch.tril,
    torch.triu,
    torch.masked_select,
    F.max_pool1d,
    F.max_pool2d,
    F.adaptive_max_pool1d,
    F.adaptive_max_pool2d,
    F.dropout,
    F.dropout1d,
    F.dropout2d,
    F.dropout3d,
]

_WRAP_RET_TO_BOOL = [
    torch.Tensor.logical_and,
    torch.Tensor.logical_or,
    torch.Tensor.logical_not,
    torch.logical_and,
    torch.logical_or,
    torch.logical_not,
]

_WRAP_RETS = [
    torch.split,
    torch.Tensor.split,
    torch.unbind,
    torch.Tensor.unbind,
    torch.chunk,
    torch.Tensor.chunk,
]

_WRAP_FIRST = [
    F.max_pool1d_with_indices,
    F.max_pool2d_with_indices,
    F.adaptive_max_pool1d_with_indices,
    F.adaptive_max_pool2d_with_indices,
    torch.sort,
    Tensor.sort,
]


# In dictionary order
_FUNC_MAPPING = {
    torch.Tensor.to: _qtensor_to,
    torch.Tensor.requires_grad_: _qtensor_requires_grad_,
    torch.Tensor.grad.__get__: _qtensor_grad_get_,
    torch._C._TensorBase.clamp: _qtensor_clamp,
    torch._C._TensorBase.clip: _qtensor_clamp,
    torch._C._TensorBase.masked_fill: _qtensor_masked_fill,
    torch._C._TensorBase.mul: (_qtensor_mul_scalar, _qtensor_mul_scalar_guard),
    torch._C._TensorBase.sign: _qtensor_sign,
    torch.max: _qtensor_max,
    torch.Tensor.max: _qtensor_max,
    torch.min: _qtensor_min,
    torch.Tensor.min: _qtensor_min,
    torch._C._TensorBase.topk: _qtensor_topk,
    torch._C._TensorBase.abs: _qtensor_abs,
    torch.Tensor.__setitem__: (_qtensor_setitem, _refuse_all_guard),
    torch.Tensor.where: (_qtensor_where, _refuse_all_guard),
    torch.Tensor.remainder: (_qtensor_remainder, _refuse_all_guard),
    torch.remainder: (_qtensor_remainder, _refuse_all_guard),
    torch.Tensor.masked_scatter: (_qtensor_masked_scatter, _refuse_all_guard),
    torch.masked_scatter: (_qtensor_masked_scatter, _refuse_all_guard),
    torch.clamp: _qtensor_clamp,
    torch.clip: _qtensor_clamp,
    torch.masked_fill: _qtensor_masked_fill,
    torch.mul: (_qtensor_mul_scalar, _qtensor_mul_scalar_guard),
    torch.ones_like: _qtensor_ones_like,
    torch.ones: _qtensor_ones,
    Tensor.new_ones: _qtensor_new_ones,
    torch.topk: _qtensor_topk,
    torch.zeros_like: _qtensor_zeros_like,
    Tensor.new_zeros: _qtensor_new_zeros,
    torch.abs: _qtensor_abs,
    torch.where: (_qtensor_where, _refuse_all_guard),
    torch.sign: _qtensor_sign,
    Tensor.new_tensor: _qtensor_new_tensor,
    torch.Tensor.float: _qtensor_float,
    # functional
    F.avg_pool2d: _qtensor_avg_pool2d,
    F.channel_shuffle: _qtensor_channel_shuffle,
    F.affine_grid: _qtensor_affine_grid,
    F.pixel_shuffle: _qtensor_pixel_shuffle,
    F.pixel_unshuffle: _qtensor_pixel_unshuffle,
    F.pad: _qtensor_pad,
    F.relu: _qtensor_relu,
}


def copy_from(dst, src):
    assert isinstance(dst, QTensor)
    assert isinstance(src, QTensor)

    if has_torch_function((dst, src)):
        return handle_torch_function(copy_from, (dst, src), dst, src)

    raise RuntimeError("Should not be here")


class QTensor(Tensor):
    # A mapping from torch func to its QTensor implementation
    _DISPATCHER: dict = {}
    # the args of guard func should be (func, types, args, kwargs)
    # NOTE: args and kwargs could be fx.Node !!!
    _DISPATCHER_GUARD: dict = {}
    # Whether allow float operation directly applied on QTensor
    _allow_float_operation = False

    class DispatchMode(enum.Enum):
        """Predefined func wrap mode for `register_func_impl`."""

        CALL_AS_TENSOR = enum.auto()
        WRAP_RET = enum.auto()
        WRAP_RETS = enum.auto()
        WRAP_FIRST = enum.auto()
        WRAP_RET_TO_BOOL = enum.auto()
        COMPARE_CALL_ON_TENSOR = enum.auto()
        CALL_ON_FLOAT = enum.auto()

    _DISPATCH_MAPPIMG = {
        DispatchMode.CALL_AS_TENSOR: _call_as_tensor,
        DispatchMode.WRAP_RET: _wrap_ret,
        DispatchMode.WRAP_RETS: _wrap_rets,
        DispatchMode.WRAP_FIRST: _wrap_first,
        DispatchMode.WRAP_RET_TO_BOOL: _wrap_ret_to_bool,
        DispatchMode.COMPARE_CALL_ON_TENSOR: _compare_call_on_tensor,
        DispatchMode.CALL_ON_FLOAT: _call_on_float,
    }

    def __new__(cls, data, scale, dtype, per_channel_axis=-1):
        """Generate a QTensor with quantized data.

        Args:
            data (Tensor): Quantized int data or float data from fake quanti.
            scale (Tensor): Scale.
            dtype (str): Quantize type.
            per_channel_axis (int, optional): The channel axis for per channel
                quantized data, -1 for per tensor quanti. Defaults to -1.

        Returns:
            QTensor
        """
        if scale is not None and scale.numel() > 1:
            assert per_channel_axis > -1, (
                "Please specify per_channel_axis "
                + "for per channel quantized QTensor"
                + "receive scale: {}".format(scale)
            )
        if (
            per_channel_axis > -1
            and not torch.jit.is_scripting()
            and not torch._C._get_tracing_state()
        ):
            assert scale.numel() == data.size(per_channel_axis), (
                "Invalid scale size for per channel quantized QTensor, "
                + "data shape is {} but scale shape is {} (ch_axis={})".format(
                    data.shape, scale.shape, per_channel_axis
                )
            )

        instance = data.as_subclass(cls)
        instance._scale = scale
        instance._zero_point = (
            None
            if scale is None
            else torch.zeros_like(scale, dtype=torch.long)
        )
        # we cannot rewrite Tensor.dtype
        instance._qtype = (
            QuantDType(dtype) if not isinstance(dtype, QuantDType) else dtype
        )
        instance._per_channel_axis = per_channel_axis

        return instance

    @classmethod
    def init_class(cls):
        # make all interface of QTensor call through __torch_functional__
        qtensor_torch_func = {
            Tensor.is_quantized.__get__: QTensor.is_quantized.__get__,
            Tensor.__repr__: QTensor.__repr__,
            Tensor.qscheme: QTensor.qscheme,
            Tensor.q_scale: QTensor.q_scale,
            Tensor.q_zero_point: QTensor.q_zero_point,
            Tensor.q_per_channel_scales: QTensor.q_per_channel_scales,
            Tensor.q_per_channel_zero_points: (
                QTensor.q_per_channel_zero_points
            ),
            Tensor.q_per_channel_axis: QTensor.q_per_channel_axis,
            Tensor.dtype.__get__: QTensor.dtype.__get__,
            Tensor.dequantize: QTensor.dequantize,
            Tensor.int_repr: QTensor.int_repr,
            copy_from: QTensor._copy_from,
        }
        for k, v in qtensor_torch_func.items():
            cls.register_func_impl(k)(v)

        for f in _CALL_AS_TENSOR:
            cls.register_func_impl(f)(cls.DispatchMode.CALL_AS_TENSOR)
        for f in _CALL_ON_FLOAT:
            cls.register_func_impl(f)(cls.DispatchMode.CALL_ON_FLOAT)
        for f in _COMPARE_CALL_ON_TENSOR:
            cls.register_func_impl(f)(cls.DispatchMode.COMPARE_CALL_ON_TENSOR)
        for f in _WRAP_RET:
            cls.register_func_impl(f)(cls.DispatchMode.WRAP_RET)
        for f in _WRAP_RET_TO_BOOL:
            cls.register_func_impl(f)(cls.DispatchMode.WRAP_RET_TO_BOOL)
        for f in _WRAP_RETS:
            cls.register_func_impl(f)(cls.DispatchMode.WRAP_RETS)
        for f in _WRAP_FIRST:
            cls.register_func_impl(f)(cls.DispatchMode.WRAP_FIRST)
        for k, v in _FUNC_MAPPING.items():
            cls.register_func_impl(k)(v)

    def __deepcopy__(self, memo):
        return QTensor(
            copy.deepcopy(self.as_subclass(Tensor)),
            copy.deepcopy(self.q_scale()),
            copy.deepcopy(self.dtype),
            self.q_per_channel_axis(),
        )

    def _copy_from(self, src):
        self.as_subclass(Tensor).copy_(src.as_subclass(Tensor))
        self._scale = src.q_scale()
        self._zero_point = src._zero_point
        self._qtype = src._qtype
        self._per_channel_axis = src._per_channel_axis

        return self

    @property
    def dtype(self) -> QuantDType:
        """Quanti type."""
        return self._qtype

    @property
    def scale(self) -> Tensor:
        deprecated_interface_warning("2.1.3", "2.4.0", QTensor.q_scale)
        return self._scale

    @property
    def zero_point(self) -> Tensor:
        deprecated_interface_warning("2.1.3", "2.4.0", QTensor.q_zero_point)
        return self._zero_point

    @property
    def qtype(self) -> str:
        return self._qtype

    @property
    def per_channel_axis(self) -> int:
        deprecated_interface_warning(
            "2.1.3", "2.4.0", QTensor.q_per_channel_axis
        )
        return self._per_channel_axis

    @property
    def is_quantized(self) -> bool:
        """Is True if the Tensor is quantized, False otherwise."""
        return not self.as_subclass(Tensor).is_floating_point()

    def is_nested(self) -> bool:
        """Is True if the Tensor is nested, False otherwise."""
        return False

    def dequantize(self) -> Tensor:
        """Return the dequantized float Tensor.

        Returns:
            Tensor
        """
        if self.is_quantized:
            from .nn.quantized.functional import dequantize

            return dequantize(
                self.as_subclass(Tensor),
                self.q_scale(),
                self.q_zero_point(),
                self.q_per_channel_axis(),
            )
        else:
            return self.as_subclass(Tensor)

    def int_repr(self) -> Tensor:
        """Refine this docstring in the future.

        Return the quantized int Tensor

        Returns:
            Tensor
        """
        if self.is_quantized:
            return self.as_subclass(Tensor)
        else:
            from .nn.quantized.functional import quantize

            return quantize(
                self.as_subclass(Tensor),
                self.q_scale(),
                self.q_zero_point(),
                self.q_per_channel_axis(),
                self.dtype,
            )

    def qscheme(self):
        """Get the quantization scheme of a given QTensor."""
        if self.q_scale().numel() == 1:
            if self.q_zero_point().prod() == 0:
                return torch.per_tensor_symmetric
            else:
                return torch.per_tensor_affine
        else:
            if self.q_zero_point().prod() == 0:
                return torch.per_channel_symmetric
            else:
                return torch.per_channel_affine

    def q_scale(self) -> Tensor:
        """Refine this docstring in the future.

        Given a Tensor quantized by linear(affine) quantization,
        returns the scale of the underlying quantizer().

        Returns:
            Tensor
        """
        return self._scale

    def q_zero_point(self) -> Tensor:
        """Refine this docstring in the future.

        Given a Tensor quantized by linear(affine) quantization,
        returns the zero_point of the underlying quantizer().

        Returns:
            Tensor
        """
        return self._zero_point

    def q_per_channel_scales(self) -> Tensor:
        """Refine this docstring in the future.

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a Tensor of scales of the underlying quantizer.
        It has the number of elements that matches the
        corresponding dimensions (from q_per_channel_axis) of the tensor.

        Returns:
            Tensor
        """
        return self.q_scale()

    def q_per_channel_zero_points(self) -> Tensor:
        """Refine this docstring in the future.

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a tensor of zero_points of the underlying quantizer.
        It has the number of elements that matches the
        corresponding dimensions (from q_per_channel_axis) of the tensor.

        Returns:
            Tensor
        """
        return self.q_zero_point()

    def q_per_channel_axis(self) -> int:
        """Refine this docstring in the future.

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns the index of dimension on which per-channel
        quantization is applied.

        Returns:
            int
        """
        return self._per_channel_axis

    def to_fake_quantized(self):
        if self.is_quantized:
            return QTensor(
                self.dequantize(),
                self.q_scale(),
                self.dtype,
                self.q_per_channel_axis(),
            )
        else:
            return self

    def to_quantized(self):
        if self.is_quantized:
            return self
        else:
            return QTensor(
                # if last conv scale = None, just keep float data
                (
                    self.int_repr()
                    if self.q_scale() is not None
                    else self.as_subclass(Tensor)
                ),
                self.q_scale(),
                self.dtype,
                self.q_per_channel_axis(),
            )

    @property
    def _base(self):
        return Tensor.__torch_function__(
            Tensor._base.__get__, (Tensor,), (self,)
        )

    @classmethod
    def get_dispatched_func(cls, func, types, args, kwargs):
        wrapped_func = cls._DISPATCHER.get(func, None)
        guard = cls._DISPATCHER_GUARD.get(func, None)

        if guard is not None:
            if not guard(func, types, args, kwargs):
                wrapped_func = None

        return wrapped_func

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        wrapped_func = cls.get_dispatched_func(func, types, args, kwargs)

        try:
            if wrapped_func is not None:
                return wrapped_func(func, types, args, kwargs)
            elif cls._allow_float_operation:
                return _call_on_float(func, types, args, kwargs)
            else:
                return _unsupported(func, types, args, kwargs)
        except Exception as e:
            raise type(e)(
                str(e.args[0]) + "\n when calling function {}".format(func)
            ).with_traceback(sys.exc_info()[-1])

    @classmethod
    def register_func_impl(cls, symbol):
        """Register function impl for QTensor.

        The impl can be:
        1. Callable: The func implementation on QTensor.
        2. DispatchMode: A enum to indicate one kind of warp method.
        3. Tuple: Its first element should be the implementation,
           and second element should be a guard to check if input args can be
           handled by the implementation. The args of guard shoud be the same
           with the symbol.

        Args:
            symbol (callable): The Tensor func to be dispatched.

        Example::

            # Use as decorator when define the func impl for subclass
            @DispatchedTensor.register_func_impl(torch_func)
            def func_impl_for_tensor_subclass():
                ...

            # Use with DispatchMode
            DispatchedTensor.register_func_impl(DispatchMode.ENUM)

            # Use with guard
            @DispatchedTensor.register_func_impl(torch.mul)(
                qtensor_mul_scalar,
                qtensor_mul_scalar_guard,
            )
        """

        def inner(impl):
            if symbol in cls._DISPATCHER:
                raise ValueError("Symbol {} already registered".format(symbol))

            if isinstance(impl, tuple):
                impl, guard = impl
                cls._DISPATCHER_GUARD[symbol] = cls.wrap_torch_function(guard)

            if isinstance(impl, cls.DispatchMode):
                impl = cls._DISPATCH_MAPPIMG[impl]
            else:
                impl = cls.wrap_torch_function(impl)

            cls._DISPATCHER[symbol] = impl

            return impl

        return inner

    @classmethod
    def get_dispatcher(cls):
        """Get current dispatcher."""
        return cls._DISPATCHER

    @classmethod
    def allow_float_operation(cls, enabled: bool):
        """Whether allow to directly use QTensor as input of float operations.

        The default behaviour is False.
        """
        assert isinstance(enabled, bool)
        cls._allow_float_operation = enabled

    @classmethod
    def patch_torch_func(cls, torch_func, patcher_func):
        """Update dispatcher when monkey patching torch func."""
        impl = cls._DISPATCHER.get(torch_func, None)
        if impl is None:
            raise ValueError(
                "Trying to copy dispatched impl of {} to {}, but cannot find"
                " {} in dispatcher.".format(
                    torch_func, patcher_func, torch_func
                )
            )
        cls._DISPATCHER[patcher_func] = impl

        guard = cls._DISPATCHER_GUARD.get(torch_func, None)
        if guard is not None:
            cls._DISPATCHER_GUARD[patcher_func] = guard

    @staticmethod
    def wrap_torch_function(f):
        def wrapped_func(func, types, args, kwargs):
            return f(*args, **kwargs)

        return wrapped_func

    def __repr__(self, *, tensor_contents=None):
        return (
            "QTensor(\n  data = {},\n  scale = {},\n  zero_point = {},\n  "
            + "dtype = {},\n  per_channel_axis = {},\n  is_quantized = {}\n)"
        ).format(
            self.as_subclass(Tensor),
            self.q_scale(),
            self.q_zero_point(),
            self.dtype,
            self.q_per_channel_axis(),
            self.is_quantized,
        )

    def __reduce_ex__(self, proto):
        return (
            _build_qtensor_from_args,
            (
                self.as_subclass(Tensor),
                self.q_scale(),
                self.dtype,
                self.q_per_channel_axis(),
            ),
        )


def _build_qtensor_from_args(data, scale, dtype, per_channel_axis):
    return QTensor(data, scale, dtype, per_channel_axis)


QTensor.init_class()
