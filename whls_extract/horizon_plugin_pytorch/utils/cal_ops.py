import inspect
from functools import reduce
from typing import Any

import torch

import horizon_plugin_pytorch.nn.qat.conv2d as qat_conv2d
import horizon_plugin_pytorch.nn.qat.conv3d as qat_conv3d
import horizon_plugin_pytorch.nn.qat.conv_bn2d as qat_conv_bn2d
import horizon_plugin_pytorch.nn.qat.conv_transpose2d as qat_convtranspose2d
import horizon_plugin_pytorch.nn.qat.linear as qat_linear
from horizon_plugin_pytorch.nn.qat import FloatFunctional
from horizon_plugin_pytorch.utils.model_helper import _as_tuple

__all__ = ["cal_flops"]

_CALOPS_MAP = {}


def register_calops_func(modules):
    def wrapper(func):
        mods = (modules,) if not isinstance(modules, tuple) else modules
        for mod in mods:
            if isinstance(mod, str):
                _CALOPS_MAP[mod] = func
            else:
                pkg_name = mod.__name__
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    if cls.__module__.startswith(pkg_name):
                        _CALOPS_MAP[cls] = func

    return wrapper


@register_calops_func(qat_linear)
def count_linear(m, x, y):
    # N x Cout x H x Cin
    return int(y.numel() * m.weight.shape[-1])


@register_calops_func((qat_conv2d, qat_conv3d, qat_conv_bn2d))
def count_convnd(m, x, y):
    # for conv2d, conv3d
    kernel_ops = reduce(lambda x, y: x * y, m.weight.size()[2:])  # Kw x Kh

    # N x Cout x H x W x (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (x.shape[1] // m.groups * kernel_ops)
    return int(total_ops)


@register_calops_func(qat_convtranspose2d)
def count_convtranspose2d(m, x, y):
    kernel_ops = reduce(lambda x, y: x * y, m.weight.size()[2:])  # Kw x Kh

    # N x Cout x Hin x Win x Cin x Kw x Kh
    total_ops = x.nelement() * (y.shape[1] // m.groups * kernel_ops)
    return int(total_ops)


@register_calops_func("matmul")
def count_matmul(input, output):
    in1hw = {input[0].shape[-1], input[0].shape[-2]}
    in2hw = {input[1].shape[-1], input[1].shape[-2]}
    outhw = {output.shape[-1], output.shape[-2]}
    inhw = in1hw | in2hw
    if len(inhw) == 1:
        num_mul = list(inhw)[0]
    elif len(inhw) == 2:
        if len(outhw) == 1:
            num_mul = list(inhw - outhw)[0]
        elif len(outhw) == 2:
            assert len(in1hw) == 1 or len(in2hw) == 1
            num_mul = list(in1hw)[0] if len(in1hw) == 1 else list(in2hw)[0]
        else:
            raise RuntimeError(f"Unexpected output hw {outhw}!")
    elif len(inhw) == 3:
        num_mul = list(in1hw & in2hw)[0]
    else:
        raise RuntimeError(f"Unexpected input hw {inhw}")
    return output.numel() * num_mul


def cal_flops(
    model: torch.nn.Module,
    example_inputs: Any,
):
    """Calculate flops of model (only calculate TAE ops now).

    Args:
        model: qat model
        example_inputs: model input
    """
    dtype_flops_mapping = {}
    op_flops_mapping = {}

    def append_flops(name, xdtype, ydtype, flops):
        key = f"{xdtype}_{ydtype}"
        dtype_flops_mapping[key] = dtype_flops_mapping.get(key, 0) + flops
        op_flops_mapping[name] = op_flops_mapping.get(name, 0) + flops

    def _hook(module, input, output, name):
        if type(module) in list(_CALOPS_MAP.keys()):
            if hasattr(module, "_swap_inputs") and module._swap_inputs:
                conv_input = input[1]
            else:
                conv_input = input[0]
            append_flops(
                name,
                conv_input.dtype,
                module.weight_fake_quant.dtype,
                flops=_CALOPS_MAP[type(module)](module, conv_input, output),
            )
        elif (
            isinstance(module, FloatFunctional)
            and module._last_called_method_name == "matmul"
        ):
            append_flops(
                name,
                input[0].dtype,
                input[1].dtype,
                flops=_CALOPS_MAP["matmul"](input, output),
            )

    def _register_hooks(module, name):
        module.register_forward_hook(
            lambda mod, input, output: _hook(mod, input, output, name)
        )

    # register forward hook
    for name, m in model.named_modules():
        _register_hooks(m, name)

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)

    return dtype_flops_mapping, op_flops_mapping
