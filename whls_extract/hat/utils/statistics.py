# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import warnings
from distutils.version import LooseVersion
from typing import Any, Dict, Tuple, Union

import horizon_plugin_pytorch as horizon
import torch
import torch.fx
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Target

__all__ = ["cal_ops"]


global module_total_params
module_total_params = 0
global module_total_ops
module_total_ops = 0


def count_parameters(m):
    global module_total_params
    for p in m.parameters():
        module_total_params += p.numel()
    for p in m.buffers():
        module_total_params += p.numel()


def count_linear(m, x, y):
    total_mul = m.in_features
    num_elements = (
        y.numel() if isinstance(y, torch.Tensor) else y.float.numel()
    )
    total_ops = total_mul * num_elements
    global module_total_ops
    module_total_ops += int(total_ops)


def count_convNd(m, x, y):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    nelement = (
        y.nelement() if isinstance(y, torch.Tensor) else y.float.nelement()
    )
    # index 1 represent channel
    in_channels = (
        x[0].shape[1]
        if isinstance(x[0], torch.Tensor)
        else x[0].float.shape[1]
    )
    try:
        groups = m._conv_kwargs["groups"]
    except Exception:
        # for nn.Conv2d, qat.Conv2d
        groups = m.groups

    total_ops = nelement * (in_channels // groups * kernel_ops)
    global module_total_ops
    module_total_ops += int(total_ops)


def count_convtranspose2d(m, x, y):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh

    # N x Cout x Hin x Win x Cin x Kw x Kh

    nelement = (
        x[0].nelement()
        if isinstance(x[0], torch.Tensor)
        else x[0].float.nelement()
    )
    # index 1 represent channel
    out_channels = (
        y.shape[1] if isinstance(y, torch.Tensor) else y.float.shape[1]
    )
    try:
        groups = m._conv_kwargs["groups"]
    except Exception:
        groups = m.groups

    total_ops = nelement * (out_channels // groups * kernel_ops)
    global module_total_ops
    module_total_ops += int(total_ops)


def count_matmul(output, args, kwargs):
    if (len(args) > 2 and args[2] is True) or (
        "x_trans" in kwargs and kwargs["x_trans"] is True
    ):
        total_mul = (
            args[0].shape[-2]
            if isinstance(args[0], torch.Tensor)
            else args[0].float.shape[-2]
        )
    else:
        total_mul = (
            args[0].shape[-1]
            if isinstance(args[0], torch.Tensor)
            else args[0].float.shape[-1]
        )
    num_elements = (
        output.numel()
        if isinstance(output, torch.Tensor)
        else output.float.numel()
    )
    total_ops = total_mul * num_elements
    global module_total_ops
    module_total_ops += int(total_ops)


def count_floatfunctional(m, x, y):
    print(m._last_called_method_name)
    if m._last_called_method_name == "matmul":
        # matmul cannot set `x_trans=True` when calculating
        # ops, otherwise the ops calculated will not be precise.
        count_matmul(y, x, {})


ops_count_mappings = {
    nn.Conv2d: count_convNd,
    nn.Linear: count_linear,
    nn.ConvTranspose2d: count_convtranspose2d,
    nn.Conv3d: count_convNd,
    horizon.nn.quantized.FloatFunctional: count_floatfunctional,
    # use method name to map func when call fx
    "matmul": count_matmul,
}


op_collect = {}


def add_op_count_hooks(m: nn.Module):
    if m in op_collect:
        return

    m_type = type(m)
    if m_type in ops_count_mappings:
        fn = ops_count_mappings[m_type]
        op_collect[m] = m.register_forward_hook(fn)


class CalopsInterpreter(torch.fx.Interpreter):
    """Calculate ops through torch.fx.Interpreter.

    Execute an FX graph Node-by-Node and
    record the parameters and ops of the node execution.

    Args:
         module: The module to be executed.
    """

    def __init__(self, module: GraphModule):
        super(CalopsInterpreter, self).__init__(module)

    def call_method(
        self,
        target: "Target",
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        result = super().call_method(target, args, kwargs)

        if target in ops_count_mappings:
            self_obj, *args_tail = args
            ops_count_func = ops_count_mappings[target]
            ops_count_func(result, args_tail, kwargs)

        return result

    def call_module(
        self,
        target: "Target",
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        result = super().call_module(target, args, kwargs)

        submod = self.fetch_attr(target)
        if type(submod) in ops_count_mappings:
            ops_count_func = ops_count_mappings[type(submod)]
            ops_count_func(submod, args, result)

        return result


def horizon_symbolic_trace(module: nn.Module):
    get_qat_module_mappings = (
        horizon.quantization.quantization_mappings.get_qat_module_mappings
    )
    tracer = horizon.quantization.quantize_fx.QuantizationTracer(
        [],
        list(get_qat_module_mappings().keys()),
    )
    graph = tracer.trace(module)
    graph_module = horizon.quantization.fx.graph_module.GraphModuleWithAttr(
        module, graph
    )
    return graph_module


def cal_ops(
    model: nn.Module, inputs: Union[torch.Tensor, dict], method: str = None
):
    """Calculate total ops and parameters of model.

    Use method `fx` or `hook` to record the ops
    during the model forward.

    Args:
        model: The module to be calculated.
        inputs: The inputs to the model.
        method: The method used by cal_ops, can be `fx` or `hook`.
    """
    prof_model = copy.deepcopy(model)
    prof_model.eval()
    global module_total_params
    module_total_params = 0
    count_parameters(prof_model)
    global module_total_ops
    module_total_ops = 0
    if method == "fx":
        if LooseVersion(horizon.__version__) < LooseVersion("1.0.0"):
            raise ValueError(
                "cal ops method 'fx' requires horizon_plugin_pytorch>=1.0.0"
            )
        try:
            graph_model = horizon_symbolic_trace(prof_model)
            calops = CalopsInterpreter(graph_model)
            calops.run(inputs)
            return module_total_ops, module_total_params
        except Exception as e:
            warnings.warn("The model cannot use torch.fx! " + str(e))
            raise e
    else:
        prof_model.apply(add_op_count_hooks)
        with torch.no_grad():
            prof_model(inputs)
    return module_total_ops, module_total_params
