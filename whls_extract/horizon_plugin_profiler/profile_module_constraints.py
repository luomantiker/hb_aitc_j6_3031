import copy
import math
import warnings
from typing import Any, Iterable

from horizon_plugin_profiler.utils.model_helper import _as_tuple
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch import nn as horizon_nn

# from torchvision import ops as vision_nn
from horizon_plugin_pytorch._torchvision_wrapper import ops as vision_nn
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import quantization_mappings


def _is_qat_mod(mod):
    return hasattr(mod, "qconfig") and (mod.config is not None)


def _warn_op_exist_by_march(marches=None, explanation=""):
    if marches is not None and not isinstance(marches, Iterable):
        marches = tuple(marches)

    def _func(mod, input, output):
        if _is_qat_mod(mod):
            return

        march = get_march()
        if marches is None or march in marches:
            op_name = str(mod.__class__)
            warnings.warn(
                "Operator %s detected and will harm the numerical accuracy"
                % op_name
                + " "
                + explanation
            )

    return _func


def _check_interpolate_inout_size(mod, input, output):
    if _is_qat_mod(mod):
        return

    def _check_quant_has_error(value, shift):
        int_value = value * (2 ** shift)
        return int_value - math.floor(int_value) > 0

    march = get_march()

    input_shape = input[0].shape
    output_shape = output.shape

    shift = 8 if march == March.BERNOULLI2 else 16

    if _check_quant_has_error(
        output_shape[2] / input_shape[2], shift
    ) or _check_quant_has_error(output_shape[3] / input_shape[3], shift):
        op_name = str(mod.__class__)
        warnings.warn(
            "Interpolation error will increase with output size"
            + " because the step is quantized %s" % op_name
        )


MODULE_CONSTRAINTS = {
    nn.AvgPool2d: lambda mod, input, output: warnings.warn(
        "Too large kernel size of nn.AvgPool2d"
        + " will harm the numerical accuracy"
    )
    if (_pair(mod.kernel_size)[0] * _pair(mod.kernel_size)[1] > 9)
    else None,
    nn.ReLU: _warn_op_exist_by_march(
        None, "Please consider replace it with ReLU6"
    ),
    nn.Sigmoid: _warn_op_exist_by_march(
        March.BERNOULLI2,
        "Because the output characteristic are not quantization friendly",
    ),
    nn.Softmax: _warn_op_exist_by_march(
        (
            March.BAYES,
            March.BAYES_E,
            March.NASH,
            March.NASH_E,
            March.NASH_M,
            March.NASH_P,
        ),
        "Because this op is implemented by look up reciprocal table",
    ),
    nn.SiLU: _warn_op_exist_by_march(
        (
            March.BAYES,
            March.BAYES_E,
            March.NASH,
            March.NASH_E,
            March.NASH_M,
            March.NASH_P,
        ),
        "Because this op is implemented by look up reciprocal table",
    ),
    horizon_nn.interpolate.Interpolate: _check_interpolate_inout_size,
    nn.Upsample: _check_interpolate_inout_size,
    nn.UpsamplingBilinear2d: _check_interpolate_inout_size,
    vision_nn.RoIAlign: _warn_op_exist_by_march(
        None, "Because the interpolate step is quantized"
    ),
}


class FunctionalWrapper:
    def __init__(self, mod):
        super(FunctionalWrapper, self).__init__()
        self.op_name = str(mod.__class__)
        self.mod = mod

    def div(self, x, y):
        warnings.warn(
            "Operator %s detected and will harm the numerical accuracy"
            % (self.op_name + ".div")
        )
        return self.mod.div(x, y)

    def cat(self, x, dim=0):
        if isinstance(x[0], QTensor):
            scales = torch.cat([data.q_scale().clone() for data in x])
            max_scale = scales.max()
            min_scale = scales.min()
            if max_scale / min_scale > 2:
                warnings.warn(
                    "Input scales of %s varies too much"
                    % (self.op_name + ".cat")
                    + " and will harm the numerical accuracy"
                )
        return self.mod.cat(x, dim)

    def __getattr__(self, name):
        return getattr(self.mod, name)


@typechecked
def profile_module_constraints(model: torch.nn.Module, example_inputs: Any):
    """Profile module contraints."""

    # init module constraints
    qat_mapping = quantization_mappings.get_qat_module_mappings()
    qat_constraints = {}
    for k, v in MODULE_CONSTRAINTS.items():
        if k in qat_mapping:
            qat_constraints[qat_mapping[k]] = v
    MODULE_CONSTRAINTS.update(qat_constraints)

    # register forward hook
    model = copy.deepcopy(model)
    functional_modules = {}
    for name, m in model.named_modules():
        if type(m) in (
            nn.quantized.FloatFunctional,
            horizon_nn.quantized.FloatFunctional,
        ):
            functional_modules[name] = FunctionalWrapper(m)
        if type(m) in MODULE_CONSTRAINTS:
            m.register_forward_hook(MODULE_CONSTRAINTS[type(m)])

    for k, v in functional_modules.items():
        model._modules[k] = v

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)
