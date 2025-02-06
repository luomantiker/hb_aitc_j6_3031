import copy
from typing import Any, Tuple, Type

from horizon_plugin_profiler.utils.deprecate import deprecated_warning
from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    apply_to_collection,
    attach_qualified_name,
    register_hook_on_leaf,
    swap_ff_with_horizonff,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from torch import Tensor

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.nn.qat import FloatFunctional as QATFloatFunctional
from horizon_plugin_pytorch.nn.quantized import FloatFunctional, QFunctional
from horizon_plugin_pytorch.qtensor import QTensor


@typechecked
def get_raw_features(
    model: torch.nn.Module,
    example_inputs: Any,
    prefixes: Tuple[str, ...] = (),
    types: Tuple[Type[torch.nn.Module], ...] = (),
    device: torch.device = None,
    preserve_int: bool = False,
    use_class_name: bool = False,
    skip_identity: bool = False,
):
    """Get raw features.

    Use hooks to get raw features to be profiled. Default insert hooks in all
    leaf modules. If the origin model is too large to show info in tensorboard,
    use prefixes or types to insert hooks in specific modules.

    Args:
        model: can be float/fused/calibration/qat/quantized model
        example_inputs: the input data feed to model
        prefixes: get features info by the prefix of qualified name
            Default: tuple().
        types: get features info by module type. Default: tuple().
        device: model run on which device. Default: None
        preserve_int(Deprecated): if True, record each op result in int type.
            Default: False
        use_class_name: if True, record class name not class type.
            Default: False
        skip_identity: if True, will not record the result of Identity module.
            Default: False

    Returns:
        output(List(dict)): A list of dict. Each dict contains:
            "module_name": (str) the module name in the model
            "module_type": (str) the module type
            "attr": (str) the attr of module. Maybe input/output/weight/bias
                Multi-inputs will be suffixed by input-i(i>=0)
            "data": (Tensor|QTensor) the featuremap
            "scale": (Tensor, None) the scale of the feature if it has.
            "ch_axis": (int) channel axis of per channel quantized data.
            "ff_method": (str) actual func name if module_type is
                FloatFunctional or QFunctional else None
    """
    if preserve_int:
        deprecated_warning(
            "horizon_plugin_pytorch",
            "1.7.1",
            "1.9.0",
            "preserve_int",
            None,
            "get_raw_features",
        )

    model = copy.deepcopy(model)
    swap_ff_with_horizonff(model)
    attach_qualified_name(model, True)
    if device is not None:
        model = model.to(device)
        example_inputs = apply_to_collection(
            example_inputs, Tensor, lambda x: x.to(device)
        )
        print("model will be run on {}.".format(device))
    model.eval()

    result = []

    def _record_data(data, mod, attr):
        if skip_identity and type(mod) in (
            torch.nn.Identity,
            horizon_nn.Identity,
        ):
            return
        ff_method = (
            f"[{mod._last_called_method_name}]"
            if type(mod) in (FloatFunctional, QATFloatFunctional, QFunctional)
            else ""
        )
        call_times = (
            f"({str(mod._shared_times)})" if mod._shared_times > 0 else ""
        )
        module_name = mod._qualified_name + ff_method + call_times
        if isinstance(data, QTensor):
            result.append(
                {
                    "module_name": module_name,
                    "module_type": mod.__class__.__name__
                    if use_class_name
                    else type(mod),
                    "attr": attr,
                    "data": data,
                    "scale": data.q_scale(),
                    "ch_axis": data.q_per_channel_axis(),
                    "dtype": data.dtype,
                }
            )

        elif isinstance(data, (tuple, list)):
            if len(data) == 1:
                _record_data(data[0], mod, f"{attr}")
            else:
                for i, d in enumerate(data):
                    _record_data(d, mod, f"{attr}-{i}")
        elif isinstance(data, Tensor):
            result.append(
                {
                    "module_name": module_name,
                    "module_type": mod.__class__.__name__
                    if use_class_name
                    else type(mod),
                    "attr": attr,
                    "data": data,
                    "scale": None,
                    "ch_axis": 0 if attr == "weight" else -1,
                    "dtype": data.dtype,
                }
            )

    def _pre_hook(module, input):
        _record_data(input, module, "input")
        for name, param in module.named_parameters():
            if "." not in name:
                _record_data(param, module, name)
        for name, buf in module.named_buffers():
            # skip scale and zero_point
            if all([x not in name for x in (".", "scale", "zero_point")]):
                if (
                    "." not in name
                    and "scale" not in name
                    and "zero_point" not in name
                ):
                    _record_data(buf, module, name)

    def _hook(module, input, output):
        _record_data(output, module, "output")
        module._shared_times += 1

    register_hook_on_leaf(
        model,
        forward_hook=_hook,
        forward_pre_hook=_pre_hook,
        to_reg_prefixes=prefixes,
        to_reg_types=types,
    )

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)
    del model

    return result
