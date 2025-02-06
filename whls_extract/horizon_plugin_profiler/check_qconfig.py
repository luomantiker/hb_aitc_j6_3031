import copy
import os
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from horizon_plugin_profiler.utils.logger import format_msg
from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    attach_qualified_name,
    register_hook_on_leaf,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from tabulate import tabulate
from torch import Tensor

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.nn.qat import FloatFunctional as QATFloatFunctional
from horizon_plugin_pytorch.nn.qat.stubs import DeQuantStub, QuantStub
from horizon_plugin_pytorch.nn.quantized import FloatFunctional, QFunctional
from horizon_plugin_pytorch.quantization.observer import FixedScaleObserver
from horizon_plugin_pytorch.quantization.observer_v2 import (
    FixedScaleObserver as FixedScaleObserverv2,
)


@typechecked
def check_qconfig(
    model: torch.nn.Module,
    example_inputs: Any,
    prefixes: Tuple[str, ...] = (),
    types: Tuple[Type[torch.nn.Module], ...] = (),
    custom_check_func: Optional[Callable] = None,
    out_dir: Optional[str] = None,
) -> Tuple[List, List, List]:
    """Check quantization configuration of calibration/qat model.

    This function
    1) checks activation and weight quantization configurations of each layer
        in the model. These infos will be saved in "qconfig_info.txt".
    2) checks input and output dtype of each layer in the model.

    Defaultly this function prints warnings when checking:
    1) activation = None
    2) fixed scale observer
    3) not qint8 weight
    4) module inputs and outputs dtype diff
    If you want to check more info, define a custom check function and use
    `custom_check_func` parameter.

    Args:
        model: MUST be calibration or qat model
        example_inputs (Any[Tensor]): The input data feed to model.
        prefixes: get features info by the prefix of qualified name
            Default: tuple().
        types: get features info by module type. Default: tuple().
        custom_check_func: a user-defined function to check other infos. This
            function is invoked in module hooks, so it has the same signature
            with torch.nn.Module hooks:
                func(module, input, output) -> None
        out_dir: path to save the result txt 'qconfig_info.txt'. If None, will
            save in the current directory. Default: None

    Returns:
        (out_info_list, weight_info_list, warning_layers_info)
    """
    model = copy.deepcopy(model).eval()
    attach_qualified_name(model, True)

    out_info_map = []
    weight_info_map = []
    unusual_map = []
    out_dtype_statistics = {}

    def _get_qconfig_info(m, name, input_dtypes, output_dtypes):
        out_info = None
        weight_info = None
        input_dtype = (
            input_dtypes[0] if len(input_dtypes) == 1 else input_dtypes
        )
        output_dtype = (
            output_dtypes[0] if len(output_dtypes) == 1 else output_dtypes
        )
        if hasattr(m, "activation_post_process"):
            out_quant = m.activation_post_process
            if out_quant is None:
                msg = "activation is None. Maybe output layer?"
                unusual_map.append([name, type(m), msg])
                out_info = [
                    name,
                    type(m),
                    input_dtype,
                    output_dtype,
                    "activation = None",
                ]
            elif isinstance(out_quant, torch.quantization.FakeQuantizeBase):
                # maybe calibration or qat model
                out_info = [
                    name,
                    type(m),
                    input_dtype,
                    output_dtype,
                    out_quant.ch_axis,
                ]
                if (
                    isinstance(
                        out_quant.activation_post_process,
                        (FixedScaleObserver, FixedScaleObserverv2),
                    )
                    or not out_quant._observer_enabled
                ):
                    msg = "Fixed scale {}".format(out_quant.scale.item())
                    unusual_map.append([name, type(m), msg])
                if not all([out_quant.dtype == od for od in output_dtypes]):
                    msg = "actual out dtype != configured out dtype"
                    unusual_map.append([name, type(m), msg])
        elif not isinstance(m, (torch.nn.Identity, horizon_nn.Identity)):
            out_info = [
                name,
                type(m),
                input_dtype,
                output_dtype,
                "qconfig = None",
            ]

        if out_info is not None:
            out_info_map.append(out_info)
            out_dtype_statistics[output_dtype] = (
                out_dtype_statistics.get(output_dtype, 0) + 1
            )

        if hasattr(m, "weight_fake_quant") and m.weight_fake_quant is not None:
            weight_quant = m.weight_fake_quant
            weight_info = [
                name,
                type(m),
                weight_quant.dtype,
                weight_quant.ch_axis,
            ]
            if weight_quant.dtype != "qint8":
                msg = "{} weight!!!".format(weight_quant.dtype)
                unusual_map.append([name, type(m), msg])
            if weight_info is not None:
                weight_info_map.append(weight_info)

    # custom get_input_dtypes to skip no tensor input
    def _get_dtypes(input):
        ret = []
        if isinstance(input, Tensor):
            return [input.dtype]

        if isinstance(input, Mapping):
            for _, v in input.items():
                ret += _get_dtypes(v)
            return ret

        if isinstance(input, Sequence) and not isinstance(input, str):
            for x in input:
                ret += _get_dtypes(x)
            return ret

        return ret

    def _hook(module, input, output):
        input_dtypes = _get_dtypes(input)
        out_dtypes = _get_dtypes(output)
        ff_method = (
            f"[{module._last_called_method_name}]"
            if type(module)
            in (FloatFunctional, QATFloatFunctional, QFunctional)
            else ""
        )
        name = (
            module._qualified_name
            + ff_method
            + (
                ""
                if module._shared_times == 0
                else f"({module._shared_times})"
            )
        )
        # skip quant and dequant input output dtype check
        if not isinstance(module, (QuantStub, DeQuantStub)):
            if (
                len(out_dtypes) == 1
                and out_dtypes[0] not in (torch.float32, "float32")
                and not all([x == out_dtypes[0] for x in input_dtypes])
            ):
                msg = "input dtype {} is not same with out dtype {}".format(
                    input_dtypes, out_dtypes[0]
                )
                unusual_map.append([name, type(module), msg])

        # get each layer qconfig info
        _get_qconfig_info(module, name, input_dtypes, out_dtypes)

        if custom_check_func is not None:
            custom_check_func(module, input, output)
        module._shared_times += 1

    # check input and output dtypes of each layer
    register_hook_on_leaf(
        model, _hook, to_reg_prefixes=prefixes, to_reg_types=types
    )
    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)
    del model

    out_dir = "." if out_dir is None else out_dir
    file_path = os.path.join(out_dir, "qconfig_info.txt")
    with open(file_path, "w") as f:
        f.write(f"Out dtype statistics: {str(out_dtype_statistics)}\n")
        f.write("Each layer out qconfig:\n")
        f.write(
            tabulate(
                out_info_map,
                headers=(
                    "Module Name",
                    "Module Type",
                    "Input dtype",
                    "out dtype",
                    "ch_axis",
                ),
                tablefmt="psql",
            )
        )
        if weight_info_map:
            f.write("\n\nWeight qconfig:\n")
            f.write(
                tabulate(
                    weight_info_map,
                    headers=(
                        "Module Name",
                        "Module Type",
                        "weight dtype",
                        "ch_axis",
                        "len(scale)",
                    ),
                    tablefmt="psql",
                )
            )
        if unusual_map:
            f.write("\n\nPlease check if these OPs qconfigs are expected..\n")
            f.write(
                tabulate(
                    unusual_map,
                    headers=("Module Name", "Module Type", "Msg"),
                    tablefmt="psql",
                )
            )

    # print warnings on the screen
    if unusual_map:
        print(
            format_msg(
                "\nPlease check if these OPs qconfigs are expected..", "red"
            )
        )
        print(
            tabulate(
                unusual_map,
                headers=("Module Name", "Module Type", "Msg"),
                tablefmt="psql",
            )
        )

    return out_info_map, weight_info_map, unusual_map
