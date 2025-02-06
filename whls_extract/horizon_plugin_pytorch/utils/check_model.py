import copy
import io
import logging
import os
from contextlib import redirect_stdout
from typing import Any, Mapping, Optional, Sequence

import torch
from tabulate import tabulate
from torch import Tensor

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.nn.qat import DeQuantStub as QATDeQuantStub
from horizon_plugin_pytorch.nn.qat import FloatFunctional as QATFloatFunctional
from horizon_plugin_pytorch.nn.qat import QuantStub
from horizon_plugin_pytorch.nn.quantized import FloatFunctional, QFunctional
from horizon_plugin_pytorch.quantization.fake_cast import FakeCast
from horizon_plugin_pytorch.quantization.fake_quantize import (
    FakeQuantState,
    set_fake_quantize,
)
from horizon_plugin_pytorch.quantization.fuse_modules import (
    get_op_list_to_fuser_mapping,
)
from horizon_plugin_pytorch.quantization.observer import FixedScaleObserver
from horizon_plugin_pytorch.quantization.observer_v2 import (
    FixedScaleObserver as FixedScaleObserverv2,
)
from horizon_plugin_pytorch.quantization.quantization_mappings import (
    get_qat_module_mappings,
)
from horizon_plugin_pytorch.utils.model_helper import (
    _as_tuple,
    attach_qualified_name,
    register_hook_on_leaf,
)
from horizon_plugin_pytorch.utils.quant_switch import GlobalFakeQuantSwitch
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)

__all__ = ["check_qat_model"]


@typechecked
def check_qat_model(
    model: torch.nn.Module,
    example_inputs: Any = None,
    example_kw_inputs: Any = None,
    save_results: bool = False,
    out_dir: Optional[str] = None,
):
    """Check calibration/qat model structure.

    This function supports calibration/qat model checker. It checks:
    1. if model has shared ops
    2. if model has unfused operations
    3. model quantization config

    Args:
        model: model to check
        example_inputs: model inputs
        example_kw_inputs: model keyword inputs
        save_results: whether to save results to txt. Default: False
        out_dir: path to save the result txt 'model_check_result.txt'. If None,
            will save in the current directory. Default: None
    """
    logger.info("Begin check qat model...")
    model = copy.deepcopy(model)

    attach_qualified_name(model)

    # record model called time
    module_refcount = {}

    # qconfig map
    out_info_map = []
    weight_info_map = []
    unusual_map = []
    dtype_list = []
    dtype_statistics = {}

    # qconfig: record each layer qconfig info
    def _get_qconfig_info(m, name, input_dtype, output_dtype):
        out_info = None
        weight_info = None
        if isinstance(m, torch.quantization.FakeQuantizeBase):
            # support hybrid model check
            out_info = [
                name,
                type(m),
                input_dtype,
                output_dtype,
                m.ch_axis,
                m.activation_post_process.__class__.__name__,
            ]
        elif hasattr(m, "activation_post_process"):
            # special implement of qat GELU...
            if isinstance(m, horizon_nn.qat.GELU):
                out_quant = m.lut.activation_post_process
            else:
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
                    None,
                ]
            elif isinstance(out_quant, torch.quantization.FakeQuantizeBase):
                out_info = [
                    name,
                    type(m),
                    input_dtype,
                    output_dtype,
                    out_quant.ch_axis,
                    out_quant.activation_post_process.__class__.__name__,
                ]
                if isinstance(
                    out_quant.activation_post_process,
                    (FixedScaleObserver, FixedScaleObserverv2),
                ):
                    msg = f"Fixed scale {out_quant.scale.item()}"
                    unusual_map.append([name, type(m), msg])
                if isinstance(m, QuantStub) and m.scale is not None:
                    scale = (
                        m.scale.item()
                        if isinstance(m.scale, Tensor)
                        else m.scale
                    )
                    msg = f"Fixed input scale {scale}"
                    unusual_map.append([name, type(m), msg])
            elif isinstance(out_quant, FakeCast):
                out_info = [
                    name,
                    type(m),
                    input_dtype,
                    output_dtype,
                    f"fakecast to {out_quant.dtype}",
                    None,
                ]
        elif not isinstance(m, (torch.nn.Identity, horizon_nn.Identity)):
            out_info = [
                name,
                type(m),
                input_dtype,
                output_dtype,
                "qconfig = None",
                None,
            ]

        # check if input output dtype same
        if (
            not isinstance(m, (QuantStub, QATDeQuantStub))
            and not all(
                [x in (qint8, qint16) for x in input_dtype + output_dtype]
            )
            and not all(
                [
                    x in (torch.float32, torch.float16)
                    for x in input_dtype + output_dtype
                ]
            )
        ):
            msg = f"Mixed input dtype {input_dtype} and output dtype {output_dtype}"  # noqa E501
            unusual_map.append([name, type(m), msg])

        if out_info is not None:
            out_info_map.append(out_info)
            if type(m) not in dtype_statistics:
                dtype_statistics[type(m)] = {
                    "input": {},
                    "output": {},
                }
            for dtype in set(input_dtype):
                dtype_statistics[type(m)]["input"].setdefault(dtype, 0)
                dtype_statistics[type(m)]["input"][dtype] += 1
            for dtype in set(output_dtype):
                dtype_statistics[type(m)]["output"].setdefault(dtype, 0)
                dtype_statistics[type(m)]["output"][dtype] += 1

        if hasattr(m, "weight_fake_quant") and m.weight_fake_quant is not None:
            weight_quant = m.weight_fake_quant
            weight_info = [
                name,
                type(m),
                weight_quant.dtype,
                weight_quant.ch_axis
                if hasattr(weight_quant, "ch_axis")
                else None,
                weight_quant.activation_post_process.__class__.__name__
                if hasattr(weight_quant, "activation_post_process")
                else None,
            ]
            if weight_quant.dtype != "qint8":
                msg = f"{weight_quant.dtype} weight!"
                unusual_map.append([name, type(m), msg])
            if weight_info is not None:
                weight_info_map.append(weight_info)

    # qconfig: custom get_input_dtypes to skip no tensor input
    def _get_dtypes(input):
        ret = []
        if isinstance(input, Tensor):
            if input.dtype not in dtype_list:
                dtype_list.append(input.dtype)
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

    # check fused: prepare check unfused operations
    output_to_module_mapping = {}
    virtual_input_node = torch.nn.Identity()
    virtual_input_node._qualified_name = "placeholder"
    virtual_input_node._output_to = []

    def _add_virtual_input_node(inputs):
        if isinstance(inputs, Tensor):
            output_to_module_mapping[id(inputs)] = virtual_input_node
        # in HAT config, dict is ConfigDict, a subclass of dict
        elif issubclass(type(inputs), dict):
            for value in list(inputs.values()):
                _add_virtual_input_node(value)
        elif type(inputs) in (list, tuple):
            for i in inputs:
                _add_virtual_input_node(i)

    _add_virtual_input_node(example_inputs)
    _add_virtual_input_node(example_kw_inputs)

    # check unfused: record output to module mapping
    def _record_output(module, input, output):
        if isinstance(output, (list, tuple)):
            for x in output:
                _record_output(module, input, x)
        elif isinstance(output, Tensor):
            output_to_module_mapping[id(output)] = module

    def _hook(module, input, output):
        # qconfig: record input/output dtypes and each layer info
        input_dtypes = _get_dtypes(input)
        out_dtypes = _get_dtypes(output)
        ff_method = (
            f"[{module._last_called_method_name}]"
            if type(module)
            in (FloatFunctional, QATFloatFunctional, QFunctional)
            else ""
        )
        shared_times = module_refcount[module._qualified_name]
        name = (
            module._qualified_name
            + ff_method
            + ("" if shared_times == 0 else f"({shared_times})")
        )
        _get_qconfig_info(module, name, input_dtypes, out_dtypes)

        # check shared: record module called times
        module_refcount[module._qualified_name] += 1

        # check unfused: record output to module mapping
        _record_output(module, input, output)

    # check unfused: make graph
    def _pre_hook(module, input):
        if not hasattr(module, "_input_from"):
            module._input_from = []
        if not hasattr(module, "_output_to"):
            module._output_to = []

        if isinstance(input, (list, tuple)):
            for x in input:
                _pre_hook(module, x)
        elif isinstance(input, Tensor):
            # if input from functions like reshape
            # use virtual_input_node as input
            input_from = (
                virtual_input_node
                if id(input) not in output_to_module_mapping
                else output_to_module_mapping[id(input)]
            )
            if input_from not in module._input_from:
                module._input_from.append(input_from)
            if module not in input_from._output_to:
                input_from._output_to.append(module)

    handle_dict = register_hook_on_leaf(model, _hook, _pre_hook)
    for name in handle_dict:
        module_refcount[name] = 0

    origin_state = GlobalFakeQuantSwitch.state()
    set_fake_quantize(model, FakeQuantState._FLOAT)
    if example_inputs is None:
        example_inputs = ()
    if example_kw_inputs is None:
        example_kw_inputs = {}
    example_inputs = _as_tuple(example_inputs)
    with torch.no_grad():
        model(*example_inputs, **example_kw_inputs)
    if origin_state:
        GlobalFakeQuantSwitch.enable()

    def match_node(module, expected_type):
        if hasattr(module, "_matched"):
            return False
        if not type(module) == expected_type:
            return False
        if expected_type is QATFloatFunctional:
            return module._last_called_method_name == "add"
        return True

    def match_pattern(root: torch.nn.Module, pattern):
        if match_node(root, pattern[0]):
            if len(pattern) == 1:
                root._matched = True
                return [[root]]
            else:
                tile = []
                for next_node in root._output_to:
                    tile += match_pattern(next_node, pattern[1:])
                for matched_seq in tile:
                    root._matched = True
                    matched_seq.insert(0, root)
                return tile
        else:
            return []

    def get_unmatched_next(root: torch.nn.Module, visited_nodes):
        if root in visited_nodes:
            return set()
        next_nodes = set()
        visited_nodes.add(root)
        for node in root._output_to:
            if hasattr(node, "_matched"):
                next_nodes |= get_unmatched_next(node, visited_nodes)
            else:
                if node not in visited_nodes:
                    next_nodes.add(node)

        return next_nodes

    def search_pattern(root: torch.nn.Module, patterns):
        current_roots = [root]
        ret = []

        visited_nodes = set()
        while len(current_roots) > 0:
            next_roots = set()
            for node in current_roots:
                if not hasattr(node, "_matched"):
                    matched_seqs = []
                    for pattern in patterns:
                        matched_seqs = match_pattern(node, pattern)
                        if len(matched_seqs) > 0:
                            ret += matched_seqs
                            break

                next_roots |= get_unmatched_next(node, visited_nodes)
            if current_roots == next_roots:
                logger.warning(
                    "There are circles in the graph, "
                    "related nodes are {}".format(
                        [m._qualified_name for m in current_roots]
                    ),
                    UserWarning,
                )
                break
            current_roots = next_roots

        return ret

    float2qat_map = get_qat_module_mappings()
    fuse_patterns = list(get_op_list_to_fuser_mapping().keys())
    fuse_patterns.sort(key=len, reverse=True)
    # no qat SyncBatchNorm and relu6
    fuse_patterns = [
        [float2qat_map.get(x, x) for x in patterns]
        for patterns in fuse_patterns
    ]

    matched_seqs = search_pattern(virtual_input_node, fuse_patterns)
    module_to_fuse = []
    for matched_seq in matched_seqs:
        valid = True
        shared_conv = False
        if len(matched_seq[0]._output_to) > 1:
            shared_conv = True
        for m in matched_seq[1:]:
            if isinstance(m, QATFloatFunctional):
                if len(m._input_from) > 2 or len(m._output_to) > 1:
                    valid = False
                    break
            else:
                if len(m._input_from) > 1 or len(m._output_to) > 1:
                    valid = False
                    break

        if valid:
            module_to_fuse.append(
                [
                    (
                        module._qualified_name
                        + ("(shared)" if i == 0 and shared_conv else ""),
                        type(module),
                    )
                    for i, module in enumerate(matched_seq)
                ]
            )

    # check unfused result
    if len(module_to_fuse) == 0:
        logger.info("All fusable modules are fused in model!")
    else:
        logger.warning("Fusable modules are listed below:")
        for item in module_to_fuse:
            logger.warning(
                "\n"
                + tabulate(item + [["", ""]], headers=["name", "type"])
                + "\n"
            )

    # Check if high precision output not enabled
    warned_output_mod = []
    hp_out_name = []
    for _, m in model.named_modules():
        # Use DeQuantStub to identify output mod.
        if isinstance(m, QATDeQuantStub):
            dequant_name = m._qualified_name
            while hasattr(m, "_input_from") and isinstance(
                m._input_from, torch.nn.Identity
            ):
                m = m._input_from
            if hasattr(m, "_input_from"):
                for output_mod in m._input_from:
                    # Ensure the output of output_mod is not used by other op.
                    if (
                        hasattr(output_mod, "_output_to")
                        and all(
                            (
                                isinstance(next_m, QATDeQuantStub)
                                for next_m in output_mod._output_to
                            )
                        )
                        and (
                            "Conv" in output_mod.__class__.__name__
                            or "Linear" in output_mod.__class__.__name__
                        )
                    ):
                        if not hasattr(output_mod, "qconfig"):
                            logger.warning(
                                f"{dequant_name} input module "
                                f"{output_mod._qualified_name} has no "
                                "qconfig. Please check if it is expected."
                            )
                            continue
                        if output_mod.qconfig.activation is not None:
                            warned_output_mod.append(output_mod)
                        else:
                            hp_out_name.append(output_mod._qualified_name)
    # Get mod names.
    if warned_output_mod:
        for n, m in model.named_modules():
            if m in warned_output_mod:
                logger.warning(
                    "Output module '{}' support high precision output, but is"
                    " not configured to enable it.".format(n)
                )

    # remove high precision output in unusual_map
    if len(hp_out_name) > 0 and len(unusual_map) > 0:
        unusual_map = [
            x
            for x in unusual_map
            if x[0].split("(")[0].split("[")[0] not in hp_out_name
            and "Mixed" in x[-1]
        ]

    # check shared result
    unused_module = {
        k: v
        for k, v in module_refcount.items()
        if v == 0
        and not isinstance(
            model.get_submodule(k), (torch.nn.Identity, horizon_nn.Identity)
        )
    }
    # dequant can be used multi-times
    shared_module = {
        k: v
        for k, v in module_refcount.items()
        if v > 1
        and not isinstance(
            model.get_submodule(k),
            (torch.nn.Identity, horizon_nn.Identity, QATDeQuantStub),
        )
    }
    if len(unused_module) == 0 and len(shared_module) == 0:
        logger.info("All modules in the model run exactly once.")
    else:
        if len(unused_module) > 0:
            logger.warning(
                "Modules below are not used:\n"
                + tabulate(
                    unused_module.items(), headers=["name", "called times"]
                )
                + "\n"
            )
        if len(shared_module) > 0:
            logger.warning(
                "Modules below are used multi times:\n"
                + tabulate(
                    shared_module.items(), headers=["name", "called times"]
                )
                + "\n"
            )

    # qconfig result
    if unusual_map:
        logger.warning(
            "Please check these modules qconfig if expected:\n"
            + tabulate(
                unusual_map,
                headers=("module name", "module type", "msg"),
                tablefmt="psql",
            )
            + "\n"
        )
    else:
        logger.info("No obvious abnormal qconfig in the model.")

    if save_results:
        out_dir = "." if out_dir is None else out_dir
        file_path = os.path.join(out_dir, "model_check_result.txt")
        with open(file_path, "w") as f:
            if len(module_to_fuse) == 0:
                f.write("All fusable modules are fused in model!\n")
            else:
                f.write("Fusable modules are listed below:\n")
                for item in module_to_fuse:
                    f.write(
                        tabulate(item + [["", ""]], headers=["name", "type"])
                        + "\n"
                    )

            if len(unused_module) == 0 and len(shared_module) == 0:
                f.write("All modules in the model run exactly once.\n")
            else:
                if len(unused_module) > 0:
                    f.write(
                        "\n\nModules below are not used:\n"
                        + tabulate(
                            unused_module.items(),
                            headers=["name", "called times"],
                        )
                    )
                if len(shared_module) > 0:
                    f.write(
                        "\n\nModules below are used multi times:\n"
                        + tabulate(
                            shared_module.items(),
                            headers=["name", "called times"],
                        )
                    )

            input_dtype_statistics = []
            output_dtype_statistics = []
            total_input = [0] * len(dtype_list)
            total_output = [0] * len(dtype_list)
            for mod_type, statistics in dtype_statistics.items():
                input_dtype_statistics.append(
                    [mod_type]
                    + [
                        statistics["input"].get(dtype, 0)
                        for dtype in dtype_list
                    ]
                )
                for i, v in enumerate(input_dtype_statistics[-1][1:]):
                    total_input[i] += v
                output_dtype_statistics.append(
                    [mod_type]
                    + [
                        statistics["output"].get(dtype, 0)
                        for dtype in dtype_list
                    ]
                )
                for i, v in enumerate(output_dtype_statistics[-1][1:]):
                    total_output[i] += v
            input_dtype_statistics.append(["total"] + total_input)
            output_dtype_statistics.append(["total"] + total_output)

            f.write("\n\ninput dtype statistics:\n")
            f.write(
                tabulate(
                    input_dtype_statistics,
                    headers=["module type"] + dtype_list,
                    tablefmt="psql",
                )
            )

            f.write("\n\noutput dtype statistics:\n")
            f.write(
                tabulate(
                    output_dtype_statistics,
                    headers=["module type"] + dtype_list,
                    tablefmt="psql",
                )
            )

            f.write("\n\nEach layer out qconfig:\n")
            f.write(
                tabulate(
                    out_info_map,
                    headers=(
                        "Module Name",
                        "Module Type",
                        "Input dtype",
                        "out dtype",
                        "ch_axis",
                        "observer",
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
                            "observer",
                        ),
                        tablefmt="psql",
                    )
                )
            if unusual_map:
                f.write(
                    "\n\nPlease check if these OPs qconfigs are expected..\n"
                )
                f.write(
                    tabulate(
                        unusual_map,
                        headers=("Module Name", "Module Type", "Msg"),
                        tablefmt="psql",
                    )
                )
            if hasattr(model, "graph"):
                with io.StringIO() as buf, redirect_stdout(buf):
                    model.graph.print_tabular()
                    graph_output = buf.getvalue()
                f.write(f"\n\nGraph:\n{graph_output}\n")

        logger.info(f"Check full result in {file_path}")

    else:
        logger.info("To check full result info, set `save_results=True`.")

    for handles in handle_dict.values():
        for handle in handles:
            if handle is not None:
                handle.remove()

    logger.info("End check")
