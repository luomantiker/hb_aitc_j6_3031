import os
import tempfile
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Optional, Sequence, Union

from horizon_plugin_profiler.utils.logger import format_msg
from horizon_plugin_profiler.utils.model_helper import _as_tuple
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from tabulate import tabulate

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.march import get_march


def _compare_with_hbdk_parser(plugin_results, hbdk_results):
    table = []
    unequal_table = []
    if len(plugin_results) != len(hbdk_results):
        keys = set(plugin_results.keys()) | set(hbdk_results.keys())
        print("not in ret: ", keys - set(plugin_results.keys()))
        print("not in hbdk: ", keys - set(hbdk_results))
        raise RuntimeError(
            f"{len(plugin_results)} pt ops != {len(hbdk_results)} "
            "hbdk parser ops"
        )
    for k in hbdk_results.keys():
        hbdk_ret = hbdk_results[k][1]
        hbdk_ret = (
            hbdk_ret if "torch_native" in k else hbdk_ret.permute(0, 3, 1, 2)
        )
        is_equal = torch.equal(plugin_results[k], hbdk_ret)
        if not is_equal:
            print(k)
            unequal_table.append(
                [k, torch.max(torch.abs(plugin_results[k] - hbdk_ret))]
            )
        table.append([k, torch.equal(plugin_results[k], hbdk_ret)])
    print(tabulate(table, headers=["name", "if equal"]))
    if table[-1][-1]:
        print(
            format_msg(
                "Torch run pt output is same with hbdk parser.", "green"
            )
        )
    else:
        print(
            format_msg(
                "Torch run pt output is different with hbdk parser!", "red"
            )
        )

    if unequal_table:
        print(format_msg("The following ops results are different...", "red"))
        print(tabulate(unequal_table, headers=["name", "atol"]))


@typechecked
def script_profile(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    example_inputs: Any,
    out_dir: Optional[str] = None,
    march: Optional[str] = None,
    mark_node_func: Optional[Callable] = None,
    compare_with_hbdk_parser: bool = True,
):
    """Get each module result in scriptmodel to compare with hbdk parser.

    This function will save the results in `plugin_quantized_result.npy`
    with `allow_pickle = True`, which MUST also be set True when loading the
    file by numpy. This function also returns the results in dict(dict) format.

    Args:
        model: must be quantized model or ScriptModule
        example_inputs: the input data feed to model
        out_dir: path to save the each op results in scriptmodule. If None,
            will save in the current directory. Default: None
        march: march of bpu. If None, will use get_march(). Default: None.
        mark_node_func: the func to determine which nodes to save results in
            scriptmodule. If None, will use default mark_node_func
        compare_with_hbdk_parser: whether to compare scriptmodule results with
            hbdk parser. Default: True

    Returns:
        output(dict<str, tensor>): A dict recording each op results
            A dict with schema:

            * KEY (str): the processed layer name same with hbdk parser
            * VALUE (tensor): op values
    """
    if not isinstance(model, torch.jit.ScriptModule):
        model = torch.jit.trace(model, example_inputs)

    pt_file = NamedTemporaryFile(suffix=".pt", delete=True)
    torch.jit.save(model, pt_file.name)
    model = torch.jit.load(pt_file.name)
    if march is None:
        march = get_march()

    # run hbdk parser to get node info and results
    from ._hbdk_parser_helper import run_hbdk_parser

    hbdk_results, const_check_map = run_hbdk_parser(
        model, example_inputs, march
    )

    for name, mod in model.named_modules():
        mod._debug_name = name

    name_count = {}

    def _node2name(prefix, func_name, op_type):
        prefix = "_" + prefix if prefix != "" else prefix
        name = f"{prefix.replace('.', '_')}_{op_type}_{func_name}"
        # count op invoke times in submodule with prefix
        if name not in name_count:
            name_count[name] = 0
        else:
            name_count[name] += 1
            name = name + "_" + str(name_count[name])
        return name

    to_save_map = {}
    attr_map = {}

    def _default_mark_node(g, prefix=""):
        # mark qtensor operation node in graph
        for n in g.nodes():
            hbir_name = None
            func_type = None
            if n.kind() == "prim::CallFunction":
                func_name = n.inputsAt(0).node().s("name")
                is_const = [const_check_map[v] is None for v in n.outputs()]
                if hasattr(
                    horizon_nn.quantized.functional, func_name
                ) and not all(is_const):
                    hbir_name = [const_check_map[v] for v in n.outputs()]
                    func_type = "hz"
            elif n.kind() == "prim::CallMethod":
                if not all(const_check_map[v] is None for v in n.inputs()):
                    submodule = attr_map[n.inputsAt(0)]
                    method_graph = getattr(submodule, n.s("name")).graph
                    attr_map[list(method_graph.inputs())[0]] = submodule
                    _default_mark_node(method_graph, submodule._debug_name)
            elif n.kind().split("::")[0] == "aten":
                func_name = n.kind().split("::")[1]
                if func_name.endswith("_") and not func_name.endswith("__"):
                    func_name = func_name[:-1]
                is_const = [const_check_map[v] is None for v in n.outputs()]
                if func_name != "alias" and not all(is_const):
                    hbir_name = [const_check_map[v] for v in n.outputs()]
                    func_type = "aten"
            elif n.kind() == "prim::GetAttr":
                if n.inputsAt(0) in attr_map and isinstance(
                    attr_map[n.inputsAt(0)], torch.jit.ScriptModule
                ):
                    attr = getattr(attr_map[n.inputsAt(0)], n.s("name"))
                    if isinstance(attr, torch.jit.ScriptModule):
                        attr_map[n.outputsAt(0)] = attr

            if hbir_name is not None and func_type is not None:
                # node output maybe one tuple/list while hbir_name have two
                nm = _node2name(prefix, func_name, func_type)
                if n.outputsSize() == 1 and type(hbir_name) != str:
                    hbir_name = hbir_name[0]
                    to_save_map.update({n.outputsAt(0): (g, nm, hbir_name)})
                else:
                    assert len(hbir_name) == n.outputsSize()
                    for i, v in enumerate(hbir_name):
                        to_save_map.update(
                            {n.outputsAt(i): (g, nm + f"_{i}", v)}
                        )

    attr_map[list(model.graph.inputs())[0]] = model
    if mark_node_func is None:
        mark_node_func = _default_mark_node
    mark_node_func(model.graph)

    actual_value_pos = {}
    # insert node in scriptmodel
    results_path = ".horizon_script_results"

    files = []
    results_path = os.path.join(tempfile.gettempdir(), results_path)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    for value, info in to_save_map.items():
        g, name, record = info
        actual_value_pos[name] = record
        node = value.node()
        pt_name = os.path.join(results_path, name + ".pt")
        pt_name_node = g.insertConstant(pt_name).node()
        save_node = g.create("aten::save")
        save_node.addInput(value)
        save_node.addInput(pt_name_node.outputsAt(0))
        save_node.eraseOutput(0)
        pt_name_node.moveAfter(node)
        save_node.insertAfter(pt_name_node)
        files.append(name + ".pt")
    model.graph.lint()

    pt_file = NamedTemporaryFile(suffix=".pt", delete=True)
    torch.jit.save(model, pt_file.name)
    model = torch.jit.load(pt_file.name)

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)

    # record result and delete tmp files
    ret = {}

    def _apply_data(data, names):
        if names is None:
            pass
        elif isinstance(names, str) and isinstance(data, torch.Tensor):
            ret.update({names: data})  # [names] == data
        elif isinstance(names, Sequence):
            assert len(data) == len(names)
            for d, n in zip(data, names):
                _apply_data(d, n)
        else:
            raise RuntimeError("Unknown data dtype ", type(data))

    for file in files:
        index_name = file.split(".")[0]
        data = torch.load(os.path.join(results_path, file))
        if "hz_dequantize" not in index_name:
            _apply_data(data, actual_value_pos[index_name])
        os.remove(os.path.join(results_path, file))
    try:
        os.removedirs(results_path)
    except FileNotFoundError:
        pass

    if compare_with_hbdk_parser:
        _compare_with_hbdk_parser(ret, hbdk_results)

    out_dir = "." if out_dir is None else out_dir
    path = os.path.join(out_dir, "horizon_script_results.pt")
    torch.save(ret, path)
    return ret


@typechecked
def compare_script_models(
    model1: torch.jit.ScriptModule,
    model2: torch.jit.ScriptModule,
    example_inputs: Any,
    march: Optional[str] = None,
):
    """Compare two ScriptModules results.

    This function compares each op results in two scriptmodules traced from the
    same model in different versions of plugin.

    Args:
        model1: ScriptModule to be compared
        model2: ScriptModule to be compared
        example_inputs: the input data feed to model
        march: march of bpu. If None, will use get_march(). Default: None.
    """
    ret1 = script_profile(
        model1, example_inputs, march=march, compare_with_hbdk_parser=False
    )
    ret2 = script_profile(
        model2, example_inputs, march=march, compare_with_hbdk_parser=False
    )

    assert len(ret1) == len(ret2), (
        f"Two ScriptModules ops num are not same. {len(ret1)} != {len(ret2)} "
        "Please make sure two ScriptModules are traced from the same model."
    )

    table = []
    unequal_table = []

    def _compare(x, y):
        assert type(x) == type(y)
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, (list, tuple)):
            return all([_compare(xx, yy) for xx, yy in zip(x, y)])

    for k, v in ret1.items():
        is_equal = _compare(v, ret2[k])
        table.append([k, is_equal])
        if not is_equal:
            unequal_table.append([k, is_equal])

    print(tabulate(table, headers=["name", "if equal"]))
    if unequal_table:
        print(format_msg("The following ops results are different...", "red"))
        print(tabulate(unequal_table, headers=["name", "if equal"]))
    else:
        print(format_msg("All ops in two ScriptModules are same.", "green"))
