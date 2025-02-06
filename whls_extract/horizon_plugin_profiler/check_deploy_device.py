import os
from typing import Dict, Optional, Tuple

from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from tabulate import tabulate

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.quantization.hybrid_ops import (
    get_hybrid_qat_module_mappings,
    get_hybrid_quantized_module_mappings,
    get_hybrid_supported_functions,
    get_hybrid_supported_methods,
)


@typechecked
def check_deploy_device(
    model: torch.fx.GraphModule,
    print_tabulate: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, Tuple[str, str]]:
    """Check deploy device(BPU or CPU) of hybrid model.

    Args:
        model: qat or quantized model. MUST be converted by prepare_qat_fx.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.
        out_dir: path to save the result txt 'deploy_device.txt'. If None, will
            save in the current directory. Default: None

    Returns:
        A dict of model deploy infos with schema
            * KEY (str): module name
            * VALUE (Tuple): (deploy device(BPU or CPU), module type)
    """
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            "input model must be a GraphModule, "
            + "Got type:"
            + str(type(model))
            + " Please make "
            + "sure to follow the tutorials."
        )

    module_device = {}
    bpu_module_leafs = (
        set(get_hybrid_qat_module_mappings().values())
        | set(get_hybrid_quantized_module_mappings().keys())
        | set(get_hybrid_quantized_module_mappings().values())
    )
    bpu_func_leafs = get_hybrid_supported_functions()
    bpu_method_leafs = get_hybrid_supported_methods()
    for node in model.graph.nodes:
        if node.op == "call_module":
            m_type = type(model.get_submodule(node.target))
            if m_type not in (torch.nn.Identity, horizon_nn.Identity):
                is_quant_dequant = m_type in (
                    horizon_nn.qat.QuantStub,
                    horizon_nn.qat.DeQuantStub,
                    horizon_nn.quantized.Quantize,
                    horizon_nn.quantized.DeQuantize,
                )
                module_device[node.target] = (
                    "BPU"
                    if m_type in bpu_module_leafs and not is_quant_dequant
                    else "CPU",
                    "module",
                )
        elif node.op == "call_function":
            module_device[node.name] = (
                "BPU" if node.target in bpu_func_leafs else "CPU",
                "function",
            )
        elif node.op == "call_method":
            m = model.get_submodule(node.args[0].target)
            is_float_module = (
                True
                if (
                    isinstance(m, horizon_nn.quantized.FloatFunctional)
                    and isinstance(
                        m.activation_post_process,
                        (torch.nn.Identity, horizon_nn.Identity),
                    )
                )
                or type(m) in get_hybrid_qat_module_mappings().keys()
                else False
            )
            module_device[node.name] = (
                "BPU"
                if node.target in bpu_method_leafs and not is_float_module
                else "CPU",
                "method",
            )

    if print_tabulate:
        print(
            tabulate(
                [(k,) + v for k, v in module_device.items()],
                headers=("name", "deploy device", "type"),
            )
        )

    out_dir = "." if out_dir is None else out_dir
    file_path = os.path.join(out_dir, "deploy_device.txt")
    with open(file_path, "w") as f:
        f.write(
            tabulate(
                [(k,) + v for k, v in module_device.items()],
                headers=("name", "deploy device", "type"),
            )
        )

    return module_device
