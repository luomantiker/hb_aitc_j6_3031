import copy
from typing import Any, Callable, Dict, Optional

from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    attach_qualified_name,
    is_leaf_module,
    register_hook_on_leaf,
    swap_ff_with_horizonff,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from tabulate import tabulate


@typechecked
def get_module_called_count(
    model: torch.nn.Module,
    example_inputs: Any,
    check_leaf_module: Optional[Callable] = None,
    print_tabulate: bool = True,
) -> Dict[str, int]:
    """
    Count called times for all leaf modules in a model.

    Args:
        model (torch.nn.Module): The input model.
        example_inputs (Any[Tensor]): The input data feed to model.
        check_leaf_module (callable, optional): A function to check if
            a module is leaf. Pass None to use pre-defined `is_leaf_module`.
            Defaults to None.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.

    Returns:
        Dict[str, int]:
            The qualified name and called times of each leaf module.
    """
    if check_leaf_module is None:
        check_leaf_module = is_leaf_module

    model = copy.deepcopy(model)

    swap_ff_with_horizonff(model)
    attach_qualified_name(model)

    module_refcount = {}

    def _count_call_hook(module, input, output):
        module_refcount[module._qualified_name] += 1

    handler_dict = register_hook_on_leaf(
        model, _count_call_hook, check_leaf_module=check_leaf_module
    )

    for name in handler_dict:
        module_refcount[name] = 0

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)
    del model

    if print_tabulate:
        print(
            tabulate(module_refcount.items(), headers=["name", "called times"])
        )

    return module_refcount
