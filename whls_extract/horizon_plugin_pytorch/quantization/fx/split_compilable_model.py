import copy
import logging
from collections import OrderedDict, defaultdict
from operator import getitem
from typing import Any, Tuple

import torch
from torch.fx import Node

from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization.quantization_mappings import (
    get_qat_module_mappings,
)
from horizon_plugin_pytorch.utils.pytree_packer import PytreePacker
from .graph_module import ObservedGraphModule
from .utils import graph_repr

__all__ = ["get_compilable_submodule", "split_compilable_model"]

logger = logging.getLogger(__name__)
_compilable_functions = set()
_compilable_modules = set()
_compilable_methods = defaultdict(set)  # mapping from method name to owners


def init_compile_collection():
    # add functions in QTensor dispatcher
    for func in QTensor.get_dispatcher().keys():
        if func.__qualname__.startswith(
            "_TensorBase"
        ) or func.__qualname__.startswith("Tensor"):
            _compilable_methods[func.__name__].add(QTensor)
        else:
            _compilable_functions.add(func)

    # add QAT Modules
    _compilable_modules.update(get_qat_module_mappings().values())

    # add getitem
    _compilable_functions.add(getitem)


init_compile_collection()


def unfold_node_args(n: Node):
    unfold_args = []

    for arg in n.args + tuple(n.kwargs.values()):
        if isinstance(arg, Node):
            unfold_args.append(arg)
        elif isinstance(arg, (list, tuple)):
            for v in arg:
                if isinstance(v, Node):
                    unfold_args.append(v)
        elif isinstance(arg, dict):
            for v in arg.values():
                if isinstance(v, Node):
                    unfold_args.append(v)

    return unfold_args


def if_node_is_compilable(n: Node, model: ObservedGraphModule):
    if n.op == "call_module":
        return type(model.get_submodule(n.target)) in _compilable_modules
    elif n.op == "call_function":
        return n.target in _compilable_functions
    elif n.op == "call_method":
        owner_node = n.args[0]
        if n.target in _compilable_methods and isinstance(owner_node, Node):
            if owner_node.op == "get_attr":
                if owner_node.target == "_self":
                    # use root because _self is subclass of root
                    # cannot get the root by get_submodule
                    return type(model.root) in _compilable_methods[n.target]
                else:
                    try:
                        mod = model.get_submodule(owner_node.target)
                        return type(mod) in _compilable_methods[n.target]
                    except AttributeError:
                        return False
            else:
                return QTensor in _compilable_methods[n.target]
    elif n.op == "get_attr":
        # Preserve all get_attr nodes in compilable graph and outer graph.
        # Unused node will be cleared by eliminate_dead_code.
        return True
    else:
        return False


@fx_helper.wrap()
def reorganize_output(*args):
    """Organize the output of compilable part into a tuple.

    There must be only one output node in FX Graph. If a module has multi
    output, we use a call_function node to gather them, thus can be outputed
    all in one.
    """
    return args


def split_compilable_model(
    model: ObservedGraphModule,
    compilable_part_name: str = "compilable_model",
    output_packer_name="compilable_output_packer",
) -> ObservedGraphModule:
    """Automatically split the compilable part into a submodule.

    Args:
        model: Input QAT/Calib model.
        compilable_part_name: The name of compilable submodule.
            Defaults to "compilable_model".
        output_packer_name: The name of output packer for compilable part.
            Defaults to "compilable_output_packer".

    Returns:
        ObservedGraphModule: The computation graph of output model should be
            the same as input model, and the compilable part is called
            as a submodule.
    """
    if hasattr(model, compilable_part_name):
        raise ValueError(
            "Input model already has attribute {}, "
            "please specify another `compilable_part_name`.".format(
                compilable_part_name
            )
        )
    if hasattr(model, output_packer_name):
        raise ValueError(
            "Input model already has attribute {}, "
            "please specify another `output_packer_name`.".format(
                output_packer_name
            )
        )

    logger.debug("in split_compilable_model")

    ######################################################################
    # 1. Detect static nodes that can be handled by constant derivation
    ######################################################################
    is_node_static = defaultdict(
        lambda: False
    )  # whose output is not determined by model input

    for n in model.graph.nodes:
        n: Node
        if n.op == "get_attr":
            if len(n.args) == 0:
                # get a submodule from root
                is_node_static[n.name] = True
            else:
                # get member from container
                assert isinstance(n.args[0], Node)
                if is_node_static[n.args[0].name]:
                    is_node_static[n.name] = True
        elif n.op in (
            "call_function",
            "call_module",
            "call_method",
            "output",
        ):
            if all(
                isinstance(arg, Node) and is_node_static[arg.name]
                for arg in unfold_node_args(n)
            ):
                is_node_static[n.name] = True
                if n.op == "output":
                    raise RuntimeError(
                        "Static output node detected. "
                        "Output of the model is always unchanged."
                    )
        else:
            if n.op != "placeholder":
                raise ValueError("Unrecognized node op {}".format(n.op))

    ######################################################################
    # 2. Distinguish compilable operations
    ######################################################################
    is_node_compilable = defaultdict(lambda: False)

    for n in model.graph.nodes:
        n: Node
        if if_node_is_compilable(n, model):
            is_node_compilable[n.name] = True

    logger.debug(
        "input graph is:\n"
        + graph_repr(
            model.graph,
            {"is_compilable": is_node_compilable, "is_static": is_node_static},
        )
    )

    ######################################################################
    # 3. Generate compilable graph and model
    ######################################################################
    input_pairs = OrderedDict()
    output_pairs = OrderedDict()

    def insert_placeholder(n: Node):
        # Leave getitem in the outer model to make compilable model
        # input a tuple of Tensor
        if is_node_compilable[n.name] and n.target is not getitem:
            raise RuntimeError("Illegal input node")

        for u in list(n.users.keys()):
            if is_node_compilable[u.name] and u.target is not getitem:
                if n.name in input_pairs:
                    input_node = input_pairs[n.name]
                else:
                    compilable_graph.inserting_before()
                    input_node = compilable_graph.create_node(
                        "placeholder", target=n.name
                    )
                # Do not use replace_all_uses_with in case that n is used by
                # other uncompilable node
                u.replace_input_with(n, input_node)
                # Put generated placeholder in compilable set to
                # supress duplicated generation
                is_node_compilable[input_node.name] = True
                input_pairs[n.name] = input_node
            else:
                insert_placeholder(u)

    def record_output(n: Node):
        # Leave getitem in the compilable model to make compilable model
        # output a tuple of Tensor
        if is_node_compilable[n.name]:
            raise RuntimeError("Illegal input node")

        for arg in unfold_node_args(n):
            if is_node_compilable[arg.name]:
                if arg.op != "get_attr":
                    output_pairs[arg.name] = arg
            else:
                record_output(arg)

    compilable_graph = copy.deepcopy(model.graph)
    for n in compilable_graph.nodes:
        n: Node
        # Generate placeholder for dangling input
        if n.op == "placeholder" and n.target not in input_pairs:
            insert_placeholder(n)

        elif n.op == "output":
            # Record dangling output
            record_output(n)

            # Reuse the origin output node by replacing its args
            compilable_graph.inserting_before(n)
            # Gather recorded output into one list
            # NOTE: Use dict will cause trace error
            tuple_output_node = compilable_graph.create_node(
                "call_module",
                target=output_packer_name,
                args=(tuple(output_pairs.values()),),
            )
            n.args[0].replace_all_uses_with(tuple_output_node)

    logger.debug("quant graph before clean:\n" + graph_repr(compilable_graph))

    output_packer_mod = PytreePacker()
    model.add_module(output_packer_name, output_packer_mod)
    compilable_graph.owning_module = model
    compilable_graph.eliminate_dead_code()

    # Remove unused placeholders manually because eliminate_dead_code does
    # not clear them
    for n in compilable_graph.nodes:
        if n.op == "placeholder" and len(n.users) == 0:
            if n in input_pairs.values():
                input_pairs.pop(n.target)
            compilable_graph.erase_node(n)

    logger.debug("quant graph is:\n" + graph_repr(compilable_graph))

    compilable_model = ObservedGraphModule(model, compilable_graph)

    ######################################################################
    # 4. Generate outer graph and model
    ######################################################################
    outer_graph = copy.deepcopy(model.graph)
    compilable_model_args = []
    compilable_model_kwargs = {}
    for n in outer_graph.nodes:
        n: Node
        if n.name in input_pairs:
            # Search inputs for compilable submodule
            compilable_model_kwargs[n.name] = n

            # After the searching complete, generate the calling
            # operation of compilable submodule
            if len(compilable_model_kwargs) == len(input_pairs):
                # Reorder inputs
                for name in reversed(input_pairs):
                    compilable_model_args.append(compilable_model_kwargs[name])

                outer_graph.inserting_after(n)
                call_compilable_model_node = outer_graph.create_node(
                    "call_module",
                    target=compilable_part_name,
                    args=tuple(compilable_model_args),
                )
                outer_graph.inserting_after(call_compilable_model_node)
                output_packer_getter = outer_graph.create_node(
                    "get_attr",
                    target=output_packer_name,
                    args=(),
                )
                outer_graph.inserting_after(output_packer_getter)
                reconstruct_output_node = outer_graph.create_node(
                    "call_method",
                    target="reconstruct",
                    args=(output_packer_getter, call_compilable_model_node),
                )
                outer_graph.inserting_after(reconstruct_output_node)
                # Checkout each output from the output list
                # of compilable submodule
                for i, name in enumerate(output_pairs):
                    output_pairs[name] = outer_graph.create_node(
                        "call_function",
                        target=getitem,
                        args=(reconstruct_output_node, i),
                    )
        elif n.name in output_pairs:
            # Connect the compilable submodule output with post processes
            assert output_pairs[n.name] in outer_graph.nodes
            n.replace_all_uses_with(output_pairs[n.name])

    # The deep copy is necessary to avoid infinate recursion when
    # copy the compilable_model
    model = copy.deepcopy(model)

    model.add_submodule(compilable_part_name, compilable_model)

    outer_graph.owning_module = model
    outer_graph.eliminate_dead_code()

    logger.debug("outer graph is:\n" + graph_repr(outer_graph))

    outer_model = ObservedGraphModule(model, outer_graph)

    outer_model._compilable_submodule_name = compilable_part_name

    setattr(outer_model, output_packer_name, output_packer_mod)
    setattr(
        getattr(outer_model, compilable_part_name),
        output_packer_name,
        output_packer_mod,
    )

    return outer_model


def get_compilable_submodule(
    model: torch.nn.Module,
    example_inputs: Tuple[Any] = None,
    compilable_submodule_name: str = None,
) -> Tuple[torch.nn.Module, Any]:
    """Get compileable submodule in a model and its input.

    Args:
        model: Input model.
        example_inputs: Example inputs to the outter model. Defaults to None.
        compilable_submodule_name: Submodule name, leave empty to detect it
            autolly. Defaults to None.

    Raises:
        ValueError: Raise ValueError if user do not give
            compilable_submodule_name and cannot detect autolly.

    Returns:
        Compilable submodule and its inputs.
    """
    if compilable_submodule_name is None:
        if not hasattr(model, "_compilable_submodule_name"):
            raise ValueError(
                "Input model must has attr '_compilable_submodule_name' "
                "to detect submodule name"
            )
        compilable_submodule_name = model._compilable_submodule_name

    submod = model.get_submodule(compilable_submodule_name)

    if example_inputs is not None:
        sub_input = []

        def _hook(mod, input):
            sub_input.append(input)

        handle = submod.register_forward_pre_hook(_hook)
        model(*example_inputs)
        handle.remove()

        submod_input = sub_input[0]
    else:
        submod_input = None

    return submod, submod_input
