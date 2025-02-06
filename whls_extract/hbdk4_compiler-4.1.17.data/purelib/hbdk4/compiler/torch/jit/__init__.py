from typing import Optional, Any, List
import torch

from hbdk4.compiler.frontend.registry import OpConvertorRegistry
from hbdk4.compiler._mlir_libs._hbdk import _const_fold
from hbdk4.compiler.overlay import Module

from hbdk4.compiler import ir as mlir

from hbdk4.compiler.torch.jit.adaptor import TorchJitGraphAdaptor
import hbdk4.compiler.torch.aten  # noqa: F401
import hbdk4.compiler.torch.jit.prim  # noqa: F401
import hbdk4.compiler.torch.jit.horizon  # noqa: F401


def export(
    jit: torch.jit.ScriptModule,
    example_input: Any,
    *,
    name: Optional[str] = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    lower_non_tensor: bool = True,
) -> Module:
    """export a torch.jit.ScriptModule to hbir mlir

    Args:
        jit (torch.jit.ScriptModule): a ScriptModule created from torch.jit.trace
        example_input (Any): input format of the ScriptModule, used for analysis
        name (Optional[str], optional): rename the function. "None" means uses the name recorded in ScriptModule.
        input_names (List[str], optional): rename inputs. "None" means uses input names recorded in ScriptModule.
        output_names (List[str], optional): rename outputs. "None" means uses input names recorded in ScriptModule.
        lower_non_tensor (bool, optional): flatten the pytree in ScriptModule input and return or keep the tree in hbir.

    Returns:
        Module: a helper for mlir.Module that manages hbdk operations
    """

    if isinstance(example_input, (torch.Tensor, dict)):
        example_input = [example_input]

    module = mlir.Module.create()
    with mlir.InsertionPoint(module.body):
        graph = TorchJitGraphAdaptor(jit, *example_input, name=name)

        if input_names is not None:
            if len(input_names) != len(graph.operands):
                raise ValueError(
                    "length of input_names {} mismatch graph number of inputs {}".format(
                        len(input_names), len(graph.operands)
                    )
                )

            for name, operand in zip(input_names, graph.operands):
                operand.name = name

        if output_names is not None:
            if len(output_names) != len(graph.results):
                raise ValueError(
                    "length of output_names {} mismatch graph number of outputs {}".format(
                        len(output_names), len(graph.results)
                    )
                )

            for name, result in zip(output_names, graph.results):
                result.name = name

        graph.emit_mlir_func_op(OpConvertorRegistry(), lower_non_tensor)

        _const_fold(module.operation, module.context)
        return Module(module)


def statistics(jit: torch.jit.ScriptModule, example_input: Any):
    """Print op statics of given ScriptModule module.

    Args:
        jit (torch.jit.ScriptModule): a ScriptModule created from torch.jit.trace
    """

    if isinstance(example_input, (torch.Tensor, dict)):
        example_input = [example_input]

    graph = TorchJitGraphAdaptor(jit, *example_input)

    print("ops encountered in torchscript graph @", graph.name)
    graph.statistics()
