from typing import Optional, List
import torch

from hbdk4.compiler.frontend.registry import OpConvertorRegistry
from hbdk4.compiler._mlir_libs._hbdk import _const_fold
from hbdk4.compiler.overlay import Module

from hbdk4.compiler import ir as mlir

from hbdk4.compiler.torch.export.adaptor import TorchExportGraphAdaptor
import hbdk4.compiler.torch.aten  # noqa: F401


def export(
    prog: torch.export.ExportedProgram,
    *,
    name: Optional[str] = None,
    input_names: List[str] = None,
    output_names: List[str] = None,
    lower_non_tensor: bool = True,
) -> Module:
    """export a torch.export.ExportedProgram to hbir mlir

    Args:
        prog (torch.export.ExportedProgram): a ExportedProgram captured by torch.export.export
        name (Optional[str], optional): rename the function. "None" means uses the name recorded in ExportedProgram.
        input_names (List[str], optional): rename inputs. "None" means uses input names recorded in ExportedProgram.
        output_names (List[str], optional): rename outputs. "None" means uses input names recorded in ExportedProgram.
        lower_non_tensor (bool, optional): flatten the pytree in ExportedProgram input and return or keep the tree in hbir.

    Returns:
        Module: a helper for mlir.Module that manages hbdk operations
    """

    module = mlir.Module.create()
    with mlir.InsertionPoint(module.body):
        graph = TorchExportGraphAdaptor(prog, name=name)

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
        exported = Module(module)
        exported.functions[0]._in_tree_spec = prog.call_spec.in_spec
        exported.functions[0]._out_tree_spec = prog.call_spec.out_spec
        return exported


def statistics(prog: torch.export.ExportedProgram):
    """Print op statics of given ExportedProgram module.

    Args:
        prog (torch.export.ExportedProgram): a ExportedProgram captured by torch.export.export
    """

    graph = TorchExportGraphAdaptor(prog)

    print("ops encountered in torch.export.ExportedProgram graph @", graph.name)
    graph.statistics()
