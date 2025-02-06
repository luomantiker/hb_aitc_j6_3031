from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Union, cast

from .onnx_graph import OnnxGraph
from .onnx_model import OnnxModel
from .onnx_variable import OnnxVariable

if TYPE_CHECKING:
    from .onnx_node import OnnxNode


def extract_submodel(
    onnx_model: OnnxModel,
    input_vars: Optional[Sequence[Union[str, OnnxVariable]]] = None,
    output_vars: Optional[Sequence[Union[str, OnnxVariable]]] = None,
    infer_shapes: bool = True,
    check_model: bool = True,
) -> OnnxModel:
    # preprocess {input/output}_vars
    input_vars = (
        list(input_vars) if input_vars is not None else list(onnx_model.graph.inputs)
    )
    output_vars = (
        list(output_vars) if output_vars is not None else list(onnx_model.graph.outputs)
    )

    # further validity check for input_vars
    for input_idx, input_var in enumerate(input_vars):
        if isinstance(input_var, str):
            input_var = onnx_model.graph.variable_mappings[input_var]
        assert isinstance(input_var, OnnxVariable), (
            f"type(input_var) should be either str or "
            f"OnnxVariable, but got {type(input_var)}."
        )
        assert input_var.owning_graph is onnx_model.graph, (
            f"The owning_graph of variable {input_var.name} "
            f"and onnx model's graph are different."
        )
        input_vars[input_idx] = input_var
    # remove potential duplicate input variables
    input_vars = cast(List[OnnxVariable], list(set(input_vars)))
    # remove input variables which are model parameters or dummy activations
    input_vars = cast(
        List[OnnxVariable],
        [
            input_var
            for input_var in input_vars
            if not input_var.is_param and not input_var.is_dummy
        ],
    )

    # further validity check for output_vars
    for output_idx, output_var in enumerate(output_vars):
        if isinstance(output_var, str):
            output_var = onnx_model.graph.variable_mappings[output_var]
        assert isinstance(output_var, OnnxVariable), (
            f"type(output_var) should be either str or "
            f"OnnxVariable, but got {type(output_var)}."
        )
        assert output_var.owning_graph is onnx_model.graph, (
            f"The owning_graph of variable {output_var.name} "
            f"and onnx model's graph are different."
        )
        output_vars[output_idx] = output_var
    # remove potential duplicate output variables
    output_vars = cast(List[OnnxVariable], list(set(output_vars)))
    # remove output variables which are dummy activations
    output_vars = cast(
        List[OnnxVariable],
        [output_var for output_var in output_vars if not output_var.is_dummy],
    )

    # obtain shape && dtype info for submodel inputs && outputs
    if infer_shapes:
        onnx_model.infer_shapes()

    # Step-1. We figure out all the required nodes && variables && graphs
    # from the given output variables.
    # unresolved_vars represent all the potential required variables
    unresolved_vars: Set[OnnxVariable] = set(output_vars)
    # required_vars represent all the required variables (include attr vars)
    required_vars: Set[OnnxVariable] = set()
    # required_nodes represent all the required nodes
    required_nodes: Set[OnnxNode] = set()
    # required_graphs represent all the required sub-graphs
    required_graphs: Set[OnnxGraph] = set()
    # we expand the required_vars and required_nodes step-by-step
    while unresolved_vars:
        onnx_var = unresolved_vars.pop()
        required_vars.add(onnx_var)
        if onnx_var in input_vars or onnx_var.src_op is None:
            continue
        required_nodes.add(onnx_var.src_op)
        for input_var in onnx_var.src_op.inputs:
            if input_var not in required_vars:
                unresolved_vars.add(input_var)
    # add node outputs && node attr_var to required_vars
    # add node attr_graphs to required_graphs
    for onnx_node in required_nodes:
        required_vars.update(onnx_node.outputs)
        for attr_val in onnx_node.attributes.values():
            attr_vec = attr_val if isinstance(attr_val, list) else [attr_val]
            for item_val in attr_vec:
                if isinstance(item_val, OnnxVariable):
                    required_vars.add(item_val)
                if isinstance(item_val, OnnxGraph):
                    required_graphs.add(item_val)

    # Step-2. We reconstruct the extracted submodel based on
    # required nodes && variables.
    # Step-2-1. create empty extracted submodel
    extracted_submodel = OnnxModel(
        ir_version=onnx_model.ir_version,
        opset_import=onnx_model.opset_import,
        producer_name="horizon_nn.ir.extract_submodel",
        graph_name=onnx_model.graph.name,
    )
    # Step-2-2. copy graph from collected graphs/variables/nodes/inputs/outputs
    input_vars = cast(
        List[OnnxVariable],
        [input_var for input_var in input_vars if input_var in required_vars],
    )
    extracted_submodel.graph.copy_from(
        variables=required_vars,
        graphs=required_graphs,
        nodes=required_nodes,
        inputs=input_vars,
        outputs=output_vars,
    )
    extracted_submodel.sort_topologically()

    # Step-3. We check the validity of the extracted submodel
    if check_model:
        extracted_submodel.check_validity()

    return extracted_submodel
