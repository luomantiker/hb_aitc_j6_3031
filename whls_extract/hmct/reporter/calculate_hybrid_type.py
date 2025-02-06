from typing import Any, Dict, Optional, Tuple

from hmct.ir import DataType, OnnxModel


def update_node_datatype_info(
    quantized_model: OnnxModel, node_info: Dict[str, Dict[str, Any]]
):
    node_datatype_info_dict = {}
    variable_mappings = quantized_model.graph.variable_mappings
    for node in quantized_model.graph.nodes:
        node_datatype_info_dict[node.name] = {}
        node_datatype_info_dict[node.name]["InputType"] = "int8"
        for idx, node_input in enumerate(node.inputs):
            # If multiple inputs, the one with highest precision is displayed.
            if node_input.name not in variable_mappings:
                continue
            value = variable_mappings[node_input.name]
            if value.dtype == DataType.FLOAT32:
                node_datatype_info_dict[node.name]["InputType"] = "float"
                break
            if value.dtype == DataType.INT16:
                node_datatype_info_dict[node.name]["InputType"] = "int16"
            # TODO(yps): 对于某些算子如Transpose输入可能为int32, 但直接将int32
            # 增加到判断条件中, 会导致Conv的输入类型因为bias是int32而被展示为int32,
            # 并不符合我们的预期。当前的处理方式是仅考虑输入0为int32的情况。
            if value.dtype == DataType.INT32 and idx == 0:
                node_datatype_info_dict[node.name]["InputType"] = "int32"
            if value.dtype == DataType.INT64 and idx == 0:
                node_datatype_info_dict[node.name]["InputType"] = "int64"
            if value.dtype == DataType.BOOL:
                node_datatype_info_dict[node.name]["InputType"] = "bool"
    for node_name, datatype in node_datatype_info_dict.items():
        if node_name not in node_info:
            continue
        node_info[node_name]["DataType"] = datatype["InputType"]

    return node_info


def calculate_hybrid_type(
    quantized_model: Optional[OnnxModel] = None,
    hybrid_builder: Optional["HybridBuilder"] = None,  # noqa: F821
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    # obtain model_hybrid_info
    if hybrid_builder is None:
        return {}
    model_hybrid_info = hybrid_builder.model_info()
    # update model_hybrid_info
    if quantized_model is not None:
        model_hybrid_info = update_node_datatype_info(
            quantized_model, model_hybrid_info
        )

    return model_hybrid_info
