from typing import List, Union, Tuple


def get_type_for_hbtl_call(attached_op):
    schema = attached_op.schema
    node_type = attached_op.type + "::" + schema.namespace + "::" + schema.signature
    return node_type


node_type_mapping = {
    "qnt.quantize": "Quantize",
    "qnt.dequantize": "Dequantize",
    "b30vpu.quantize": "Quantize",
    "b30vpu.dequantize": "Dequantize",
    "hbir.transpose": "Transpose",
    "hbtl.call::quant::qcast": "Quantize",
    "hbtl.call::quant::dcast": "Dequantize",
    "hbtl.call::quant::quantize": "Quantize",
    "hbtl.call::quant::dequantize": "Dequantize",
    "hbtl.call::native::Transpose": "Transpose",
    "hbtl.call::horizon::FilterCopy": "FilterCopy",
    "hbir.cast_type": "Cast",
    "hbir.reshape": "Reshape",
    "hbtl.call::native::Cast": "Cast",
    "hbtl.call::native::Reshape": "Reshape",
    "hbtl.call::native::Softmax": "Softmax",
    "hbtl.call::horizon::RlePostProcess": "RlePostProcess",
}


def remove_done_log(removed, op_names, node_name, op_types, node_type, diagnostic):
    if removed is True:
        if op_names:
            print(
                f"Remove node {node_name} successfully,",
                f"delete node name list {op_names}",
            )
        else:
            print(
                f"Remove op type {node_type} successfully,",
                f"delete node type list {op_types}",
            )
    if removed is False:
        if op_names:
            raise ValueError(
                f"Remove node name {op_names} Failed when deleting {node_name} operator, error: {diagnostic}"
            )
        else:
            raise ValueError(
                f"Remove node type {op_types} Failed when deleting {node_type} operator, error: {diagnostic}"
            )


def remove_loc(loc, func, label, index, op_types=None, op_names=None):
    attached_op = loc.get_attached_op[0]
    node_type = (
        get_type_for_hbtl_call(attached_op)
        if attached_op.type == "hbtl.call"
        else attached_op.type
    )
    node_name = attached_op.name
    removed = None
    diagnostic = None
    if op_names and node_name in op_names:
        if node_type_mapping.get(node_type):
            removed, diagnostic = loc.remove_attached_op()
    elif op_types:
        if (
            node_type in node_type_mapping.keys()
            and node_type_mapping[node_type] in op_types
        ):
            removed, diagnostic = loc.remove_attached_op()

    remove_done_log(removed, op_names, node_name, op_types, node_type, diagnostic)


def run_remove_io_op(func, op_types=None, op_names=None):
    for index in range(len(func.flatten_inputs)):
        loc = func.flatten_inputs[index]
        if not loc.is_removable[0]:
            continue
        remove_loc(loc, func, "inputs", index, op_types, op_names)

    for index in range(len(func.flatten_outputs)):
        # remove FiterCopy op will reduce outputs num, trick here
        if index >= len(func.flatten_outputs):
            break
        loc = func.flatten_outputs[index]
        if not loc.is_removable[0]:
            continue
        remove_loc(loc, func, "outputs", index, op_types, op_names)


def get_removable_io_op(
    func,
    op_types: Union[None, List[str]] = None,
    op_names: Union[None, List[str]] = None,
) -> List[Tuple[str, str]]:
    result = []
    for loc in func.flatten_inputs + func.flatten_outputs:
        if not loc.is_removable[0]:
            continue
        attached_op = loc.get_attached_op[0]
        node_type = (
            get_type_for_hbtl_call(attached_op)
            if attached_op.type == "hbtl.call"
            else attached_op.type
        )
        if op_names and attached_op.name in op_names:
            if node_type_mapping.get(node_type):
                result.append((attached_op.name, node_type_mapping[node_type]))
        elif op_types:
            if (
                node_type in node_type_mapping
                and node_type_mapping[node_type] in op_types
            ):
                result.append((attached_op.name, node_type_mapping[node_type]))
    return result
