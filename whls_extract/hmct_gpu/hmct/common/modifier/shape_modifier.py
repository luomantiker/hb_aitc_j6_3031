import logging
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

from .constant_folding import constant_folding

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxVariable


def modify_model_shape(
    input_shapes: Dict[str, Sequence[int]],
    onnx_model: "OnnxModel",
    referenced_model: "OnnxModel",
) -> "OnnxModel":
    """修改onnx_model以保证其在给定input_shapes下正确推理.

    若修改的是batch维度, 则对模型中的resize, reshape, 常量输入进行处理;
    若修改的是spatial维度, 则对模型中的resize进行处理。
    """
    modified_dim = parse_input_shapes(input_shapes, referenced_model, onnx_model)

    onnx_model = process_resize_op(onnx_model)

    if modified_dim == "batch":
        # 第0维为batch
        batch_size = onnx_model.graph.inputs[0].shape[0]
        onnx_model = process_reshape_op(referenced_model, onnx_model, batch_size)
        if batch_size > 1:
            onnx_model = process_initializer_input(
                referenced_model, onnx_model, batch_size
            )
            onnx_model = process_elementwise_op(onnx_model, batch_size)

    return infer_and_check(onnx_model)


def parse_input_shapes(
    input_shapes: Dict[str, Sequence[int]],
    referenced_model: "OnnxModel",
    onnx_model: "OnnxModel",
) -> str:
    """检查input_shapes合法性后写入onnx_model, 判断并返回修改输入shape的维度."""
    batch_sizes = []
    modify_spatial = False
    for input_name in referenced_model.graph.input_names:
        if input_name not in input_shapes:
            raise ValueError(
                f"Please check if {input_name} is included in the model inputs!"
            )
        batch_sizes.append(input_shapes[input_name][0])
        if list(input_shapes[input_name][1:]) != list(
            onnx_model.graph.input_shapes[input_name][1:]
        ):
            modify_spatial = True

    if len(set(batch_sizes)) > 1:
        raise ValueError("Input_shape must have same batch_size")

    if batch_sizes[0] != 1 and modify_spatial:
        raise ValueError(
            "Batch and spatial dimensions cannot be changed simultaneously."
        )

    # 设置为给定input_shape
    for input_name in input_shapes:
        onnx_model.graph.input_mappings[input_name].shape = input_shapes[input_name]

    return "spatial" if modify_spatial else "batch"


def parse_node_info_from_referenced_model(
    referenced_model: "OnnxModel", batch_size: int
) -> Dict[str, np.ndarray]:
    """获取修改输入shape后各个Reshape节点的target shape取值.

    该函数首先修改referenced_model的input_shape并清空模型中所有variable的shape信息; 然后
    使用constant_folding将模型中reshape的原本动态的shape信息常量折叠到各个reshape中
    进而可以得到修改shape后的每个reshape的target shape属性; 最后会保存此reshape
    信息, 方便后续修改量化后reshape的target shape时读取调用

    Args:
        referenced_model: 未进行过优化的浮点模型
        batch_size: 指定输入的batch size
    """
    # 2. 将输入shape配置成指定的shape信息
    for input_var in referenced_model.graph.inputs:
        # 当前J5上有修改量化模型layout的功能, 且在J5上self.input_shapes
        # 是根据修改layout后的量化模型的输入以及batch信息得到的, 因此
        # 可能会出现self.input_shapes和当前model的layout对不上的情况
        # 故这里不能直接将self.input_shapes作为当前model的输入来进行推理
        input_var.shape = [batch_size] + list(input_var.shape)[1:]

    # 3. 针对原始浮点模型中的Reshape节点调整为支持batch的情况
    referenced_model = batch_format_reshape_op(referenced_model)
    # 4. 调用常量折叠, 计算得到包含动态target_shape的Reshape在新输入shape下
    # 对应target shape并检查是否合理
    referenced_model = constant_folding(referenced_model)
    referenced_model = infer_and_check(referenced_model)

    # 5. 模型已经是batch-format, 记录Reshape/Squeeze/Unsqueeze/Concat/ScatterND的信息
    target_shapes = {}
    for onnx_node in referenced_model.graph.type2nodes["Reshape"]:
        shape_var = onnx_node.inputs[1]
        if shape_var.is_param:
            target_shapes[shape_var.name] = np.array(shape_var.value)

    squeeze_unsqueeze_nodes = []
    concat_scatternd_nodes = []
    # TODO(jilei.hou): GatherElements可能也需要被记录
    for node in referenced_model.graph.nodes:
        if node.op_type in ["Squeeze", "Unsqueeze"]:
            squeeze_unsqueeze_nodes.append(node)
        if node.op_type in ["Concat", "ScatterND"]:
            concat_scatternd_nodes.append(node)

    squeeze_unsqueeze_output_shapes = {}
    for onnx_node in squeeze_unsqueeze_nodes:
        if onnx_node.outputs[0].shape is not None:
            squeeze_unsqueeze_output_shapes[onnx_node.name] = np.array(
                onnx_node.outputs[0].shape
            )

    node_input_var = {}
    for onnx_node in concat_scatternd_nodes:
        for input_var in onnx_node.inputs:
            if input_var.shape is not None and input_var.is_param:
                node_input_var[input_var.name] = input_var.value

    return target_shapes, squeeze_unsqueeze_output_shapes, node_input_var


def infer_and_check(model: "OnnxModel") -> "OnnxModel":
    """执行infer_shapes并对模型合法性做检查."""
    try:
        model.infer_shapes()
        check_reshape_target_shape(model)
        model.check_validity()
    except Exception as exc:
        raise RuntimeError(
            "This model does not support to modify input shape, "
            "please use original input shape and try again."
        ) from exc
    return model


def process_reshape_op(
    referenced_model: "OnnxModel", onnx_model: "OnnxModel", batch_size: int
) -> "OnnxModel":
    """修改输入batch对针对reshape算子进行处理.

    首先通过原始浮点模型在给定input_shape下进行shape_inference, 并记录reshape节点的
    target_shape; 然后对onnx_model中节点的shape进行修改, 并完成shape_inference.

    对于target_shape, 如果在reference_model推理时已经记录则直接修改;
    否则, 默认情况下, 将第0维修改为-1
    """
    # 原始浮点模型在给定input_shapes下进行shape_inference
    target_shapes, squeeze_unsqueeze_output_shapes, _ = (
        parse_node_info_from_referenced_model(referenced_model, batch_size)
    )

    reshape_param = [
        n.inputs[-1] for n in onnx_model.graph.nodes if n.op_type == "Reshape"
    ]

    for param in reshape_param:
        if param.name in target_shapes:
            # 如果在记录的target_shapes中找到了该reshape的target_shape值,
            # 那么直接用该值替换
            # todo(df.li)此时的模型为量化模型, 经历过各种Reshape融合替换等操作
            # 可能会导致Reshape的name发生改变进而引起索引不到原始shape信息的情况
            param.value = target_shapes[param.name]
        elif param.value is not None and (param.value > 0).all():
            # 如果target_shape全为正, 那么将第0维理解成batch所在维度, 并将其修改为-1
            param.value[0] = -1
    for onnx_node in onnx_model.graph.type2nodes["Reshape"]:
        if onnx_node.name in squeeze_unsqueeze_output_shapes:
            onnx_node.inputs[-1].value = squeeze_unsqueeze_output_shapes[onnx_node.name]

    return onnx_model


def process_initializer_input(
    referenced_model: "OnnxModel", onnx_model: "OnnxModel", batch_size: int
) -> "OnnxModel":
    """修改输入batch后针对存在常量输入op进行处理.

    首先通过原始浮点模型在给定input_shape下进行shape_inference,
    并记录Concat/ScatterND节点的常量输入.

    对于Concat/ScatterND节点的常量输入, 如果在reference_model推理时已经记录则直接修改;
    否则, 通过copy数据batch_size份的方式支持修改成batch mode
    """
    # 原始浮点模型在给定input_shapes下进行shape_inference
    _, _, node_input_var = parse_node_info_from_referenced_model(
        referenced_model, batch_size
    )

    supported_nodes = [
        node
        for node in onnx_model.graph.nodes
        if node.op_type in ["Concat", "ScatterND", "GatherElements"]
    ]

    for node in supported_nodes:
        for input_var in node.inputs:
            input_var = get_initializer_input(input_var)
            if input_var and input_var.shape is not None and input_var.is_param:
                if input_var.name in node_input_var:
                    input_var.value = node_input_var[input_var.name]
                else:
                    array = input_var.value
                    array = np.repeat(array, batch_size, axis=0)
                    input_var.value = array

    return onnx_model


def process_resize_op(onnx_model: "OnnxModel") -> "OnnxModel":
    """针对onnx_model中的resize op进行处理.

    对于opset=10的resize, 输入的个数为2, 且input[-1]为 resize ratio,
    不做修改.
    对于opset=11的resize和HzResize11, 输入个数为3~4, 且input[2]
    是resize ratio(如果有的话), input[3]是指定的resize shape.
    为了支持输入shape的修改, 我们将resize shape转换为resize ratio,
    且令resize ratio[0] = 1.
    """
    resize_nodes = [
        n
        for n in onnx_model.graph.nodes
        if ((n.op_type == "HzResize11" or n.op_type == "Resize") and len(n.inputs) == 4)
    ]

    for resize_node in resize_nodes:
        # 只有当resize shape被给出,并且input_shape的非batch维度均为正数才进行转换
        if (
            not resize_node.inputs[3].is_param
            or resize_node.inputs[0].shape is None
            or not all(
                isinstance(dim, int) and dim > 0
                for dim in resize_node.inputs[0].shape[1:]
            )
        ):
            continue

        scales_shape = [1.0]
        scales_shape.extend(
            resize_node.inputs[3].value[idx] / resize_node.inputs[0].shape[idx]
            for idx in range(1, len(resize_node.inputs[0].shape))
        )

        if len(resize_node.inputs[2].dest_ops) > 1:
            scales = onnx_model.graph.create_variable(
                name=resize_node.name + "_" + resize_node.inputs[2].name,
                is_param=True,
                is_attr=False,
                value=np.array(scales_shape, dtype=np.float32),
            )
            resize_node.replace_input(2, scales)
        else:
            resize_node.inputs[2].value = np.array(scales_shape, dtype=np.float32)
        resize_node.remove_input(3)

    return onnx_model


def process_elementwise_op(onnx_model: "OnnxModel", batch_size: int) -> "OnnxModel":
    """修改输入batch后针对elementwise op进行处理.

    对于elementwise op, 如果存在常量输入时可能需要对该常量输入进行broadcast, 但因为
    elementwise op本身具有broadcast能力, 因此在一些场景下是不需要broadcast的

    TODO(jilei.hou): 考虑将elementwise算子也修改为从referenced model中记录信息的形式
    """
    elementwise_ops = [
        "Mul",
        "Add",
        "Sub",
        "Div",
        "HzSElementwiseMul",
        "HzSElementwiseAdd",
        "HzSElementwiseSub",
    ]
    elementwise_inputs = []
    for node in onnx_model.graph.nodes:
        if node.op_type in elementwise_ops:
            input0_shape = node.inputs[0].shape
            input1_shape = node.inputs[1].shape
            # 以下几种情况, 不需要复制batch_size份
            # 1. 输入shape为空
            # 2. 两个输入shape的长度不一致
            # 3. 输入shape是动态的
            # 4. 如果两个输入的长度均为1
            #    TODO(jilei.hou): 这种情况大概率是由于常量折叠没有完全折叠, 暂不处理
            # 5. 存在某个输入shape是全1的(可自动broadcast)
            if (
                input0_shape is None
                or input1_shape is None
                or len(input0_shape) != len(input1_shape)
                or any(isinstance(s, str) for s in input0_shape)
                or any(isinstance(s, str) for s in input1_shape)
                or sum(input0_shape) == len(input0_shape)
                or sum(input1_shape) == len(input1_shape)
                or len(input0_shape) == 1
                or len(input1_shape) == 1
            ):
                continue
            elementwise_inputs.extend(
                (
                    get_initializer_input(node.inputs[0]),
                    get_initializer_input(node.inputs[1]),
                )
            )
    for elementwise_input in elementwise_inputs:
        if elementwise_input is not None:
            # 如果常量输入的第0维是1, 可以自己broadcast
            if elementwise_input.shape[0] == 1:
                continue
            array = elementwise_input.value
            array = np.repeat(array, batch_size, axis=0)
            elementwise_input.value = array
    return onnx_model


# J6针对常量输入需要修改校准节点的输入
def get_initializer_input(input_var: "OnnxVariable") -> Optional["OnnxVariable"]:
    if input_var.src_op and input_var.src_op.op_type == "HzCalibration":
        input_var = input_var.src_op.inputs[0]

    return input_var if input_var.is_param else None


def check_reshape_target_shape(model: "OnnxModel") -> None:
    """检查reshape算子的输入shape是否能够转换为target shape."""
    input_shapes = {}
    target_shapes = {}
    for reshape_node in model.graph.type2nodes["Reshape"]:
        reshape_name = reshape_node.name
        input_shapes[reshape_name] = reshape_node.inputs[0].shape
        if reshape_node.inputs[1].is_param:
            target_shapes[reshape_name] = reshape_node.inputs[1].value.tolist()

    for reshape_name, input_shape in input_shapes.items():
        # check whether input shape can be reshaped to target shape
        if reshape_name not in target_shapes:
            continue
        target_shape = target_shapes[reshape_name]
        if (
            input_shape
            and target_shape
            and all(isinstance(dim_val, int) for dim_val in input_shape)
            and all(isinstance(dim_val, int) for dim_val in target_shape)
            and (np.array(input_shape) > 0).all()
            and (np.array(target_shape) > 0).all()
        ):
            # model.infer_shapes()无法检查出这种错误, 因此保留check_reshape_target_shape
            assert np.prod(input_shape) == np.prod(target_shape), (
                f"Error occur during reshape info modification for reshape node: "
                f"{reshape_name}. input shape: {input_shape} will be reshape "
                f"to target shape: {target_shape}, which is mismatched."
            )


def batch_format_reshape_op(referenced_model: "OnnxModel") -> "OnnxModel":
    """Modify model with reshape node batch format.

    Args:
        referenced_model: Model has reshape nodes with all positive target_shape.

    Returns:
        onnx model whose reshape node is modified.
    """
    target_shapes = [n.inputs[1] for n in referenced_model.graph.type2nodes["Reshape"]]
    for target_shape in target_shapes:
        arr = target_shape.value
        if arr is None:
            continue
        if (arr > 0).all():
            arr[0] = -1
            logging.warning(
                "In order to support batch format, the target reshape of "
                f"reshape param ({target_shape.name}) will be modified to "
                f"({', '.join(str(shape) for shape in arr)}), "
                "however this modification is not guaranteed to be correct!",
            )
    return referenced_model
