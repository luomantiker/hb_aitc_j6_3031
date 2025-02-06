# This file is a original-float onnx to ptq onnx mapping dict which will be used
# by integration team


# node in this set will be converted to LUT.
lut_node_set = (
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cos",
    "Cosh",
    # Ceil和floor在int8下会转成查表,在int16下会直接将浮点算子给到hbdk,
    # 因为int8查表没有尺寸约束,所以直接使用hbir.floor和hbir.ceil的约束
    # "Ceil",
    # "Floor",
    "Erf",
    "Exp",
    "Log",
    "Pow",
    "Reciprocal",
    "Round",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Sqrt",
    "Tan",
    "Tanh",
    "Celu",
    "Clip",
    "Elu",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "LeakyRelu",
    "Mish",
    "Selu",
    "Softplus",
    "Softsign",
    "ThresholdedRelu",
)

# node in this set will be deleted/fused/folded, so the node will be fully supported.
fused_node_set = (
    "BatchNormalization",
    "Constant",
    "ConstantOfShape",
    "If",
    "Pad",
    "Shape",
    "Size",
    "Relu",
    "Dropout",
    "Identity",
)

# node in this map will be replaced to other node.
replaced_node_map = {
    "Flatten": "Reshape",
    "Gemm": "Conv",
    "Squeeze": "Reshape",
    "Sum": "Add",
    "Unsqueeze": "Reshape",
}

# node in this map will be splited to multiple nodes.
split_node_map = {
    "Div": [
        "HzLut",
        "Mul",
    ],
    # matmul在一些特殊场景下会转成conv,约束应该和原始matmul相同,
    # 暂时关闭拆分映射的体现直接使用hbir.matmul的约束
    # "MatMul": [
    #     "MatMul",
    #     "Conv",
    # ],
    "InstanceNormalization": [
        "ReduceMean",
        "Sub",
        "Mul",
        "Add",
        "HzLut",
    ],
    "LSTM": [
        "HzLut",
        "Split",
        "Mul",
        "Concat",
        "Transpose",
        "Add",
        "Conv",
    ],
    "GRU": [
        "Split",
        "Transpose",
        "MatMul",
        "Mul",
        "HzLut",
        "Add",
        "Reshape",
        "Sub",
        "Concat",
    ],
    "GroupNormalization": [
        "Reshape",
        "ReduceMean",
        "Sub",
        "Mul",
        "Add",
        "HzLut",
    ],
    "LayerNormalization": [
        "ReduceMean",
        "GlobalAveragePool",
        "Sub",
    ],
    "ReduceL1": [
        "HzLut",
        "ReduceSum",
    ],
    "ReduceL2": [
        "HzLut",
        "ReduceSum",
        "HzLut",
    ],
    "Softmax": [
        "Sub",
        "HzLut",
        "ReduceSum",
        "ReduceMax",
        "HzLut",
        "Mul",
    ],
    "LogSoftmax": [
        "Sub",
        "HzLut",
        "ReduceSum",
        "ReduceMax",
        "HzLut",
        "Mul",
        "HzLut",
    ],
}


def get_special_node_constraint_mapping():
    """summary.

        此处用来记录一些特殊算子的约束,包括.
        1. 一对多被HMCT拆分的算子,需要手动书写一个原始算子的支持约束.

    返回的数据结构
        special_constraint_node_map = [
            {
                "op_name": {
                    "input":"特殊算子的手动计算约束说明字符串",
                    "output":"特殊算子的手动计算约束说明字符串"
                    }
            }
        ]
    """
    return [
        {
            "Div": {
                "input": "Type: int8, int16\n",
                "output": "If input is int8, output is int8\n\
                        If input is int16, output is int8/int16\n",
            },
            "InstanceNormalization": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "LSTM": {
                "input": "Dim: all dims < 2097152\n\
                                Type: int8, int16\n\
                                size < 2G\n",
                "output": "Same as input\n",
            },
            "GRU": {
                "input": "Dim: all dims < 2097152\n\
                                Type: int8, int16\n\
                                size < 2G\n",
                "output": "Same as input\n",
            },
            "GroupNormalization": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "LayerNormalization": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "ReduceL1": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "ReduceL2": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "Softmax": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
            "LogSoftmax": {
                "input": "Type: int8, int16\n\
                                Shape: [*]\n\
                                Dim: reduce axis dim size ∈ [1, 65535]\n\
                                Element : reduce Elements size ∈ [1, 65535]\n",
                "output": "Same as input\n",
            },
        }
    ]


def get_j6_hmct_node_mapping():
    """This func generate a list and is used by integration team.

        the structure of returned list like this:
        j6_hmct_node_mapping = [
            {
                "op_name": {
                    "op_list": []
                }
            }
        ]
    Note: Any change of the structure need to inform integration team.
    """
    j6_hmct_node_mapping = {}
    for node in lut_node_set:
        j6_hmct_node_mapping[node] = {"op_list": ["HzLut"]}
    for node in fused_node_set:
        j6_hmct_node_mapping[node] = {"op_list": ["None"]}
    j6_hmct_node_mapping.update(replaced_node_map)
    j6_hmct_node_mapping.update(split_node_map)
    return [j6_hmct_node_mapping]
