from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def convert_reshape_target_shape_to_positive(model: "OnnxModel") -> "OnnxModel":
    """如果Reshape算子的target shape中有-1, 0, 编译器不支持, 需要修改为对应的正值."""
    for reshape_node in model.graph.type2nodes["Reshape"]:
        if not reshape_node.outputs[0].is_shape_dynamic:
            output_shape = reshape_node.outputs[0].shape
            if len(reshape_node.inputs[1].dest_ops) > 1:
                target_shape = model.graph.create_variable(
                    name=reshape_node.name + "_" + reshape_node.inputs[1].name,
                    is_param=True,
                    is_attr=False,
                    value=np.array(output_shape),
                )
                reshape_node.replace_input(1, target_shape)
            else:
                reshape_node.inputs[1].value = np.array(output_shape)
    return model
