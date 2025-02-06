import logging
from typing import Any, Dict, Union

import numpy as np

from hmct.common import ColorConvert, modify_model_by_cpp_func
from hmct.ir import DataType, OnnxModel, save_model
from hmct.ir.horizon_onnx import quantizer


def add_preprocess_node(
    onnx_model: OnnxModel,
    input_dict_parser: Dict[str, Dict[str, Any]],
) -> OnnxModel:
    # add preprocess_node
    preprocess_dict = input_dict_parser.get_preprocess_info()
    input_shapes = input_dict_parser.get_input_shapes()
    for input_name, preprocess_info in preprocess_dict.items():
        if input_name in input_shapes and len(input_shapes[input_name]) != 4:
            raise ValueError(
                f"Only support to add preprocess node when the rank "
                f"of input is 4. But the rank of '{input_name}' is "
                f"{len(input_shapes[input_name])}, Please check your yaml.",
            )
        input_var = onnx_model.graph.input_mappings[input_name]
        qtype = "int8"
        if (
            len(input_var.dest_ops) == 1
            and input_var.dest_op.op_type in ["HzQuantize", "QuantizeLinear"]
            and input_var.dest_op.outputs[0].dtype == DataType.INT16
        ):
            qtype = "int16"
        if qtype == "int16":
            raise ValueError(
                "Not support to add preprocess node when the calibration_type "
                "is load and the preprocess node is quantized to int16, "
                "Please check your yaml.",
            )
        onnx_model = add_preprocess_node_at_input(
            onnx_model=onnx_model,
            input_name=input_name,
            **preprocess_info,
        )
    return onnx_model


def add_preprocess_node_at_input(
    onnx_model: OnnxModel,
    input_name: str,
    from_color: str = "BGR",
    to_color: str = "BGR",
    mean: Union[np.ndarray, None] = None,
    scale: Union[np.ndarray, None] = None,
    from_color_input_range: Union[str, int] = "255",
    to_color_input_range: Union[str, int] = "128",
    input_layout_train: str = "NCHW",
    color_convert_enum: Union[str, None] = None,
) -> OnnxModel:
    """Add preprocess node to target input of model.

    Args:
        onnx_model: onnx model to add preprocess node.
        input_name: Target input name that preprocess node will be added to.
        from_color (optional): Color type that the model was trained with.
            Choose from the list: ["YUV_BT601_FULL_RANGE", "FEATUREMAP",
            "BGR", "RGB", "GRAY"].
        to_color (optional): Color type that the model will be given.
            Choose from the list: ["YUV_BT601_VIDEO_RANGE",
            "YUV_BT601_FULL_RANGE", "FEATUREMAP", "BGR", "RGB", "GRAY"].
        mean (optional): Mean value for the normalization of input data.
            The length of the mean value can be either 1 or the same
            length as the input channel.
        scale (optional): Scale value for the normalization of input data.
            The length of the scale value can be either 1 or the same
            length as the input channel.
        from_color_input_range (optional): The input range for
            floating-point input.
        to_color_input_range (optional): The input range for
            fixed-point input.
        input_layout_train (optional): The layout of the original model that
            was trained with. Choose from the list: ["NHWC", "NCHW"].
        color_convert_enum (optional): The color convert combination.
            Choose from the ColorConvert enum. This option should not
            conflict with the from_color and to_color options. If this
            option is given, the from_color and to_color will be ignored.

    Returns:
        Onnx model with preprocess node added.
    """
    if color_convert_enum is None:
        # get color convert info from from_color & to_color
        # check if color convert is valid
        ColorConvert.get_convert_type(from_color, to_color)
    else:
        if from_color is not None or to_color is not None:
            logging.warning(
                f"color_convert_enum is given: {color_convert_enum}, "
                f"the from_color and to_color option will be ignored.",
            )
        # get color info from enum
        from_color, to_color = ColorConvert.split_color_convert(color_convert_enum)

    input_channel = None
    for input_var in onnx_model.graph.inputs:
        if input_var.name == input_name:
            idx_c = 1 if input_layout_train == "NCHW" else 3
            input_channel = input_var.shape[idx_c]

    if input_channel is None:
        raise ValueError(
            f"The specified input name: {input_name} is not an input of the model.",
        )

    expected_preprocess_name = "HZ_PREPROCESS_FOR_" + input_name
    for node in onnx_model.graph.nodes:
        if node.name == expected_preprocess_name:
            raise ValueError(
                f"Preprocess node for {input_name} already exists, duplicated"
                " insertion is forbidden.",
            )

    if isinstance(mean, np.ndarray):
        if mean.ndim == 0 and mean.size == 1:
            mean = mean.reshape(1)

        if len(mean) == input_channel:
            mean = list(mean)
        elif len(mean) == 1:
            mean = [mean[0]] * input_channel
        else:
            raise ValueError("The size of mean does not match input channel.")
    elif mean is None:
        mean = []
    else:
        raise ValueError("The parameter mean is of the wrong type.")

    if isinstance(scale, np.ndarray):
        if scale.ndim == 0 and scale.size == 1:
            scale = scale.reshape(1)

        if len(scale) == input_channel:
            scale = list(scale)
        elif len(scale) == 1:
            scale = [scale[0]] * input_channel
        else:
            raise ValueError("The size of scale does not match input channel.")
    elif scale is None:
        scale = [1] * input_channel
    else:
        raise ValueError("The parameter scale is of the wrong type.")

    if (from_color == "GRAY" and input_channel != 1) or (
        from_color in ["RGB", "BGR"] and input_channel != 3
    ):
        raise ValueError(
            f"Float input type {from_color} does not match "
            f"model input channel {input_channel}.",
        )

    float_input_info = {
        "input_format": from_color,
        "input_range": str(from_color_input_range),
        "input_layout": input_layout_train,
    }
    fixed_input_info = {
        "input_format": to_color,
        "input_range": str(to_color_input_range),
    }

    modified_model = modify_model_by_cpp_func(
        onnx_model,
        quantizer.insert_mean_scale_node,
        input_name,
        mean,
        scale,
        float_input_info,
        fixed_input_info,
    )

    try:
        modified_model.infer_shapes().check_validity()
    except Exception as e:
        save_model(modified_model, "add_preprocess_fail.onnx")
        logging.error(
            "onnx model validation failed, invalid model saved as "
            "add_preprocess_fail.onnx.",
        )
        raise e

    return modified_model
