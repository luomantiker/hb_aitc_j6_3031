import logging
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Set, Tuple, Union

from hmct.common import (
    Dataset,
    ModelDebugger,
    QuantConfig,
    modify_model_by_cpp_func,
)
from hmct.ir import OnnxModel, OnnxNode, OnnxVariable, save_model
from hmct.ir.horizon_onnx import quantizer

from .create_pipeline import create_calibration_pipeline, extract_cal_type

if TYPE_CHECKING:
    import numpy as np


def calibrate(
    optimized_model: OnnxModel,
    quant_config: QuantConfig,
    model_debugger: ModelDebugger,
    calibration_data: Optional[
        Union[Iterable["np.ndarray"], Mapping[str, Iterable["np.ndarray"]]]
    ] = None,
) -> Tuple[OnnxModel, str, Dataset]:
    """Calibrate onnx model for model convert.

    Returns:
        A tuple include calibrated onnx model after calibration,
        cal_type used to display in benchmark results and
        calibration_dataset used to calibrate model.
    """
    pre_calibrated_model = modify_model_by_cpp_func(
        optimized_model, quantizer.create_pre_calibrated_model
    )

    # parse calibration data
    if calibration_data is None:
        calibration_dataset = Dataset(
            input_shapes=pre_calibrated_model.graph.input_shapes,
            input_dtypes=pre_calibrated_model.graph.input_dtypes,
        )
        logging.info("No calibration data provided, using random data.")
    else:
        if not isinstance(calibration_data, Mapping):
            assert len(pre_calibrated_model.graph.inputs) == 1, (
                "The model should have single input if iterable "
                "object of ndarray given as input data."
            )
            calibration_data = {
                pre_calibrated_model.graph.input_names[0]: calibration_data
            }
        calibration_dataset = Dataset(input_data=calibration_data)
        logging.info(f"Provided calibration data md5: {calibration_dataset.md5}")

    # create pipeline for calibration
    calibration_pipeline = create_calibration_pipeline(
        quant_config=quant_config,
    )
    calibrated_model = calibration_pipeline.calibrate(
        pre_calibrated_model,
        calibration_dataset=calibration_dataset,
    )
    cal_type = extract_cal_type(calibration_pipeline)

    # dump all layers output for calibrated model
    if model_debugger.has_debug_method("dump_all_layers_output"):
        calibrated_model = dump_node_outputs(calibrated_model)
    if model_debugger.has_debug_method("dump_calibration_data"):
        calibration_dataset.save("./calibration_data")

    return calibrated_model, cal_type, calibration_dataset


def dump_node_outputs(onnx_model: OnnxModel) -> OnnxModel:
    """将Conv/MatMul类型节点的输出添加到模型输出.

    Args:
        onnx_model: 待添加输出的onnx模型

    Returns:
        添加输出后的onnx模型
    """
    dumped_op_types = ["Conv", "MatMul"]

    added_output_vars: Set[OnnxVariable] = set()
    type2nodes = onnx_model.graph.type2nodes
    for op_type in dumped_op_types:
        if op_type not in type2nodes:
            continue
        onnx_nodes = type2nodes[op_type]
        if op_type in ("Conv", "ConvTranspose"):
            # 将Conv/ConvTranspose类型节点输出添加到模型输出
            added_output_vars.update(
                find_conv_final_output_tensor(onnx_node) for onnx_node in onnx_nodes
            )
        else:
            # 将其他类型节点输出添加到模型输出
            added_output_vars.update(
                find_calibration_output_tensor(onnx_node) for onnx_node in onnx_nodes
            )

    # add collected output variables
    onnx_model.graph.extend_outputs(
        added_output_vars.difference(onnx_model.graph.outputs)
    )

    try:
        onnx_model.infer_shapes().check_validity()
    except Exception as exc:
        save_model(onnx_model, "dump_all_layers_output_fail.onnx")
        logging.error(
            "onnx model validation failed, invalid model "
            "saved as dump_all_layers_output_fail.onnx",
        )
        raise exc

    return onnx_model


def get_children_by_type(onnx_node: OnnxNode, op_type: str) -> List[OnnxNode]:
    return [next_op for next_op in onnx_node.next_ops if next_op.op_type == op_type]


def find_calibration_output_tensor(onnx_node: OnnxNode) -> OnnxVariable:
    calibration_children = get_children_by_type(onnx_node, "HzCalibration")
    if len(calibration_children) > 0:
        return calibration_children[0].outputs[0]
    return onnx_node.outputs[0]


def find_conv_final_output_tensor(onnx_node: OnnxNode) -> OnnxVariable:
    for op_type in ["Relu", "Clip"]:
        children = get_children_by_type(onnx_node, op_type)
        if len(children) > 0:
            return find_calibration_output_tensor(children[0])

    add_children = get_children_by_type(onnx_node, "Add")
    if len(add_children) > 0:
        if len(onnx_node.next_ops) == 1:
            relu_after_add = get_children_by_type(add_children[0], "Relu")
            if len(relu_after_add) == 0 or len(add_children[0].next_ops) > 1:
                return find_calibration_output_tensor(add_children[0])
            return find_calibration_output_tensor(relu_after_add[0])
        return find_calibration_output_tensor(onnx_node)

    return find_calibration_output_tensor(onnx_node)
