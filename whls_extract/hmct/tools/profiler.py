# 模型分析工具, 输入一个onnx模型, 输出结构化信息。
import argparse
from collections import defaultdict
from enum import Enum
import logging
import sys

import numpy as np

from hmct.common import print_info_dict
from hmct.ir import OnnxModel, OnnxNode, load_model
from hmct.reporter import calculate_quant_type

TAE_OPS = ["Conv", "MatMul", "Gemm", "ConvTranspose"]
NODE_INPUTS = {
    "Conv": 3,
    "ConvTranspose": 3,
    "Gemm": 3,
    "MatMul": 2,
    "Add": 2,
    "Sub": 2,
    "Mul": 2,
    "Div": 2,
    "Concat": 3,
}


class Conv:
    def __init__(self, node):
        self.input_shape = node.inputs[0].shape
        self.weight_shape = node.inputs[1].shape
        self.output_shape = node.outputs[0].shape
        self.has_bias = False
        if len(node.inputs) == 3:
            self.has_bias = True

    def ops(self):
        try:
            out_size = np.prod(self.output_shape)
            if len(self.weight_shape) > 3:
                ops = (
                    2
                    * out_size
                    * self.weight_shape[1]
                    * self.weight_shape[2]
                    * self.weight_shape[3]
                )
            elif len(self.weight_shape) == 3:
                ops = 2 * out_size * self.weight_shape[1] * self.weight_shape[2]
            if self.has_bias:
                ops += out_size
            return ops
        except Exception:
            return 0


class GemmAndMatMul:
    def __init__(self, node):
        self.input1_shape = node.inputs[0].shape
        self.input2_shape = node.inputs[1].shape
        self.output_shape = node.outputs[0].shape
        self.has_bias = False
        if len(node.inputs) == 3:
            self.has_bias = True
        self.transB = node.attributes.get("transB", False)

    def ops(self):
        try:
            ops = np.prod(self.input1_shape)
            if self.transB == 0 or self.transB is None:
                ops *= self.input2_shape[-1]
            else:
                ops *= self.input2_shape[0]
            ops *= 2
            if self.has_bias:
                ops += np.prod(self.output_shape)
            return ops
        except Exception:
            return 0


class ModelType(Enum):
    FLOAT_MODEL = 0
    CALIBRATED_MODEL = 1


class ModelProfiler:
    def __init__(self, onnx_model: OnnxModel) -> None:
        self.onnx_model = onnx_model
        self.model_type = self.get_model_type()
        self.node_quant_types = {}
        self.node_info = defaultdict(dict)

    def get_model_type(self) -> ModelType:
        for node in self.onnx_model.graph.nodes:
            if node.op_type == "HzCalibration":
                return ModelType.CALIBRATED_MODEL
        return ModelType.FLOAT_MODEL

    def input_shape(self, node: OnnxNode, idx: int) -> str:
        if idx < len(node.inputs) and node.inputs[idx].shape:
            if any(isinstance(s, str) for s in node.inputs[idx].shape):
                return ""
            shape = "x".join([str(s) for s in node.inputs[idx].shape])
            if node.inputs[idx].is_param:
                shape = f"{shape}(P)"
            elif node.inputs[idx].src_op:
                src_op = node.inputs[idx].src_op
                if src_op.op_type == "HzCalibration" and src_op.inputs[0].is_param:
                    shape = f"{shape}(P)"
            return shape

        return ""

    def params(self, node: OnnxNode, num_inputs: int) -> int:
        param_size = 0
        for i in range(min(num_inputs, len(node.inputs))):
            if node.inputs[i].is_param:
                param_size += np.prod(node.inputs[i].shape)
            elif node.inputs[i].src_op:
                src_op = node.inputs[i].src_op
                if src_op.op_type == "HzCalibration" and src_op.inputs[0].is_param:
                    param_size += np.prod(node.inputs[i].shape)
        return param_size

    def ops(self, node: OnnxNode) -> int:
        num_of_ops = 0
        if node.op_type == "Conv":
            num_of_ops = Conv(node).ops()
        if node.op_type == "MatMul" or node.op_type == "Gemm":
            num_of_ops = GemmAndMatMul(node).ops()
        return num_of_ops

    def profile(self, parser) -> None:
        if self.model_type == ModelType.CALIBRATED_MODEL:
            # Get the quantization type of all nodes.
            self.node_quant_types = calculate_quant_type(self.onnx_model)

        for node in self.onnx_model.graph.nodes:
            name = node.name
            op_type = node.op_type
            if op_type == "HzCalibration":
                continue
            if parser["tae"] and op_type not in TAE_OPS:
                continue

            # The operator type of each node.
            self.node_info[name]["Type"] = op_type

            # The quant type of each node.
            if (
                self.model_type == ModelType.CALIBRATED_MODEL
                and name in self.node_quant_types
            ):
                self.node_info[name]["DataType"] = self.node_quant_types[name][
                    "input_qtype"
                ][0]

            # The input of each node.
            self.node_info[name]["Input0"] = self.input_shape(node, 0)
            if op_type in NODE_INPUTS:
                for idx in range(1, NODE_INPUTS[op_type]):
                    self.node_info[name][f"Input{idx}"] = self.input_shape(node, idx)

            # The params of each node.
            param_size = self.params(node, num_inputs=NODE_INPUTS.get(op_type, 1))
            if param_size > 0:
                self.node_info[name]["Params"] = param_size

            # The ops of each TAE node.
            num_of_ops = self.ops(node)
            if num_of_ops > 0:
                self.node_info[name]["OPs"] = num_of_ops

        print_info_dict(self.node_info, title="Model Profile")

    def summary(self) -> None:
        # Summary Information.
        total_ops = 0
        total_params = 0
        ops_for_dtype = defaultdict(float)
        summary_info = defaultdict(dict)
        for values in self.node_info.values():
            if "OPs" in values:
                if "DataType" in values:
                    ops_for_dtype[values["DataType"]] += values["OPs"]
                total_ops += values["OPs"]
            if "Params" in values:
                total_params += values["Params"]

        for k, v in ops_for_dtype.items():
            summary_info[k] = {
                "Params": "--",
                "OPs": str(round(v * 1e-9, 2)) + "(G)",
            }

        summary_info["Total"] = {
            "Params": str(round(total_params * 1e-6, 2)) + "(M)",
            "OPs": str(round(total_ops * 1e-9, 2)) + "(G)",
        }
        print_info_dict(summary_info, title="Model Summary", key_type="DataType")
        return summary_info


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model",
        "-om",
        type=str,
        required=True,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--tae",
        default=False,
        action="store_true",
        help="Only profile TAE operators.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="When the input is a calibrated model, you can specify a data to "
        "calculate the similarity between the calibrated model and "
        "the floating-point model.",
    )

    return parser.parse_args(args)


def main(args):
    parser = parse_args(args)
    onnx_model = load_model(parser.onnx_model)
    profiler = ModelProfiler(onnx_model)
    profiler.profile(vars(parser))
    profiler.summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
