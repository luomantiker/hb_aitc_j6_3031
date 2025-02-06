from collections import defaultdict
import copy
from typing import TYPE_CHECKING, Union

from hmct.common import (
    Dataset,
    add_model_output,
    modify_flexible_batch,
)
from hmct.executor import ORTExecutor
from hmct.ir import OnnxModel, load_model

from .node_info import NodeInfo

if TYPE_CHECKING:
    from hmct.ir.onnx_utils import ModelProto


def has_calibration_node(node_info, node_name):
    calibration_nodes = node_info.get_calibration_by_node(node_name)
    return any(len(name.split("_HzCalibration")) > 1 for name in calibration_nodes)


def get_qtype_and_threshold(calibration_node):
    threshold, qtype = None, None
    for item in calibration_node.attribute:
        if item.name == "thresholds":
            threshold = round(max(list(item.floats)), 5)
        if item.name == "qtype":
            qtype = item.s.decode("utf-8")
    return threshold, qtype


class CalibrationModel:
    def __init__(
        self,
        model_or_file: Union[OnnxModel, "ModelProto", str],
        calibrated_data: Dataset = None,
    ) -> None:
        self.calibrated_model = load_model(model_or_file).proto
        self.node_info = NodeInfo(self.calibrated_model)
        self.calibrated_data = calibrated_data

        def get_node():
            sensitive_nodes = defaultdict(list)
            node_dict = {}
            conv_node_dict = {}
            for node in self.calibrated_model.graph.node:
                if has_calibration_node(
                    self.node_info,
                    node.name,
                ) and node.op_type not in ["HzCalibration", "HzFilter"]:
                    sensitive_nodes[node.name].extend(node.output)
                    node_dict[node.name] = node
                if node.op_type == "Conv":
                    conv_node_dict[node.name] = node
            return sensitive_nodes, conv_node_dict, node_dict

        # 模型中的普通节点,若能在其输入找到HzCalibration,则标记为sensitive nodes.
        self.sensitive_nodes, self.conv_node_dict, self.node_dict = get_node()

        def get_calibration_node():
            node_dict = {}
            for node in self.calibrated_model.graph.node:
                if node.op_type == "HzCalibration":
                    node_dict[node.name] = node
            return node_dict

        self.calibration_node = get_calibration_node()

        def calibration_node(node_type):
            cnode = {}
            if node_type in ["weight", "feature"]:

                def check_fn(x):
                    return x.tensor_type == node_type
            elif node_type == "tae":

                def check_fn(x):
                    return x.tensor_type == "feature" and any(
                        n.op_type in ["Conv", "MatMul", "Gemm", "ConvTranspose"]
                        for n in x.next_ops
                    )

            for node in OnnxModel(self.calibrated_model).graph.type2nodes[
                "HzCalibration"
            ]:
                if check_fn(node):
                    cnode[node.name] = {
                        "input": node.input_names[0],
                        "output": node.output_names[0],
                    }
            return cnode

        self.weight_calibration_nodes = calibration_node("weight")
        self.feature_calibration_nodes = calibration_node("feature")
        self.tae_calibration_nodes = calibration_node("tae")

        def get_calibration_nodes_message():
            nodes_input_map = defaultdict(str)
            qtype_and_threshold_map = {}
            for node in self.calibrated_model.graph.node:
                if node.op_type != "HzCalibration":
                    for i in node.input:
                        nodes_input_map[i] += node.name + ";"
                else:
                    threshold, qtype = get_qtype_and_threshold(node)
                    qtype_and_threshold_map[node.name] = [threshold, qtype]
            cali_and_normal_nodes = {}
            for k, v in self.weight_calibration_nodes.items():
                if v["output"] in nodes_input_map:
                    cali_and_normal_nodes[k] = nodes_input_map[v["output"]][:-1]
            for k, v in self.feature_calibration_nodes.items():
                if v["output"] in nodes_input_map:
                    cali_and_normal_nodes[k] = nodes_input_map[v["output"]][:-1]
            return cali_and_normal_nodes, qtype_and_threshold_map

        self.cali_normal_map, self.qtype_and_threshold_map = (
            get_calibration_nodes_message()
        )

        # original float model
        self.original_model = self.get_partial_qmodel_by_blacklist(
            blacklist_ops=["weight", "feature"],
        )

        def get_model_outputs():
            output_with_node_dict = {}
            for node in self.calibrated_model.graph.node:
                for o in node.output:
                    output_with_node_dict[o] = node
            temp_outputs = [
                output.name for output in self.calibrated_model.graph.output
            ]
            model_outputs = []
            for i in temp_outputs:
                if output_with_node_dict[i].op_type == "HzFilter":
                    model_outputs.extend(output_with_node_dict[i].input)
                else:
                    model_outputs.append(i)
            return model_outputs

        self.model_outputs = get_model_outputs()

    def get_calibration_by_node(self, nodes):
        calibration_nodes = []
        for node in nodes:
            calibration_nodes.extend(self.node_info.get_calibration_by_node(node))
        return calibration_nodes

    # Get a partially quantized model, the specified blacklist node
    # is not quantized.
    def get_partial_qmodel_by_blacklist(
        self,
        blacklist_nodes=None,
        blacklist_ops=None,
        bits=32,
    ):
        if blacklist_nodes is None:
            blacklist_nodes = []
        if blacklist_ops is None:
            blacklist_ops = []
        calibration_nodes = self.get_calibration_by_node(blacklist_nodes)
        model = OnnxModel(copy.deepcopy(self.calibrated_model))
        for node in model.graph.type2nodes["HzCalibration"]:
            if node.tensor_type in blacklist_ops or node.name in calibration_nodes:
                if bits == 32:
                    node.switch = "OFF"
                elif bits == 16:
                    node.qtype = "int16"
        return model.proto

    # Get a partially quantized model, only the specified whitelist nodes
    # are quantified.
    def get_partial_qmodel_by_whitelist(
        self, input_model=None, whitelist_nodes=None, whitelist_ops=None
    ):
        if whitelist_nodes is None:
            whitelist_nodes = []
        if whitelist_ops is None:
            whitelist_ops = []
        if isinstance(whitelist_nodes, str):
            whitelist_nodes = [whitelist_nodes]
        calibration_nodes = self.get_calibration_by_node(whitelist_nodes)
        if input_model is None:
            model = OnnxModel(copy.deepcopy(self.original_model))
        else:
            model = OnnxModel(copy.deepcopy(input_model))
        for node in model.graph.type2nodes["HzCalibration"]:
            if node.tensor_type in whitelist_ops or node.name in calibration_nodes:
                node.switch = "ON"
        return model.proto

    def get_batch_model(self, model, batch_size=1):
        if batch_size > 1:
            inference_model = modify_flexible_batch(OnnxModel(model)).proto
        else:
            inference_model = model
        return inference_model

    def get_model_output(
        self,
        model,
        model_output_list=None,
        batch_size=1,
        data_num=None,
    ):
        if model_output_list:
            model = add_model_output(
                OnnxModel(copy.deepcopy(model)),
                output_tensors=model_output_list,
            ).proto
        if data_num is None:
            num_of_data = self.calibrated_data.number_of_samples
        else:
            num_of_data = data_num
        executor = ORTExecutor(model).create_session()
        output_dict = executor.forward_with_batch(
            self.calibrated_data.copy(num_of_data),
            batch_size=batch_size,
            output_names=model_output_list,
        )
        return output_dict  # noqa: RET504
