from collections import defaultdict
import copy
import functools
import logging
import operator
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from hmct.common import Dataset, Loss, add_model_output
from hmct.ir import OnnxModel

from .debugger import CalibrationModel
from .util import set_calibration_data

if TYPE_CHECKING:
    from hmct.ir.onnx_utils import ModelProto


class NodeSensitivity:
    def __init__(
        self,
        model_or_file: Union[OnnxModel, "ModelProto", str],
        calibrated_data: Optional[Union[str, Dataset]] = None,
    ) -> None:
        self.calibrated_data = set_calibration_data(calibrated_data)

        self.cali_model_attr = CalibrationModel(model_or_file, self.calibrated_data)
        self.original_model = self.cali_model_attr.original_model
        self.sensitive_nodes = self.cali_model_attr.sensitive_nodes
        self.weight_calibration_nodes = self.cali_model_attr.weight_calibration_nodes
        self.feature_calibration_nodes = self.cali_model_attr.feature_calibration_nodes
        self.tae_calibration_nodes = self.cali_model_attr.tae_calibration_nodes
        self.node_info = self.cali_model_attr.node_info
        self.cali_normal_map = self.cali_model_attr.cali_normal_map
        self.qtype_and_threshold_map = self.cali_model_attr.qtype_and_threshold_map

    def sensitivity(
        self,
        original_output_dict: Dict[str, List],
        output_dict: Dict[str, List],
        metric_name: str,
    ) -> str:
        loss_function = Loss.create(metric_name)

        list_out = []  # Used to save output data of per output in model
        for output_name in output_dict:
            out1 = original_output_dict[output_name]
            out2 = output_dict[output_name]
            assert len(out1) == len(out2)
            value_temp = 0
            list_in = []  # Used to save output data of per data for one output
            for i in range(len(out1)):
                loss_value = loss_function.run(out1[i], out2[i])
                if not np.isnan(loss_value):
                    list_in.append(loss_value)
            if len(list_in) > 0:
                value_temp = np.mean(list_in)
                list_out.append(value_temp)
            else:
                continue
        return np.mean(list_out) if len(list_out) > 0 else np.float32("nan")

    def sensitivity_of_nodes(
        self,
        node_type: str = "node",
        batch_size: int = 1,
        data_num: Optional[int] = None,
        metrics: Union[List[str], str] = "cosine-similarity",
        output_node: Optional[Union[str, List[str]]] = None,
        interested_nodes: Optional[Union[str, List[str], List[List[str]]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        # Calculate the sensitivity of each node.
        node_info = defaultdict(dict)
        if node_type not in ["node", "weight", "activation", "tae"]:
            raise ValueError(f"node_type:{node_type} is unavailable.")
        if node_type == "node":
            node_name_dict = self.sensitive_nodes
        elif node_type == "weight":
            node_name_dict = self.weight_calibration_nodes
        elif node_type == "activation":
            node_name_dict = self.feature_calibration_nodes
        elif node_type == "tae":
            node_name_dict = self.tae_calibration_nodes
        node_dict2list = list(node_name_dict.keys())
        if interested_nodes is not None:
            logging.info(
                "Interested_nodes is given, only calculate sensitivity "
                "for these nodes, no matter what node_type is specified.",
            )
            node_dict2list = (
                interested_nodes
                if isinstance(interested_nodes, List)
                else [interested_nodes]
            )

        if output_node is None:
            model_output_list = [self.cali_model_attr.model_outputs]
        else:

            def find_output(model: OnnxModel, output_node: str) -> Tuple[str]:
                if output_node not in model.graph.node_names:
                    raise ValueError(
                        f"The output_node({output_node}) is not model node "
                        "name, please check output_node params."
                    )
                return model.graph.node_mappings[output_node].output_names

            cali_model = OnnxModel(self.cali_model_attr.calibrated_model)
            model_output_list = (
                [find_output(cali_model, output_node)]
                if isinstance(output_node, str)
                else [find_output(cali_model, _) for _ in output_node]
            )
        model_output_list = functools.reduce(operator.iadd, model_output_list, [])
        is_batch = True
        org_model = self.cali_model_attr.get_batch_model(
            self.original_model,
            batch_size,
        )
        try:
            original_output_dict = self.cali_model_attr.get_model_output(
                org_model,
                model_output_list,
                batch_size,
                data_num,
            )
        except Exception:
            logging.info(
                "The model does not support using batch inference, "
                "reset BATCH_SIZE=1 and try again.",
            )
            batch_size = 1
            is_batch = False
            original_output_dict = self.cali_model_attr.get_model_output(
                self.original_model,
                model_output_list,
                batch_size,
                data_num,
            )

        ori_test_model = copy.deepcopy(self.original_model)
        if is_batch:
            ori_test_model = self.cali_model_attr.get_batch_model(
                ori_test_model,
                batch_size,
            )
        ori_test_model = add_model_output(
            OnnxModel(ori_test_model),
            output_tensors=model_output_list,
            keep_original_output=False,
        ).proto
        for nodes in tqdm(node_dict2list, desc="Progress"):
            test_model = self.cali_model_attr.get_partial_qmodel_by_whitelist(
                ori_test_model, nodes
            )

            output_dict = self.cali_model_attr.get_model_output(
                test_model,
                None,
                batch_size,
                data_num,
            )
            nodes_name = "+".join(nodes) if isinstance(nodes, List) else nodes
            if node_type != "node":
                node_info[nodes_name]["node"] = self.cali_normal_map.get(nodes_name, "")
            if node_type == "activation":
                node_info[nodes_name]["threshold"] = self.qtype_and_threshold_map[
                    nodes_name
                ][0]
                node_info[nodes_name]["qtype"] = self.qtype_and_threshold_map[
                    nodes_name
                ][1]
            for metric in metrics:
                metric = metric.lower()
                node_info[nodes_name][metric] = self.sensitivity(
                    original_output_dict,
                    output_dict,
                    metric,
                )

        self.node_info.update(node_type, node_info)
        return self.node_info
