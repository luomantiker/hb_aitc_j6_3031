import copy
import logging
import operator
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from hmct.common import (
    Dataset,
    Loss,
    print_info_dict,
    set_model_switch,
    sort_info_dict,
)
from hmct.quantizer.debugger import get_sensitivity_of_nodes

from .base import (
    Calibrater,
    set_calibration_method,
    set_calibration_sensitivities,
)
from ..calibration_method import CalibrationMethod
from ..utils import calculate_models_similarities, convert_to_ptq_model

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class MixCalibrater(Calibrater):
    def __init__(self, layerwise_search: Dict[str, Any], **kwargs):
        """Initialization function for Mix Calibrater.

        Args:
            layerwise_search: parameters considered in mix calibrater are as follows.
                1. mix_topk: The top-k sensitivity nodes to consider during
                calibration.
                2. mix_metric: The metric used to calculate sensitivity nodes.
                3. use_int16: Whether to use int16 quantization for top-k
                sensitivity nodes.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.mix_topk = layerwise_search.get("topk")
        self.mix_metric = layerwise_search.get("metric", "cosine-similarity")
        # TODO(saiqiang.zhang): 支持敏感节点采用不同数据类型
        self.use_int16 = layerwise_search.get("qtype") == "int16"
        self.loss_func = Loss.create(self.mix_metric)
        self.cosine_threshold_for_topk = 0.999
        self.calibration_method = CalibrationMethod()
        self.calibration_method.set("max")
        self.calibration_method.set("max-percentile", percentile=0.99995)
        self.calibration_method.set("kl")
        self.qtype.set_method("mix")

    @property
    def name(self) -> str:
        return "mix_calibrater"

    def compare_calibrated_models(
        self,
        baseline_model: "OnnxModel",
        calibrated_models: "OnnxModel",
        input_data: Dict[str, np.ndarray],
    ) -> List[str]:
        """Compare calibrated models against the baseline model.

        Args:
            baseline_model: Float model as baseline to compare against.
            calibrated_models: Model with calibration thresholds under
                self.calibration_method recorded.
            input_data: Input data for model evaluation.

        Returns:
            List of method names with sensitivity value from high to low.
        """
        calibrated_models = [
            set_calibration_method(
                convert_to_ptq_model(copy.deepcopy(calibrated_models)),
                method,
            )
            for method in self.calibration_method
        ]
        similarities = calculate_models_similarities(
            baseline_model,
            calibrated_models,
            input_data,
            self.mix_metric,
        )
        return [
            method
            for method, _ in sorted(
                zip(map(str, self.calibration_method), similarities),
                key=operator.itemgetter(1),
                reverse=self.loss_func.optimal == np.argmax,
            )
        ]

    def set_topk_sensitivity_nodes(
        self,
        sensitivity_of_nodes: Dict[str, Dict[str, float]],
    ) -> None:
        """Sets the top-k sensitivity nodes based on cosine similarity.

        This method determines the top-k sensitivity nodes by sorting the nodes
        based on cosine similarity. If mix_topk is not provided, it selects
        nodes with cosine similarity less than a specified threshold.

        Args:
            sensitivity_of_nodes: Dictionary containing sensitivity
                                  information of calibration nodes.
        """
        if self.mix_topk is not None:
            self.topk_sensitivity_nodes = list(
                sort_info_dict(
                    sensitivity_of_nodes,
                    self.mix_metric,
                    self.loss_func.optimal == np.argmin,
                ).keys()
            )[: self.mix_topk]
        else:

            def left_bound(sensitivity_values, cosine_threshold=0.999):
                left, right = 0, len(sensitivity_values)
                while left < right:
                    mid = (left + right) // 2
                    if np.isnan(
                        float(sensitivity_values[mid]["cosine-similarity"])
                    ) or (
                        float(sensitivity_values[mid]["cosine-similarity"])
                        == cosine_threshold
                    ):
                        right = mid
                    elif (
                        float(sensitivity_values[mid]["cosine-similarity"])
                        < cosine_threshold
                    ):
                        left = mid + 1
                    elif (
                        float(sensitivity_values[mid]["cosine-similarity"])
                        > cosine_threshold
                    ):
                        right = mid
                return left

            sensitivity_of_nodes = sort_info_dict(
                sensitivity_of_nodes,
                "cosine-similarity",
                reverse=False,
            )
            topk = left_bound(
                list(sensitivity_of_nodes.values()),
                cosine_threshold=self.cosine_threshold_for_topk,
            )

            self.topk_sensitivity_nodes = list(sensitivity_of_nodes.keys())[:topk]

    def update_topk_sensitivity_nodes_qtype(
        self,
        calibrated_model: "OnnxModel",
    ) -> "OnnxModel":
        """Updates the qtype of top-k sensitivity nodes to int16.

        Args:
            calibrated_model: The calibrated model to update the qtype.

        Returns:
            The calibrated model with updated qtype for top-k
            sensitivity nodes.
        """
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            if (
                node.tensor_type == "feature"
                and node.name in self.topk_sensitivity_nodes
            ):
                node.qtype = "int16"
        return calibrated_model

    def merge_candidate_methods_threshold(
        self,
        calibrated_model: "OnnxModel",
        merge_info_verbose: bool = True,
    ) -> "OnnxModel":
        """Compare and select the best calibration methods for each activation.

        Compares activation sensitivities of the calibrated models using
        different calibration methods, selects the best method for each
        sensitivity node, and merges them to form a mix calibrated model.

        Args:
            calibrated_model: List of calibrated models with different
                calibration methods.
            merge_info_verbose: Whether or not to show merge information.

        Returns:
            The calibrated model with the best calibration method
            based on activation sensitivities.
        """
        if self.mix_topk is not None:
            logging.info(
                f"Topk {self.mix_metric} nodes has been set by the user "
                f"as {self.mix_topk}."
            )
        else:
            logging.info(
                f"Topk is not set by the user, select "
                f"top-{len(self.topk_sensitivity_nodes)} nodes "
                f"with cosine-similarity less than "
                f"{self.cosine_threshold_for_topk} to merge "
                f"multi calibration methods."
            )

        merge_info = {}
        node_mappings = calibrated_model.graph.node_mappings
        for node in self.topk_sensitivity_nodes:
            onnx_node = node_mappings[node]
            sensitivity_values = {
                method: sensitivity[onnx_node.qtype][self.mix_metric]
                for method, sensitivity in onnx_node.sensitivities.items()
            }
            best_method = list(sensitivity_values.keys())[
                self.loss_func.optimal(list(sensitivity_values.values()))
            ]
            onnx_node.thresholds = onnx_node.calibration_thresholds.get(
                best_method,
                onnx_node.thresholds,
            )
            onnx_node.constant = 1

            merge_info[node] = {
                method.split(":")[-1]: "{:.6f}(select)".format(
                    float(
                        onnx_node.sensitivities[method][onnx_node.qtype][
                            self.mix_metric
                        ]
                    )
                )
                if method == best_method
                else "{:.6f}".format(
                    float(
                        onnx_node.sensitivities[method][onnx_node.qtype][
                            self.mix_metric
                        ]
                    )
                )
                for method in sensitivity_values
            }

        if merge_info_verbose:
            print_info_dict(merge_info, title=f"Merge_Info({self.mix_metric})")

        return calibrated_model

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Run the calibration process using the mix method.

        Args:
            calibrated_model: The calibrated model to be used for calibration.
            calibration_dataset: The calibration dataset to be used for
                calibration.
            **kwargs: Other unused parameters.

        Returns:
            OnnxModel: The calibrated model with thresholds set.
        """
        logging.info("Run calibration model with layerwise search method.")
        # Step1: 尝试基础校准方法,并对校准方法进行排序。
        calibration_model = self._calibrate(
            calibrated_model,
            self.calibration_method,
            calibration_dataset,
        )
        methods_sorted = self.compare_calibrated_models(
            set_model_switch(copy.deepcopy(calibrated_model), "OFF"),
            calibration_model,
            calibration_dataset.get_data(),
        )

        # Step2: 以Step1中最优校准方法为基准, 计算节点相似度,
        # 确定需要阈值混合的校准节点。
        logging.info(
            f"Calculate all sensitivity nodes with {methods_sorted[0]} "
            f"calibration method using {self.mix_metric} metric."
        )
        sensitivity_of_nodes = get_sensitivity_of_nodes(
            model_or_file=convert_to_ptq_model(
                set_calibration_method(
                    copy.deepcopy(calibration_model),
                    methods_sorted[0],
                )
            ).proto,
            metrics=[self.mix_metric, "cosine-similarity"]
            if self.mix_metric != "cosine-similarity"
            else self.mix_metric,
            calibrated_data=calibration_dataset,
            output_node=None,
            node_type="activation",
            data_num=1,
            verbose=False,
        )
        calibration_model = set_calibration_sensitivities(
            calibration_model,
            methods_sorted[0],
            sensitivity_of_nodes,
        )
        self.set_topk_sensitivity_nodes(sensitivity_of_nodes)
        if self.use_int16:
            calibration_model = self.update_topk_sensitivity_nodes_qtype(
                calibration_model
            )

        # Step3: 计算其他校准方法下的节点相似度。
        for method in methods_sorted if self.use_int16 else methods_sorted[1:]:
            logging.info(
                f"Calculate top{len(self.topk_sensitivity_nodes)} "
                f"sensitivity nodes with {method} calibration method "
                f"using {self.mix_metric} metric."
            )
            sensitivity_of_nodes = get_sensitivity_of_nodes(
                model_or_file=convert_to_ptq_model(
                    set_calibration_method(copy.deepcopy(calibration_model), method)
                ).proto,
                metrics=self.mix_metric,
                calibrated_data=calibration_dataset,
                output_node=None,
                node_type="activation",
                data_num=1,
                verbose=False,
                interested_nodes=self.topk_sensitivity_nodes,
            )
            calibration_model = set_calibration_sensitivities(
                calibration_model,
                method,
                sensitivity_of_nodes,
            )

        # Step4: 针对需要阈值混合的校准节点,按照相似度最优原则进行分配。
        return self.merge_candidate_methods_threshold(
            set_calibration_method(calibration_model, methods_sorted[0])
        )
