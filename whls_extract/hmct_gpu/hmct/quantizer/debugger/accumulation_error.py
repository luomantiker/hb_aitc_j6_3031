import functools
import logging
import operator
import os
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmct.common import Dataset, Loss

from .debugger import CalibrationModel
from .util import set_calibration_data

if TYPE_CHECKING:
    from hmct.ir.onnx_utils import ModelProto

mpl.use("Agg")


def get_average_of_acc_error(data_list):
    return [sum(data_list[:i]) / i for i in range(1, len(data_list) + 1)]


def is_nested_list(params):
    if len(params) == 0:
        return False
    return any(isinstance(i, list) for i in params)


class AccumulationError:
    def __init__(
        self,
        save_dir: str,
        model_or_file: Union["ModelProto", str],
        calibrated_data: Optional[Union[str, Dataset]] = None,
    ) -> None:
        self.calibrated_data = set_calibration_data(calibrated_data)

        self.cali_model_attr = CalibrationModel(model_or_file, self.calibrated_data)
        self.weight_calibration_nodes = self.cali_model_attr.weight_calibration_nodes
        self.feature_calibration_nodes = self.cali_model_attr.feature_calibration_nodes
        self.sensitive_nodes = self.cali_model_attr.sensitive_nodes
        self.original_model = self.cali_model_attr.original_model
        self.calibrated_model = self.cali_model_attr.calibrated_model
        self.conv_node_dict = self.cali_model_attr.conv_node_dict

        self.save_dir = save_dir

    def get_weight_cali_nodes_name(self) -> List[str]:
        node_name_list = []
        node_name_list.extend(self.weight_calibration_nodes.keys())
        return node_name_list

    def get_feature_cali_nodes_name(self) -> List[str]:
        node_name_list = []
        node_name_list.extend(self.feature_calibration_nodes.keys())
        return node_name_list

    def accumulate_error_analysis(
        self,
        qmodels: Optional[Dict[str, "ModelProto"]] = None,
        metric: str = "cosine-similarity",
        file_prefix: str = "",
        average_mode: bool = False,
    ) -> None:
        output_names = functools.reduce(
            operator.iadd, list(self.sensitive_nodes.values()), []
        )
        float_output_dict = self.cali_model_attr.get_model_output(
            self.original_model,
            output_names,
            1,
            1,
        )

        if qmodels is None:
            qmodels = {"normal": self.calibrated_model}
        error_data = {}
        for name in tqdm(list(qmodels.keys()), desc="Testing"):
            quanti_output_dict = self.cali_model_attr.get_model_output(
                qmodels[name],
                output_names,
                1,
                1,
            )
            loss_res = []
            for output_name in output_names:
                out1 = float_output_dict[output_name][0]
                out2 = quanti_output_dict[output_name][0]
                if out1.dtype == bool and out2.dtype == bool:
                    out1 = out1.astype("f")
                    out2 = out2.astype("f")
                loss_res.append(float(Loss.create(metric).run(out1, out2)))
            error_data[name] = {metric: loss_res}
        self.generate_accumulate_error_plt(
            error_data,
            metric,
            file_prefix,
            average_mode,
        )

    def generate_accumulate_error_plt(
        self,
        error_data: Dict[str, Dict[str, List[str]]],
        metric: str,
        file_prefix: str = "",
        average_mode=False,
    ) -> None:
        def random_color():
            color_arr = [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
            ]
            color = ""
            for _ in range(6):
                color += color_arr[random.randint(0, 14)]
            return "#" + color

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if average_mode:
            file_name = f"{self.save_dir}/average_{file_prefix}_{metric}.png"
        else:
            file_name = f"{self.save_dir}/{file_prefix}_{metric}.png"
        plt.clf()
        plt.figure(figsize=(8, 6))
        org_error_data = []
        for name in error_data:
            color_str = random_color()
            inf = float("inf")
            for _ in range(error_data[name][metric].count(inf)):
                error_data[name][metric].remove(inf)
            org_error_data = error_data[name][metric]
            if average_mode:
                org_error_data = get_average_of_acc_error(org_error_data)
            plt.plot(org_error_data, label=name, color=color_str)
        plt.title(file_prefix)
        plt.ylabel(metric)
        plt.xlabel("node index")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(file_name, bbox_inches="tight")

    def plot_acc_error(
        self,
        quantize_node: Optional[Union[str, List[str]]] = None,
        non_quantize_node: Optional[Union[str, List[str]]] = None,
        metric: str = "cosine-similarity",
        average_mode: bool = False,
    ) -> None:
        def flatten(nested_list):
            return (
                functools.reduce(operator.iadd, map(flatten, nested_list), [])
                if isinstance(nested_list, list)
                else [nested_list]
            )

        flatten_quantize_node = flatten(quantize_node)
        flatten_non_quantize_node = flatten(non_quantize_node)

        if (quantize_node is None and non_quantize_node is None) or (
            len(flatten_quantize_node) == 0 and len(flatten_non_quantize_node) == 0
        ):
            raise ValueError("'quantize_node' and 'non_quantize_node' are both NONE.")

        if quantize_node is None and non_quantize_node is not None:
            quantize_node = []
        elif quantize_node is not None and non_quantize_node is None:
            non_quantize_node = []

        if isinstance(quantize_node, str):
            quantize_node = [quantize_node]
        if isinstance(non_quantize_node, str):
            non_quantize_node = [non_quantize_node]

        if not is_nested_list(quantize_node):
            quantize_node = list(set(quantize_node))
        if not is_nested_list(non_quantize_node):
            non_quantize_node = list(set(non_quantize_node))

        # quantize weight or activation
        if (
            "weight" in flatten_non_quantize_node
            or "activation" in flatten_non_quantize_node
        ):
            raise ValueError(
                "Cannot set 'non_quantize_node' to 'weight' or 'activation'.",
            )
        if (
            "weight" in flatten_quantize_node
            and "activation" not in flatten_quantize_node
        ):
            logging.info("Only quantizing weights without quantizing activation !")
            file_prefix = "weight_quantized"
            qmodels = {}
            qmodels["weight_activation_calibrated"] = self.calibrated_model
            qmodels["only_weight_calibrated"] = (
                self.cali_model_attr.get_partial_qmodel_by_whitelist(
                    whitelist_ops=["weight"],
                )
            )
            self.accumulate_error_analysis(
                qmodels,
                metric,
                file_prefix=file_prefix,
                average_mode=average_mode,
            )
        elif (
            "weight" not in flatten_quantize_node
            and "activation" in flatten_quantize_node
        ):
            logging.info("Only quantizing activation without quantizing weight !")
            file_prefix = "activation_quantized"
            qmodels = {}
            qmodels["weight_activation_calibrated"] = self.calibrated_model
            qmodels["only_activation_calibrated"] = (
                self.cali_model_attr.get_partial_qmodel_by_whitelist(
                    whitelist_ops=["feature"],
                )
            )
            self.accumulate_error_analysis(
                qmodels,
                metric,
                file_prefix=file_prefix,
                average_mode=average_mode,
            )
        elif (
            "weight" in flatten_quantize_node and "activation" in flatten_quantize_node
        ):
            logging.info("Quantify weight and activation separately!")
            file_prefix = "weight_activation_quantized_separately"
            qmodels = {}
            qmodels["weight_activation_calibrated"] = self.calibrated_model
            qmodels["only_activation_calibrated"] = (
                self.cali_model_attr.get_partial_qmodel_by_whitelist(
                    whitelist_ops=["feature"],
                )
            )
            qmodels["only_weight_calibrated"] = (
                self.cali_model_attr.get_partial_qmodel_by_whitelist(
                    whitelist_ops=["weight"],
                )
            )
            self.accumulate_error_analysis(
                qmodels,
                metric,
                file_prefix=file_prefix,
                average_mode=average_mode,
            )

        if (
            len(quantize_node) > 0
            and "weight" not in flatten_quantize_node
            and "activation" not in flatten_quantize_node
        ):
            logging.info(
                "Only the nodes you set are quantized, and "
                "the remaining nodes are not quantized.",
            )
            # aeopq
            qmodels = {}
            if is_nested_list(quantize_node):
                file_prefix = "quantize_node_accumulate_err_of_partial_qmodel"
                qmodels["default calibrated model"] = self.calibrated_model
                for i in range(len(quantize_node)):
                    model_name = "partial_qmodel_" + str(i)
                    qmodels[model_name] = (
                        self.cali_model_attr.get_partial_qmodel_by_whitelist(
                            whitelist_nodes=quantize_node[i],
                        )
                    )
            # aeon
            else:
                file_prefix = "quantize_node_accumulate_err_of_node"
                for node_name in quantize_node:
                    qmodels[node_name] = (
                        self.cali_model_attr.get_partial_qmodel_by_whitelist(
                            whitelist_nodes=node_name,
                        )
                    )
            self.accumulate_error_analysis(
                qmodels,
                metric,
                file_prefix=file_prefix,
                average_mode=average_mode,
            )

        if len(non_quantize_node) > 0:
            logging.info(
                "Do not quantify the nodes you set, "
                "and quantify the rest of the nodes.",
            )
            # aeopq
            qmodels = {}
            if is_nested_list(non_quantize_node):
                file_prefix = "non_quantize_node_accumulate_err_of_partial_qmodel"
                qmodels["default calibrated model"] = self.calibrated_model
                for i in range(len(non_quantize_node)):
                    model_name = "partial_qmodel_" + str(i)
                    qmodels[model_name] = (
                        self.cali_model_attr.get_partial_qmodel_by_blacklist(
                            blacklist_nodes=non_quantize_node[i],
                        )
                    )
            # aeon
            else:
                file_prefix = "non_quantize_node_accumulate_err_of_node"
                for node_name in non_quantize_node:
                    qmodels[node_name] = (
                        self.cali_model_attr.get_partial_qmodel_by_blacklist(
                            blacklist_nodes=node_name,
                        )
                    )

            self.accumulate_error_analysis(
                qmodels,
                metric,
                file_prefix=file_prefix,
                average_mode=average_mode,
            )

    def plot_conv_acc_error(
        self,
        qmodels: Optional[Dict[str, "ModelProto"]] = None,
        file_prefix: str = "",
    ) -> None:
        conv_nodes_dict = {}
        for node_name, node in self.conv_node_dict.items():
            conv_nodes_dict[node_name] = node.output[0]
        fdata = self.cali_model_attr.get_model_output(
            self.original_model,
            list(conv_nodes_dict.values()),
            1,
            1,
        )

        if qmodels is None:
            qmodels = {"normal": self.calibrated_model}
        plt.clf()
        plt.figure(figsize=(8, 6))
        for name in tqdm(list(qmodels.keys()), desc="Testing"):
            cdata = self.cali_model_attr.get_model_output(
                qmodels[name],
                list(conv_nodes_dict.values()),
                1,
                1,
            )
            error_data = []
            for conv_output in conv_nodes_dict.values():
                error_array = (cdata[conv_output][0] - fdata[conv_output][0]).flatten()
                error_avg = sum(error_array) / len(error_array)
                error_data.append(error_avg)
            plt.plot(error_data, label=name)
        file_name = f"{self.save_dir}/{file_prefix}_mean-diff-error.png"
        plt.title(file_prefix)
        plt.ylabel("mean-diff-error")
        plt.xlabel("conv index")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(file_name, bbox_inches="tight")
        logging.info(f"Result has been saved as {file_name}")
