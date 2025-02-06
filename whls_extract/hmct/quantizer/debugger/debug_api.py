from collections import defaultdict
import copy
import functools
import logging
import operator
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from hmct.common import Dataset, Loss, print_info_dict, sort_info_dict
from hmct.ir import load_model

from .accumulation_error import AccumulationError
from .node_info import NodeInfo
from .node_sensitivity import NodeSensitivity
from .parameter_distribution import ParameterDistribution
from .save_node_sensitivity import save_node_sensitivity
from .sensitivity_analysis import SensitivityAnalysis
from .util import check_parameters

if TYPE_CHECKING:
    from hmct.ir.onnx_utils import ModelProto


def get_sensitivity_of_nodes(
    model_or_file: Union["ModelProto", str],
    metrics: Union[List[str], str] = "cosine-similarity",
    calibrated_data: Optional[Union[str, Dataset]] = None,
    output_node: Optional[str] = None,
    node_type: str = "node",
    data_num: int = 1,
    verbose: bool = False,
    interested_nodes: Optional[Union[str, List[str], List[List[str]]]] = None,
) -> Dict[str, Dict[str, str]]:
    logging.getLogger().setLevel(logging.INFO)
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        data_num=data_num,
        metrics=metrics,
        nodes_list=output_node,
    )
    check_parameters(model_or_file=model_or_file, nodes_list=interested_nodes)
    logging.info(f"Start calculating {node_type} sensitivity...")
    model_debugger = NodeSensitivity(model_or_file, calibrated_data)
    if isinstance(metrics, str):
        metrics = [metrics]
    node_info = model_debugger.sensitivity_of_nodes(
        node_type,
        1,
        data_num,
        metrics,
        output_node,
        interested_nodes,
    )

    node_info_with_sensitivity = defaultdict(dict)
    if isinstance(interested_nodes, str):
        interested_nodes = [interested_nodes]
    if interested_nodes is not None and len(interested_nodes) == 0:
        logging.info("interested_nodes is neither None nor set.")
        return node_info_with_sensitivity

    node_info = sort_info_dict(
        node_info.get(node_type),
        metrics[0],
        Loss.create(metrics[0]).optimal == np.argmin,
    )
    if verbose:
        verbose_node_info = copy.deepcopy(node_info)
        for m in metrics:
            for v in verbose_node_info.values():
                v[m] = round(float(v[m]), 5)
        print_info_dict(
            verbose_node_info,
            title="node sensitivity",
            key_type=node_type,
        )
    for k in node_info:
        for key in node_info[k]:
            if key in metrics:
                node_info_with_sensitivity[k][key] = node_info[k][key]
    return node_info_with_sensitivity


def plot_distribution(
    model_or_file: Union["ModelProto", str],
    calibrated_data: Union[str, Dataset],
    nodes_list: Union[List[str], str],
    save_dir: str = "./debug_result/",
):
    logging.getLogger().setLevel(logging.INFO)
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=nodes_list,
    )
    logging.info("Start getting data distribution...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_debugger = ParameterDistribution(save_dir, model_or_file, calibrated_data)
    model_debugger.plot_distribution(nodes_list)


def get_channelwise_data_distribution(
    model_or_file: Union["ModelProto", str],
    calibrated_data: Union[str, Dataset],
    nodes_list: List[str],
    axis: Optional[int] = None,
    save_dir: str = "./debug_result/",
):
    logging.getLogger().setLevel(logging.INFO)
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=nodes_list,
    )
    logging.info("Start getting channelwise data distribution...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_debugger = ParameterDistribution(save_dir, model_or_file, calibrated_data)
    model_debugger.get_channelwise_data_distribution(nodes_list, axis)


def get_calibration_nodes_stat_info(
    model_or_file: Union["ModelProto", str],
    calibrated_data: Union[str, Dataset],
    nodes_list: Optional[List[str]] = None,
    save_dir: str = "./debug_result/",
    visualize: bool = False,
):
    """获取指定校准节点的统计信息.

    统计信息项:
        - 阈值
        - 各channel的最大绝对值
        - 各channel的最小非零绝对值
    """
    logging.getLogger().setLevel(logging.INFO)
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=nodes_list,
    )
    logging.info("Start getting calibration nodes statistical info...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_debugger = ParameterDistribution(save_dir, model_or_file, calibrated_data)
    stat_info = model_debugger.get_calibration_nodes_stat_info(nodes_list)
    np.save(save_dir + "cali_nodes_stat_info.npy", stat_info)

    if visualize:
        for node_name in stat_info:
            if stat_info[node_name]["Granularity"] == "per-channel":
                stat_info[node_name]["Thresholds"] = "{:.6f},...".format(
                    stat_info[node_name]["Thresholds"][0],
                )
                stat_info[node_name]["Scales"] = "{:.6f},...".format(
                    stat_info[node_name]["Scales"][0],
                )
                stat_info[node_name]["Max_vals"] = "{:.6f},...".format(
                    stat_info[node_name]["Max_vals"][0],
                )
                stat_info[node_name]["Min_nz_vals"] = "{:.6e},...".format(
                    stat_info[node_name]["Min_nz_vals"][0],
                )
            else:
                stat_info[node_name]["Thresholds"] = "{:.6f}".format(
                    stat_info[node_name]["Thresholds"][0],
                )
                stat_info[node_name]["Scales"] = "{:.6f}".format(
                    stat_info[node_name]["Scales"][0],
                )
                stat_info[node_name]["Max_vals"] = "{:.6f}".format(
                    np.amax(stat_info[node_name]["Max_vals"]),
                )
                min_vals = np.array(stat_info[node_name]["Min_nz_vals"])
                min_val = np.amin(min_vals[min_vals > 0])
                stat_info[node_name]["Min_nz_vals"] = f"{min_val:.6e}"
        print_info_dict(stat_info)

        for tensor_type in ["weight", "feature"]:
            plt.figure()
            for curve_name in ["Thresholds", "Scales", "Max_vals", "Min_nz_vals"]:
                data = [
                    float(v[curve_name].split(",")[0])
                    for k, v in stat_info.items()
                    if v["Tensor_type"] == tensor_type
                ]
                plt.plot(np.arange(0, len(data)), data, label=curve_name)
            plt.xlabel("Node")
            plt.ylabel("Value")
            plt.yscale("log")
            plt.title(f"{tensor_type} calibration nodes")
            plt.legend()
            plt.savefig(save_dir + f"{tensor_type}_stat_info.png")

    return stat_info


def plot_acc_error(
    calibrated_data: Union[str, Dataset],
    model_or_file: Union["ModelProto", str],
    quantize_node: Optional[Union[List[str], str]] = None,
    non_quantize_node: Optional[Union[List[str], str]] = None,
    metric: str = "cosine-similarity",
    average_mode: bool = False,
    save_dir: str = "./debug_result/",
):
    logging.getLogger().setLevel(logging.INFO)
    check_parameters(
        calibrated_data=calibrated_data,
        model_or_file=model_or_file,
        metrics=metric,
    )
    logging.info("Start ploting accumulate error...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_debugger = AccumulationError(save_dir, model_or_file, calibrated_data)
    model_debugger.plot_acc_error(
        quantize_node,
        non_quantize_node,
        metric,
        average_mode,
    )


def sensitivity_analysis(
    model_or_file,
    calibrated_data,
    pick_threshold=0.999,
    data_num=1,
    sensitive_nodes=None,
    save_dir="./debug_result/",
):
    logging.getLogger().setLevel(logging.INFO)
    if sensitive_nodes is None:
        sensitive_nodes = []
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        data_num=data_num,
        nodes_list=sensitive_nodes,
    )
    logging.info("Start analyzing node sensitivity...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(sensitive_nodes) == 0:
        logging.info(
            "Unspecified sensitive node, start calculating node sensitivity...",
        )
        node_sens = get_sensitivity_of_nodes(
            model_or_file=model_or_file,
            calibrated_data=calibrated_data,
            data_num=data_num,
        )
        for k, v in node_sens.items():
            if v["cosine-similarity"] <= pick_threshold:
                sensitive_nodes.append(k)
        logging.info(
            f"Analyzing nodes that sensitivity less than {pick_threshold}.",
        )
    model_debugger = SensitivityAnalysis(model_or_file, calibrated_data)
    model_debugger.sensitivity_analysis(data_num, sensitive_nodes, save_dir)


def runall(
    model_or_file: str,
    calibrated_data: Union[str, Dataset],
    save_dir: str = "./debug_result/",
    ns_metrics: Union[str, List[str]] = "cosine-similarity",
    output_node: Optional[str] = None,
    node_type: Optional[str] = None,
    data_num: int = 1,
    verbose: bool = False,
    interested_nodes: Optional[Union[str, List[str]]] = None,
    dis_nodes_list: Optional[Union[List[str], str]] = None,
    cw_nodes_list: Optional[Union[List[str], str]] = None,
    axis: Optional[int] = None,
    quantize_node: Optional[Union[List[str], str]] = None,
    non_quantize_node: Optional[Union[List[str], str]] = None,
    ae_metric: str = "cosine-similarity",
    average_mode: bool = False,
    pick_threshold: int = 0.999,
    sensitive_nodes: Optional[list] = None,
):
    logging.getLogger().setLevel(logging.INFO)
    if sensitive_nodes is None:
        sensitive_nodes = []
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        data_num=data_num,
        metrics=ns_metrics,
        nodes_list=output_node,
    )
    # 由于参数名无法重复, 所以单独对累积误差功能的metric进行判断
    check_parameters(metrics=ae_metric)

    if node_type is not None and node_type not in ["node", "weight", "activation"]:
        raise ValueError(f"node_type:{node_type} is unavailable.")

    check_parameters(model_or_file=model_or_file, nodes_list=dis_nodes_list)

    def flatten(nested_list):
        return (
            functools.reduce(operator.iadd, map(flatten, nested_list), [])
            if isinstance(nested_list, list)
            else [nested_list]
        )

    flatten_non_quantize_node = flatten(non_quantize_node)
    if (
        "weight" in flatten_non_quantize_node
        or "activation" in flatten_non_quantize_node
    ):
        raise ValueError(
            "In acc_error section, cannot set 'non_quantize_node' "
            "to 'weight' or 'activation'.",
        )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # node sensitivity
    nodes_list = []
    ns_save_dir = save_dir + "node_sensitivity/"
    if not os.path.exists(ns_save_dir):
        os.makedirs(ns_save_dir)
    if node_type is None:
        node_type = ["weight", "activation"]
        for n in node_type:
            node_message = get_sensitivity_of_nodes(
                model_or_file=model_or_file,
                metrics=ns_metrics,
                calibrated_data=calibrated_data,
                output_node=output_node,
                node_type=n,
                data_num=data_num,
                verbose=verbose,
                interested_nodes=interested_nodes,
            )
            if len(node_message.keys()) >= 5:
                nodes_list.extend(list(node_message.keys())[:5])
            else:
                nodes_list.extend(list(node_message.keys()))
            save_path = ns_save_dir + "sensitivity_of_" + n + ".log"
            save_node_sensitivity(node_message, save_path, n)
    else:
        node_message = get_sensitivity_of_nodes(
            model_or_file=model_or_file,
            metrics=ns_metrics,
            calibrated_data=calibrated_data,
            output_node=output_node,
            node_type=node_type,
            data_num=data_num,
            verbose=verbose,
            interested_nodes=interested_nodes,
        )
        if len(node_message.keys()) >= 5:
            nodes_name = list(node_message.keys())[:5]
        else:
            nodes_name = list(node_message.keys())
        if node_type == "node":
            model = load_model(model_or_file).proto
            node_info = NodeInfo(model)
            for node_name in nodes_name:
                nodes_list.extend(node_info.get_calibration_by_node(node_name))
        else:
            nodes_list = nodes_name
        nodes_list = list(set(nodes_list))
        save_path = ns_save_dir + "sensitivity_of_" + node_type + ".log"
        save_node_sensitivity(node_message, save_path, node_type)
    # data distribution
    dd_save_dir = save_dir + "data_distribution/"
    if not os.path.exists(dd_save_dir):
        os.makedirs(dd_save_dir)
    dis_nodes_list = nodes_list if dis_nodes_list is None else dis_nodes_list
    plot_distribution(
        save_dir=dd_save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=dis_nodes_list,
    )
    # channelwise data distribution
    cd_save_dir = save_dir + "channelwise_distribution/"
    if not os.path.exists(cd_save_dir):
        os.makedirs(cd_save_dir)
    cw_nodes_list = nodes_list if cw_nodes_list is None else cw_nodes_list
    axis = int(axis) if axis is not None else None
    get_channelwise_data_distribution(
        save_dir=cd_save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=cw_nodes_list,
        axis=axis,
    )
    # accumulate error
    ae_save_dir = save_dir + "accumulate_error/"
    if not os.path.exists(ae_save_dir):
        os.makedirs(ae_save_dir)
    if quantize_node is None and non_quantize_node is None:
        quantize_node = ["weight", "activation"]
    else:
        quantize_node = quantize_node
        non_quantize_node = non_quantize_node
    plot_acc_error(
        save_dir=ae_save_dir,
        calibrated_data=calibrated_data,
        model_or_file=model_or_file,
        quantize_node=quantize_node,
        non_quantize_node=non_quantize_node,
        metric=ae_metric,
        average_mode=average_mode,
    )
    # sensitivity analysis
    sa_save_dir = save_dir + "sensitivity_analysis/"
    if not os.path.exists(sa_save_dir):
        os.makedirs(sa_save_dir)
    if (
        node_type == "node"
        and "cosine-similarity" in ns_metrics
        and len(sensitive_nodes) == 0
    ):
        for k, v in node_message.items():
            if v["cosine-similarity"] <= pick_threshold:
                sensitive_nodes.append(k)
        if len(sensitive_nodes) == 0:
            logging.info(
                "pick_threshold is too small, there is no sensitive node "
                "that meets the requirements, please increase pick_threshold.",
            )
    sensitivity_analysis(
        model_or_file,
        calibrated_data,
        pick_threshold,
        data_num,
        sensitive_nodes,
        sa_save_dir,
    )
