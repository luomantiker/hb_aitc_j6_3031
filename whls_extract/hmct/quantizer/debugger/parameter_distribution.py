from collections import defaultdict
import copy
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from hmct.common import (
    Dataset,
    Loss,
    add_model_output,
)
from hmct.executor import ORTExecutor
from hmct.ir import CalibrationNode, OnnxModel
from hmct.ir.onnx_utils import ModelProto, numpy_helper

from .debugger import CalibrationModel
from .util import set_calibration_data

mpl.use("Agg")


def find_by_name(item_name, item_list):
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def change_shape(input_data, channel_loc):
    shape_index = list(range(len(input_data.shape)))
    new_shape = [channel_loc]
    del shape_index[channel_loc]
    new_shape.extend(shape_index)
    input_data = input_data.transpose(new_shape)
    return input_data, new_shape


def calculate_tensor(value, scales, qtype):
    if qtype == "float16":
        return np.array(value).astype(np.float16).astype(np.float32).item()
    if qtype == "bfloat16":
        return (
            torch.tensor(value).to(dtype=torch.bfloat16).to(dtype=torch.float32).item()
        )
    if qtype == "float32":
        return value

    bits = int(qtype.split("int")[-1])
    max_value = pow(2.0, bits - 1) - 1
    min_value = -pow(2.0, bits - 1)
    scales = scales * 127.0 / max_value
    value = (
        round(value / scales)
        if value - round(value / scales) < 0.5
        else round(value / scales) + 1
    )
    value = min(max(value, min_value), max_value)
    return value * scales


def fake_quantize(origin_weight, scales, qtype):
    channel = origin_weight.shape[0]
    calibrated_weight = []
    for i in range(channel):
        origin_tensor = origin_weight[i]
        tensor_shape = origin_tensor.shape
        origin_tensor = origin_tensor.flatten()
        calibrated_tensor = np.array(
            [calculate_tensor(value, scales[i], qtype) for value in origin_tensor],
        )
        calibrated_tensor = calibrated_tensor.reshape(tensor_shape)
        calibrated_weight.append(calibrated_tensor)
    return np.array(calibrated_weight)


def get_weights(initializer, node, n_node):
    origin_weight = numpy_helper.to_array(initializer)
    axis = find_by_name("axis", node.attribute).i
    if axis == 1:
        group_attr = find_by_name("group", n_node.attribute)
        group = 1 if group_attr is None else group_attr.i
        weight_shape = origin_weight.shape
        origin_weight = origin_weight.reshape(
            weight_shape[0] // group,
            weight_shape[1] * group,
            weight_shape[2],
            weight_shape[3],
        )
    # change shape index, new_shape[0] is channel
    origin_weight, new_shape = change_shape(origin_weight, axis)

    scales = 0
    qtype = "int8"
    for _, attr in enumerate(node.attribute):
        if attr.name == "scales":
            scales = attr.floats
        if attr.name == "qtype":
            qtype = attr.s.decode("utf-8")
    calibrated_weight = fake_quantize(origin_weight, scales, qtype)
    # change new_shape to original shape.
    origin_weight = origin_weight.transpose(new_shape)
    calibrated_weight = calibrated_weight.transpose(new_shape)
    return origin_weight, calibrated_weight, axis


class ParameterDistribution:
    def __init__(
        self,
        save_dir: str,
        model_or_file: ModelProto or str,
        calibrated_data: str or Dataset = None,
    ) -> None:
        self.calibrated_data = set_calibration_data(calibrated_data)

        self.cali_model_attr = CalibrationModel(model_or_file, self.calibrated_data)
        self.conv_node_dict = self.cali_model_attr.conv_node_dict
        self.calibrated_model = self.cali_model_attr.calibrated_model
        self.original_model = self.cali_model_attr.original_model
        self.calibration_node = self.cali_model_attr.calibration_node

        self.save_dir = save_dir

    def get_node_by_weight(self, w_cali_node):
        output_name = w_cali_node.output[0]
        for node in self.calibrated_model.graph.node:
            if (
                node.op_type in ["Conv", "ConvTranspose", "HzPreprocess"]
                and node.input[1] == output_name
            ):
                return node

        return None

    def generate_plt(
        self,
        fdata,
        cdata,
        name,
        f_label,
        c_label,
        title,
        dir_name,
        threshold=None,
        err_label=None,
        err_data=None,
    ):
        plt.clf()
        plt_hist = partial(plt.hist, bins=1024, density=False)
        if threshold is not None:
            plt.axvline(x=threshold, ls="--", c="red")
        fdata_ = [i for i in fdata if i != 0]
        plt_hist(x=fdata_, color="#99ffbb", label=f_label)
        if cdata is not None and c_label is not None:
            cdata_ = np.array([i for i in cdata if i != 0])
            plt_hist(x=cdata_, color="#FA8072", label=c_label)
            cosin_dis = Loss.create("cosine-similarity").run(fdata, cdata)
            plt.title(title + "\n" + "cosine-similarity:" + str(cosin_dis))
        else:
            plt.title(title)
        if err_label is not None and err_data is not None:
            plt_hist(x=err_data, color="#33A1C9", label=err_label)
        plt.scatter(max(max(fdata), abs(min(fdata))), 0, marker="^", c="#0000FF")
        plt.legend()
        plt.xlabel("data range")
        plt.ylabel("Total number of data")
        file_name = "{}/{}_{}.png".format(
            self.save_dir,
            dir_name,
            name.replace("/", "_"),
        )
        plt.savefig(file_name, bbox_inches="tight")

    def get_channelwise_data_distribution(self, node_list, axis):
        if isinstance(node_list, str):
            node_list = [node_list]
        initial_dict = {}
        for node in self.calibrated_model.graph.initializer:
            initial_dict[node.name] = node

        weight_dict = {}
        weight_node_dict = {}
        for node in self.calibrated_model.graph.node:
            if node.name in list(self.cali_model_attr.weight_calibration_nodes.keys()):
                weight_dict[node.name] = initial_dict[
                    self.cali_model_attr.weight_calibration_nodes[node.name]["input"]
                ]
                weight_node_dict[node.name] = node
        for node_name in tqdm(node_list, desc="Testing"):
            if node_name in list(weight_dict.keys()):
                # weight calibration node
                n_node = self.get_node_by_weight(weight_node_dict[node_name])
                origin_weight, _, default_axis = get_weights(
                    weight_dict[node_name],
                    weight_node_dict[node_name],
                    n_node,
                )
                np_origin = np.array(origin_weight)
                tmp_axis = default_axis if axis is None else axis
                # np_origin.shape[0] is channel
                np_origin, _ = change_shape(np_origin, tmp_axis)
            elif node_name in list(
                self.cali_model_attr.feature_calibration_nodes.keys(),
            ):
                # activation calibrated node
                output_names = [
                    self.cali_model_attr.feature_calibration_nodes[node_name]["input"],
                ]
                output_list = self.cali_model_attr.get_model_output(
                    self.calibrated_model,
                    output_names,
                    1,
                    1,
                )
                np_origin_activation = np.array(output_list[output_names[0]][0])
                tmp_axis = 1 if axis is None else axis
                # np_origin.shape[0] is channel
                np_origin, _ = change_shape(np_origin_activation, tmp_axis)

            data_list = [
                list(np_origin[i].flatten()) for i in range(np_origin.shape[0])
            ]

            plt.clf()
            if len(data_list) >= 640:
                plt.figure(figsize=(len(data_list) // 5, 8))
                plt.xticks(rotation=90)
            elif len(data_list) != 1:
                plt.figure(figsize=(len(data_list) // 2, 8))
            else:
                plt.figure(figsize=(1, 8))

            plt.boxplot(
                data_list,
                medianprops={"color": "red", "linewidth": "1"},
                meanline=True,
                showmeans=True,
                meanprops={"color": "blue", "ls": "--", "linewidth": "1"},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 2},
            )
            plt.xlabel("channel")
            plt.ylabel("Range of data in channel")
            file_name = "{}/{}_boxplot.png".format(
                self.save_dir,
                node_name.replace("/", "_"),
            )
            plt.savefig(file_name, bbox_inches="tight")

    def plot_distribution(self, nodes):
        if isinstance(nodes, str):
            nodes = [nodes]

        get_cali_output = partial(
            self.cali_model_attr.get_model_output,
            model=self.calibrated_model,
            batch_size=1,
            data_num=1,
        )
        get_org_output = partial(
            self.cali_model_attr.get_model_output,
            model=self.original_model,
            batch_size=1,
            data_num=1,
        )

        # get activation nodes
        activation_node_list = defaultdict(list)
        for node in OnnxModel(self.calibrated_model).graph.nodes:
            if isinstance(node, CalibrationNode) and node.tensor_type == "feature":
                activation_node_list[node.name] = [node.input_names[0], node.proto]

        # get weight nodes
        weight_node_list = defaultdict(list)
        weight_node_name = list(self.cali_model_attr.weight_calibration_nodes.keys())
        w_nodes_dict = {}
        for node in self.calibrated_model.graph.node:
            if node.name in weight_node_name:
                w_nodes_dict[node.name] = node
        initializer = list(self.calibrated_model.graph.initializer)
        for i in initializer:
            name = i.name + "_HzCalibration"
            if name in weight_node_name:
                weight_node_list[name].extend([i, w_nodes_dict[name]])

        # get normal node
        normal_node_list = defaultdict(list)
        for node in self.original_model.graph.node:
            normal_node_list[node.name].extend(node.output)

        for node in tqdm(nodes, desc="Testing"):
            threshold = None
            err_label = None
            err_data = None
            label_1 = "Before calibration"
            label_2 = "After calibration"
            if node in activation_node_list:
                for item in activation_node_list[node][1].attribute:
                    if item.name == "thresholds":
                        threshold = max(list(item.floats))
                set_model_output = [activation_node_list[node][0]]
                fdata = get_cali_output(model_output_list=set_model_output)[
                    set_model_output[0]
                ][0].flatten()
                cdata = None
                node_type = "activation"
                title_suffix = " activation distribution"
                label_1 = "Input data distribution"
                label_2 = None
                self.generate_plt(
                    fdata=fdata,
                    cdata=cdata,
                    name=node,
                    f_label=label_1,
                    c_label=label_2,
                    title=node + title_suffix,
                    dir_name=node_type,
                    threshold=threshold,
                    err_label=err_label,
                    err_data=err_data,
                )
            elif node in weight_node_list:
                n_node = self.get_node_by_weight(weight_node_list[node][1])
                fdata, cdata, _ = get_weights(
                    weight_node_list[node][0],
                    weight_node_list[node][1],
                    n_node,
                )
                fdata = fdata.flatten()
                cdata = cdata.flatten()
                node_type = "weight"
                title_suffix = " weight distribution"
                self.generate_plt(
                    fdata=fdata,
                    cdata=cdata,
                    name=node,
                    f_label=label_1,
                    c_label=label_2,
                    title=node + title_suffix,
                    dir_name=node_type,
                    threshold=threshold,
                    err_label=err_label,
                    err_data=err_data,
                )
            elif node in normal_node_list:
                set_model_output = normal_node_list[node]
                fdata_list = get_org_output(model_output_list=set_model_output)
                cdata_list = get_cali_output(model_output_list=set_model_output)
                for i in set_model_output:
                    fdata = fdata_list[i][0].flatten()
                    cdata = cdata_list[i][0].flatten()
                    err_data = fdata - cdata
                    err_label = "Quantization error distribution"
                    title_suffix = f" output:{i} distribution"
                    node_type = "node_output"
                    self.generate_plt(
                        fdata=fdata,
                        cdata=cdata,
                        name=":" + i + "_of_" + node,
                        f_label=label_1,
                        c_label=label_2,
                        title=node + title_suffix,
                        dir_name=node_type,
                        threshold=threshold,
                        err_label=err_label,
                        err_data=err_data,
                    )

    def get_calibration_nodes_stat_info(self, node_list=None):
        if node_list is None:
            node_list = self.calibration_node.keys()

        calibrated_model = OnnxModel(copy.deepcopy(self.calibrated_model))

        # 获取weight校准节点的统计信息
        nodes_stat_info = {}
        for node in calibrated_model.graph.type2nodes["HzCalibration"]:
            if node.name in node_list and node.tensor_type == "weight":
                nodes_stat_info[node.name] = {
                    "Tensor_type": node.tensor_type,
                    "Granularity": node.granularity,
                    "Thresholds": list(node.thresholds),
                    "Scales": list(node.scales),
                }
                weight = node.inputs[0].value
                weight = np.abs(weight.reshape(weight.shape[0], -1))
                nodes_stat_info[node.name]["Max_vals"] = list(np.amax(weight, axis=-1))
                # 将零值改为np.inf, 从而确保取得大于0的最小值
                weight[weight == 0] = np.inf
                min_nz_vals = np.amin(weight, axis=-1)
                # 当输入值全为0时, 取得的最小值为np.inf, 因此需要替换为0
                min_nz_vals[min_nz_vals == np.inf] = 0
                nodes_stat_info[node.name]["Min_nz_vals"] = list(min_nz_vals)

        # 获取feature校准节点的统计信息
        output_list = []
        output_name_to_node = {}
        for node in calibrated_model.graph.type2nodes["HzCalibration"]:
            node.switch = "OFF"
            if node.name in node_list and node.tensor_type == "feature":
                nodes_stat_info[node.name] = {
                    "Tensor_type": node.tensor_type,
                    "Granularity": node.granularity,
                    "Thresholds": list(node.thresholds),
                    "Scales": list(node.scales),
                }

                for output in node.outputs:
                    output_list.append(output.name)
                    output_name_to_node[output.name] = node.name

        calibrated_model = add_model_output(
            calibrated_model,
            output_tensors=output_list,
            keep_original_output=False,
        )
        executor = ORTExecutor(calibrated_model).create_session()
        output_dict = executor.forward_with_batch(
            self.calibrated_data,
            batch_size=1,
            output_names=output_list,
        )

        for output_name, feats in output_dict.items():
            node_name = output_name_to_node[output_name]
            feats = np.concatenate(feats).transpose(1, 0, 2, 3)
            feats = np.abs(feats.reshape(feats.shape[0], -1))
            nodes_stat_info[node_name]["Max_vals"] = list(np.amax(feats, axis=-1))
            # 将零值改为np.inf, 从而确保取得大于0的最小值
            feats[feats == 0] = np.inf
            min_nz_vals = np.amin(feats, axis=-1)
            # 当输入值全为0时, 取得的最小值为np.inf, 因此需要替换为0
            min_nz_vals[min_nz_vals == np.inf] = 0
            nodes_stat_info[node_name]["Min_nz_vals"] = list(min_nz_vals)

        return nodes_stat_info
