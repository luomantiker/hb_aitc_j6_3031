from typing import TYPE_CHECKING, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from .debugger import CalibrationModel
from .node_sensitivity import NodeSensitivity
from .util import set_calibration_data

if TYPE_CHECKING:
    from hmct.common import Dataset
    from hmct.ir.onnx_utils import ModelProto

mpl.use("Agg")


class SensitivityAnalysis:
    def __init__(
        self,
        model_or_file: Union["ModelProto", str],
        calibrated_data: Optional[Union[str, "Dataset"]] = None,
    ):
        # 设置校准数据
        self.calibrated_data = set_calibration_data(calibrated_data)

        self.cali_model_attr = CalibrationModel(model_or_file, self.calibrated_data)

        # 原始模型
        self.original_model = self.cali_model_attr.original_model

        # 初始化节点敏感度模块
        self.node_sensitivity = NodeSensitivity(model_or_file, self.calibrated_data)

    # 传入部分量化模型, 获取该模型最终输出的相似度
    # model: 部分量化模型
    # data_num: 校准数据数量
    # original_model_output_dict: 浮点模型的推理结果
    def get_model_output_sensitivity(self, model, data_num, original_model_output_dict):
        output_dict = self.cali_model_attr.get_model_output(
            model=model,
            model_output_list=self.cali_model_attr.model_outputs,
            batch_size=1,
            data_num=data_num,
        )
        output_sensitivity = self.node_sensitivity.sensitivity(
            original_output_dict=original_model_output_dict,
            output_dict=output_dict,
            metric_name="cosine-similarity",
        )
        return output_sensitivity  # noqa: RET504

    def sensitivity_analysis(self, data_num, sensitive_nodes, save_dir):
        if isinstance(sensitive_nodes, str):
            sensitive_nodes = [sensitive_nodes]

        # 获取浮点模型的推理结果
        original_model_output_dict = self.cali_model_attr.get_model_output(
            self.original_model,
            self.cali_model_attr.model_outputs,
            1,
            data_num,
        )

        # 用于记录单节点量化模型最终输出的相似度
        single_node_q_sens_dict = {}
        # 用于记录部分量化模型最终输出的相似度
        partial_noq_sens_dict = {}

        # 校准模型输出的相似度
        calibrated_model_sens = self.get_model_output_sensitivity(
            self.cali_model_attr.calibrated_model,
            data_num,
            original_model_output_dict,
        )
        partial_noq_sens_dict["none"] = float(calibrated_model_sens)

        # 记录不量化节点
        blacklist_nodes = []
        for node in tqdm(sensitive_nodes, desc="Analyzing"):
            # 单节点量化模型
            single_node_q_model = self.cali_model_attr.get_partial_qmodel_by_whitelist(
                whitelist_nodes=node,
            )
            single_node_q_sens = self.get_model_output_sensitivity(
                single_node_q_model,
                data_num,
                original_model_output_dict,
            )
            # keys: 单节点量化模型中被量化的节点
            # values: 模型最终输出的相似度
            single_node_q_sens_dict[node] = float(single_node_q_sens)

            # 获取部分量化模型
            blacklist_nodes.append(node)
            partial_noq_model = self.cali_model_attr.get_partial_qmodel_by_blacklist(
                blacklist_nodes=blacklist_nodes,
            )
            partial_noq_sens = self.get_model_output_sensitivity(
                partial_noq_model,
                data_num,
                original_model_output_dict,
            )
            # keys: 当前节点, 当前节点以及之前的所有节点都不量化
            # values: 模型最终输出的相似度
            partial_noq_sens_dict[node] = float(partial_noq_sens)

        save_path = save_dir + "/partial_quantization.png"
        plt.clf()
        plt.grid(True)
        sensitive_nodes.insert(0, "none")
        plt.plot(
            sensitive_nodes,
            [1.0] * len(sensitive_nodes),
            linestyle="dashed",
            color="blue",
            label="baseline",
        )

        plt.plot(
            list(partial_noq_sens_dict.keys()),
            list(partial_noq_sens_dict.values()),
            color="red",
            label="partial quantization cumulative error",
        )

        plt.scatter(
            list(single_node_q_sens_dict.keys()),
            list(single_node_q_sens_dict.values()),
            color="green",
            marker="x",
            label="per-node quantization sensitivity",
        )
        plt.xticks(rotation=90)
        plt.title("Partial Quantization Analysis")
        plt.ylabel("Cosine Similarity")
        plt.xlabel("node")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(save_path, bbox_inches="tight")
        plt.legend()
