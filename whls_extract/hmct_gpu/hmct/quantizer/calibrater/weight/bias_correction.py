import copy
import itertools
import logging
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from tqdm import tqdm

from hmct.common import (
    Dataset,
    Loss,
    add_model_output,
    print_info_dict,
    set_model_switch,
)
from hmct.executor import ORTExecutor
from hmct.quantizer.debugger import AccumulationError

from ..base import CalibrationPass
from ..quantization_type import QuantizationType
from ..utils import calculate_models_similarities

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class BiasCorrection(CalibrationPass):
    """Bias Correction aims to reduce output bias for calibrated model.

    It make calibrated model an unbiased estimation of the
    float model by adjusting bias values of quantized
    Conv/ConvTranspose/Gemm nodes. The process is as follows:
        1. infer float model: y = wx + b
        2. infer calibrated model: q(y) = q(w) q(x) + b
        3. calculate quantization error: correction = mean(y - q(y))
        4. update bias values: b = b + correction

    Refs:
        https://arxiv.org/abs/1906.04721
    """

    def __init__(self, weight_config: Dict[str, Any]):
        bias_correction = weight_config.get("bias_correction", {})
        self.bias_samples = bias_correction.get("num_sample", 1)
        self.bias_metric = bias_correction.get("metric", "cosine-similarity")
        self.loss_func = Loss.create(self.bias_metric)
        self.qtype = QuantizationType()
        self.qtype.bias_correction = True

    @property
    def name(self) -> str:
        return "bias_correction"

    def add_bias_input(self, bias_model: "OnnxModel") -> "OnnxModel":
        """Adds bias input to Conv and ConvTranspose nodes in the model.

        Args:
            bias_model: Model to which bias inputs needs to be added.

        Returns:
            Onnx model with all Conv and ConvTranspose nodes have bias.
        """
        for node in itertools.chain(
            bias_model.graph.type2nodes["Conv"],
            bias_model.graph.type2nodes["ConvTranspose"],
        ):
            if len(node.inputs) == 2:
                # bias的shape大小为node的输出通道大小
                bias_shape = node.output_shapes[0][1]
                bias_var = bias_model.graph.create_variable(
                    name=node.name + "_bias",
                    is_param=True,
                    value=np.zeros(bias_shape).astype(np.float32),
                )
                node.append_input(bias_var)
        return bias_model

    def calculate_correction(
        self,
        bias_model: "OnnxModel",
        calibration_dataset: Dataset,
        messages_verbose=False,
    ) -> "OnnxModel":
        """Calculates bias correction for bias model.

        The errors between float model and bias model on the output
        of Conv and ConvTranspose nodes were calculated.

        If adjusting the bias based on these errors results in a
        better similarity of the bias model compared to before the
        adjustment, accept the current adjustment; otherwise, retain the
        original bias.

        Args:
            bias_model: Model for which correction needs to be calculated.
            calibration_dataset: Calibration dataset.
            messages_verbose: Whether to print verbose messages.

        Returns:
            Onnx model with bias corrected.
        """
        # 记录校准模型的原始输出, 在调整完bias之后更新到模型上。
        graph_outputs = bias_model.graph.output_names

        # 在Conv和ConvTranspose节点后插入输出, 基于校准数据推理得到浮点输出
        input_data = calibration_dataset.copy(self.bias_samples)
        float_model = set_model_switch(copy.deepcopy(bias_model), "OFF")
        float_model = add_model_output(
            float_model,
            output_op_types=["Conv", "ConvTranspose"],
        )
        float_model_outputs = (
            ORTExecutor(float_model).create_session().forward_with_batch(input_data)
        )
        bias_model = add_model_output(
            bias_model,
            output_op_types=["Conv", "ConvTranspose"],
        )

        # 逐个节点遍历对节点的bias值进行调整
        correction_messages = {}
        for node in tqdm(
            itertools.chain(
                bias_model.graph.type2nodes["Conv"],
                bias_model.graph.type2nodes["ConvTranspose"],
            ),
            desc="bias correction in progress",
        ):
            node_output_name, node_output_shape = (
                node.output_names[0],
                node.output_shapes[0],
            )
            bias_model_outputs = (
                ORTExecutor(bias_model)
                .create_session()
                .forward_with_batch(input_data, output_names=[node_output_name])
            )
            # 计算校准模型和浮点模型的量化误差均值, 对于Conv和ConvTranspose而言,
            # 需要计算在每一个输出通道(节点输出的第1维度)上的均值
            correction = np.mean(
                np.concatenate(float_model_outputs[node_output_name], axis=0)
                - np.concatenate(bias_model_outputs[node_output_name], axis=0),
                tuple(i for i in range(len(node_output_shape)) if i != 1),
                keepdims=True,
            )
            # 计算调整bias前后校准模型同浮点模型的相似度, 若调整bias能够有更优
            # 的相似度表现则当前节点进行bias的优化, 否则仍采用原来的bias值
            loss_before = self.loss_func.run(
                float_model_outputs[node_output_name],
                bias_model_outputs[node_output_name],
            )
            loss_after = self.loss_func.run(
                float_model_outputs[node_output_name],
                [o + correction for o in bias_model_outputs[node_output_name]],
            )
            if self.loss_func.optimal([loss_before, loss_after]):
                node.inputs[2].value += correction.flatten()
            correction_messages[node.name] = {
                "Before-Bias-Correlation": f"{loss_before:.6f}",
                "After-Bias-Correlation": f"{loss_after:.6f}",
            }
        # 将校准模型的输出调整为原状
        bias_model = add_model_output(
            bias_model,
            output_tensors=graph_outputs,
            keep_original_output=False,
        )
        # 对校准模型bias的调整情况进行可视化
        if messages_verbose:
            print_info_dict(
                correction_messages,
                title=f"Bias_Correction_Info({self.bias_metric})",
            )
        return bias_model

    def get_best_model(
        self,
        candidate_models: List["OnnxModel"],
        calibration_dataset: Dataset,
        messages_verbose: bool = True,
    ) -> "OnnxModel":
        """Compare original calibrated model and bias model, select best one.

        Args:
            candidate_models: List of models with calibrated model and
                bias model.
            calibration_dataset: Calibration dataset.
            messages_verbose: Whether to print verbose messages.

        Returns:
            Best model from calibrated model and bias model.
        """

        def extract_input_data(calibration_data, samples=1):
            samples = min(calibration_data.number_of_samples, samples)
            return calibration_data.copy(samples)

        # 从校准数据中抽取最后一张数据验证数据进行模型比较, 并返回相似度情况
        input_data = extract_input_data(calibration_dataset)
        similarities = calculate_models_similarities(
            set_model_switch(copy.deepcopy(candidate_models[0]), "OFF"),
            candidate_models,
            input_data.get_data(),
            loss_name=self.bias_metric,
        )
        # 可视化偏差校正前后的相似度以及误差累积情况
        if messages_verbose:
            correction_messages = {
                "output": {
                    "Before-Bias-Correlation": f"{similarities[0]:.6f}",
                    "After-Bias-Correlation": f"{similarities[1]:.6f}",
                },
            }
            print_info_dict(
                correction_messages,
                title=f"Bias_Correction_Info({self.bias_metric})",
            )
            model_debugger = AccumulationError(
                "./bias_correction",
                candidate_models[0].proto,
                input_data,
            )
            cali_models = {
                "Before-Bias-Correlation": candidate_models[0].proto,
                "After-Bias-Correlation": candidate_models[1].proto,
            }
            model_debugger.accumulate_error_analysis(
                cali_models,
                file_prefix="node_accumulate_err_of_qmodel",
            )
            model_debugger.plot_conv_acc_error(
                cali_models,
                file_prefix="conv_accumulate_err_of_qmodel",
            )

        return candidate_models[self.loss_func.optimal(similarities)]

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Optimize calibrated model with bias correction method.

        Args:
            calibrated_model: The model for applying bias correction.
            calibration_dataset: The calibration dataset to be used for
                bias correction.
            **kwargs: Other unused parameters.

        Returns:
            The best one in bias corrected model and original calibrated model.
        """
        logging.info(
            f"Run calibration model with bias correction method "
            f"with bias samples: {self.bias_samples}.",
        )
        # 为模型中的Conv和ConvTranspose节点补充bias输入
        bias_model = self.add_bias_input(copy.deepcopy(calibrated_model))

        # 基于校准数据计算并校准偏差, 并更新Conv和ConvTranspose的bias值
        bias_model = self.calculate_correction(bias_model, calibration_dataset)

        # 比较原始模型和偏差校准模型, 获取最优模型
        best_model = self.get_best_model(
            [calibrated_model, bias_model],
            calibration_dataset,
        )

        best_model.check_validity()
        return best_model
