from copy import deepcopy
import operator
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from hmct.common import Loss

from ..activation.base import set_calibration_method
from ..activation.post_calibration import PostCalibration
from ..utils import calculate_models_similarities
from ..weight.weight_max_calibrater import WeightMaxCalibrater

if TYPE_CHECKING:
    from hmct.ir import OnnxModel

    from ..calibration_method import CalibrationMethod


def shared_calibration_methods(calibrated_model: "OnnxModel") -> List[str]:
    """获取模型校准节点共有的校准方法有序列表."""
    node_methods = []
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        if len(node.calibration_thresholds) == 0:
            continue
        node_methods.append(set(node.calibration_thresholds.keys()))
    return sorted(set.intersection(*node_methods)) if node_methods else []


def compare_calibration_methods(
    input_data: Dict[str, np.ndarray],
    float_model: "OnnxModel",
    calibrated_model: "OnnxModel",
    calibration_methods: Union["CalibrationMethod", Sequence[str]],
    metric: str = "cosine-similarity",
) -> List[Tuple[str, float]]:
    """Compare calibration methods.

    Args:
        input_data: Input data for model evaluation.
        float_model: Float model as baseline to compare against.
        calibrated_model: Model with calibration thresholds under
            calibration_methods recorded.
        calibration_methods: Multi calibration methods for comparison.
        metric: Loss metric for comparison.

    Returns:
        List with tuple[0] is str calibration method,
        and tuple[1] is float sensitivity under calibration method.
    """
    # TODO(saiqiang.zhang): 后续结合J6OR和土五模型确认是否需要增加校准权重阈值
    # 用于模型比较,当前为保持跟dev分支精度一致,采用Max方法校准权重.
    calibrated_model = WeightMaxCalibrater()(deepcopy(calibrated_model))
    calibrated_models = [
        set_calibration_method(deepcopy(calibrated_model), method)
        for method in calibration_methods
    ]
    post_models = [PostCalibration()(model) for model in calibrated_models]
    similarities = calculate_models_similarities(
        float_model, post_models, input_data, metric
    )
    sorted_methods = [
        (method, similarity)
        for method, similarity in zip(map(str, calibration_methods), similarities)
        if not np.isnan(similarity)
    ]
    sorted_methods = sorted(
        sorted_methods,
        key=operator.itemgetter(1),
        reverse=Loss.create(metric).optimal == np.argmax,
    )
    nan_methods = [
        (method, similarity)
        for method, similarity in zip(map(str, calibration_methods), similarities)
        if np.isnan(similarity)
    ]
    return sorted_methods + nan_methods
