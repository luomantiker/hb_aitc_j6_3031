from typing import TYPE_CHECKING, Dict, List, Union

from hmct.common import Loss
from hmct.executor import ORTExecutor

if TYPE_CHECKING:
    import numpy as np

    from hmct.ir import OnnxModel


def convert_to_ptq_model(calibrated_model: "OnnxModel"):
    """Convert calibration model to post training quantized model.

    Args:
        calibrated_model: The input calibrated model.

    Returns:
        Onnx model with all calibration nodes fake quantization enable.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        node.switch = "ON"
        if node.constant == 0:
            node.constant = 1
    return calibrated_model


def calculate_models_similarities(
    baseline_model: "OnnxModel",
    candidate_models: List["OnnxModel"],
    input_data: Union[List["np.ndarray"], Dict[str, "np.ndarray"], "np.ndarray"],
    loss_name: str = "cosine-similarity",
) -> List[float]:
    """Calculate the similarity between the baseline model candidate models.

    Args:
        baseline_model: The baseline model to compare other models against.
        candidate_models: List of candidate models to compare against the
            baseline model.
        input_data: The input data used to infer outputs from the models.
            It can be a list of numpy arrays, a dictionary of numpy
            arrays, or a single numpy array.
        loss_name: The name of the loss function to use for comparing models.

    Returns:
        List of similarity value of input candidate models (without sorted).
    """
    loss_func = Loss.create(loss_name)
    baseline_outputs = ORTExecutor(baseline_model).inference(input_data)

    candidate_similarities = []
    for candidate_model in candidate_models:
        candidate_outputs = ORTExecutor(candidate_model).inference(input_data)
        candidate_similarities.append(
            loss_func.run(baseline_outputs, candidate_outputs),
        )
    return candidate_similarities
