from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def set_model_switch(calibrated_model: "OnnxModel", switch: str) -> "OnnxModel":
    """Set the switch attribute of HzCalibration node to "ON" or "OFF".

    Args:
        calibrated_model: The calibrated model with HzCalibration inserted.
        switch: The switch attribute to be set, only support "ON" or "OFF".

    Returns:
        The original model with all HzCalibration switches were set.
    """
    assert switch in ["ON", "OFF"], f"switch should be ON or OFF, but got {switch}"
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        node.switch = switch
    return calibrated_model
