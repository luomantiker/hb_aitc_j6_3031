from typing import TYPE_CHECKING, Any, Dict

from .weight_max_calibrater import WeightMaxCalibrater

if TYPE_CHECKING:
    from ..base import CalibrationPass


def weight_calibration(weight_config: Dict[str, Any]) -> "CalibrationPass":
    wgt_types = weight_config.get("calibration_type", "max")
    if not isinstance(wgt_types, list):
        wgt_types = [wgt_types]

    if len(wgt_types) == 1 and wgt_types[0] == "max":
        return WeightMaxCalibrater(weight_config)
    if len(wgt_types) == 1 and wgt_types[0] == "load":
        return WeightMaxCalibrater()
    raise ValueError(f"Unsupported weight calibration type: {wgt_types}.")
