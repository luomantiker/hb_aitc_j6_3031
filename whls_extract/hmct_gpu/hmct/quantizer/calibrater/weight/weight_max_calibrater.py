from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import Calibrater
from ..quantization_type import QuantizationType

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class WeightMaxCalibrater(Calibrater):
    def __init__(self, weight_config: Optional[Dict[str, Any]] = None):
        """Initialization for the max calibrater.

        Args:
            weight_config: weight calibration config for the max calibrater:
                1) max_percentile: The percentile value used to scale the
                calibration threshold of weights.
        """
        super().__init__()
        weight_config = {} if weight_config is None else weight_config
        self.qtype = QuantizationType()
        self.qtype.weight = "max_percentile" in weight_config
        self.max_percentile = weight_config.get("max_percentile", 1.0)
        self.block_size = weight_config.get("block_size", 0)
        self.quantize_type = weight_config.get("quantize_type", "scale")

    @property
    def name(self) -> str:
        return "weight_max_calibrater"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":
        """基于max校准的模型权重阈值计算.

        Args:
            calibrated_model: 模型权重待校准的模型(已插入校准节点)
            **kwargs: 其他未被使用到的传入参数

        Returns:
            模型权重完成校准的模型
        """
        return super().run_impl(calibrated_model, **kwargs)
