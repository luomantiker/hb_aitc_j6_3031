from typing import TYPE_CHECKING, Any, Dict

from .activation import (
    ActivationEqualization,
    PostCalibration,
    activation_calibration,
)
from .base import CalibrationPipeline
from .post import (
    AdjustConvQuantParams,
    AdjustEvenStrideDWS,
    PowOfTwo,
    RefineThreshold,
)
from .quantization_type import QuantizationType
from .search_methods import ModelWiseSearch
from .weight import BiasCorrection, weight_calibration

if TYPE_CHECKING:
    from hmct.common import QuantConfig


def extract_cal_type(calibration_pipeline: CalibrationPipeline) -> str:
    """获取校准类型信息, 如采用的校准方法, 是否非对称以及是否偏差校正."""
    cal_type = QuantizationType()
    for cal_pass in calibration_pipeline:
        if hasattr(cal_pass, "qtype") and isinstance(cal_pass.qtype, QuantizationType):
            cal_type.update(cal_pass.qtype)
    return cal_type.type_str()


def create_calibration_pipeline(
    quant_config: "QuantConfig",
) -> CalibrationPipeline:
    """构建校准量化pipeline.

    支持情况如下:
        1. simple: 仅进行权重激活校准, 不涉及任何优化;
        2. basic: 权重激活校准完之后, 支持基本的精度优化方法(非对称, 偏差校准)
    """
    # 现阶段基于激活校准方法选择量化pipeline, 后续如果有更合适的选择标准再进行替换.
    if quant_config.activation_config.get("calibration_type") in ["fixed", "load"]:
        return create_simple_pipeline(
            quant_config.march,
            quant_config.activation_config,
            quant_config.weight_config,
        )
    return create_basic_pipeline(quant_config)


def create_simple_pipeline(
    march: str,
    activation_config: Dict[str, Any],
    weight_config: Dict[str, Any],
) -> CalibrationPipeline:
    """构建simple量化pipeline.

    Step1: 激活阈值计算;
    Step2: 权重阈值计算;
    Step3: 检查阈值和量化类型.
    """
    pipeline = CalibrationPipeline()
    # Step1: Calculate activation calibration thresholds.
    pipeline.set(activation_calibration(activation_config, {}, {}))
    pipeline.set(PostCalibration())
    # Step2: Calculate weight calibration thresholds.
    pipeline.set(weight_calibration(weight_config))
    # Step3: Refine and modify quantization params.
    pipeline.set(RefineThreshold())
    if march == "bernoulli":
        pipeline.set(PowOfTwo())

    return pipeline


def create_basic_pipeline(
    quant_config: "QuantConfig",
) -> CalibrationPipeline:
    """构建basic量化pipeline.

    Step1: 激活阈值计算;
    Step2: 权重阈值计算;
    Step3: 检查阈值和量化类型.
    """
    pipeline = CalibrationPipeline()
    # Step1: Calculate activation calibration thresholds.
    pipeline.set(
        activation_calibration(
            quant_config.activation_config,
            quant_config.modelwise_search,
            quant_config.layerwise_search,
        )
    )

    if (
        quant_config.activation_config.get("calibration_type") != "mix"
        and quant_config.activation_config.get("calibration_type") != "min-max"
        and not (
            quant_config.activation_config.get("calibration_type") == "max"
            and quant_config.activation_config.get("block_size", 0) > 0
        )
    ):
        pipeline.set(ModelWiseSearch(quant_config.modelwise_search))

    pipeline.set(PostCalibration())
    if quant_config.activation_config.get("calibration_type") != "min-max":
        pipeline.set(ActivationEqualization())
    # Step2: Calculate weight calibration thresholds.
    pipeline.set(weight_calibration(quant_config.weight_config))
    if quant_config.weight_config.get("bias_correction"):
        pipeline.set(BiasCorrection(quant_config.weight_config))
    # Step3: Refine and modify quantization params.
    pipeline.set(AdjustEvenStrideDWS(quant_config.march))
    if quant_config.march in {"bayes", "bayes-e"}:
        pipeline.set(AdjustConvQuantParams())
    if quant_config.activation_config.get("calibration_type") != "min-max":
        pipeline.set(RefineThreshold())
    if quant_config.march == "bernoulli":
        pipeline.set(PowOfTwo())

    return pipeline
