import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from hmct.converter import parse
from hmct.ir import OnnxModel

from .model_builder import BuildInfo, ModelBuilder

if TYPE_CHECKING:
    import numpy as np

    from hmct.ir.onnx_utils import ModelProto

# ========================================================
# ---- code below are exported model build interfaces ----
# ========================================================


def build_onnx(
    onnx_file,
    march,
    name_prefix="",
    input_dict=None,
    cali_dict=None,
    hbdk_dict=None,
    output_nodes=None,
    node_dict=None,
    save_model=True,
    debug_mode=None,
    optimization=None,
    quant_config=None,
    **kwargs,
) -> Union[ModelBuilder, "ModelProto", None]:
    """Build hybrid model from onnx model.

    This function will convert onnx model to hybrid model.

    Args:
        onnx_file: File path or content of onnx model.
        march: Architecture of BPU. Available values include
            ['bayes', 'bernoulli2', 'bernoulli']
        name_prefix: The output model name or path prefix.
        input_dict: This is a dict param including input parameters.
            Its keys are names of input nodes, and values are also dicts
            contain the paired parameters.
        cali_dict: This is a dict param including calibration related
            parameters.
        hbdk_dict: This is a dict param including compiler parameters.
            All the parameters will be transparently transmitted to the compiler.
        output_nodes: This is a list of specify model output node names.
        node_dict: This is a dict param including node related
            parameters. Its keys are the name of affected nodes, and values
            are lists of the action on all affected nodes.
        save_model: Whether to save model generated during the building process.
            True to save and False not.
        debug_mode: A list to specify debug options during model compilation.
            Currently supported options are shown in
            horizon_nn/common/parser/model_debugger.py
        optimization: A sequence to specify optimization options
            for model convert. These options can improve the performance or
            accuracy of the deployment model. The supported options are
            shown in onnx/onnx/quantization/optimization_configs.h
        quant_config: This is path of a json file including quantization
            related parameters.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        model builder if return_builder is True, hybrid model otherwise.
    """
    _ = BuildInfo("Horizon NN Model Convert")
    return_builder = kwargs.get("return_builder", False)
    if kwargs.get("enable_int16", False):
        logging.error(
            "The configuration of enable_int16 has been deprecated, "
            "please using node_dict to specify int16 configuration.",
        )
    onnx_model = parse(model_type="onnx", onnx_model_or_proto=onnx_file)
    model_builder = ModelBuilder(
        onnx_model=onnx_model,
        march=march,
        return_hybrid_model=True,
        save_model=save_model,
        name_prefix=name_prefix,
        output_nodes=output_nodes,
        debug_mode=debug_mode,
        optimization=optimization,
        cali_dict=cali_dict,
        node_dict=node_dict,
        hbdk_dict=hbdk_dict,
        input_dict=input_dict,
        user_quant_config=quant_config,
        **kwargs,
    )
    hybrid_model = model_builder.build()

    if return_builder is False:
        return hybrid_model

    return model_builder


def build_caffe(
    prototxt_file,
    caffemodel_file,
    march,
    name_prefix="",
    input_dict=None,
    cali_dict=None,
    hbdk_dict=None,
    output_nodes=None,
    node_dict=None,
    save_model=True,
    debug_mode=None,
    optimization=None,
    quant_config=None,
    **kwargs,
) -> Union[ModelBuilder, "ModelProto", None]:
    """Build hybrid model from caffe model.

    This function will convert caffe model to hybrid model.

    Args:
        prototxt_file: File of caffe prototxt.
        caffemodel_file: File of caffe model.
        march: Architecture of BPU. Available values include
            ['bayes', 'bernoulli2', 'bernoulli']
        name_prefix: The output model name or path prefix.
        input_dict: This is a dict param including input parameters.
            Its keys are names of input nodes, and values are also dicts
            contain the paired parameters.
        cali_dict: This is a dict param including calibration related
            parameters.
        hbdk_dict: This is a dict param including compiler parameters.
            All the parameters will be transparently transmitted to the compiler.
        output_nodes: This is a list of specify model output node names.
        node_dict: This is a dict param including node related
            parameters. Its keys are the name of affected nodes, and values
            are lists of the action on all affected nodes.
        save_model: Whether to save model generated during the building process.
            True to save and False not.
        debug_mode: A list to specify debug options during model compilation.
            Currently supported options are shown in
            horizon_nn/common/parser/model_debugger.py
        optimization: A sequence to specify optimization options
            for model convert. These options can improve the performance or
            accuracy of the deployment model. The supported options are
            shown in onnx/onnx/quantization/optimization_configs.h
        quant_config: This is path of a json file including quantization
            related parameters.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        model builder if return_builder is True, hybrid model otherwise.
    """
    onnx_model = parse(
        model_type="caffe",
        prototxt_file=prototxt_file,
        caffemodel_file=caffemodel_file,
    )

    return build_onnx(
        onnx_model,
        march=march,
        name_prefix=name_prefix,
        input_dict=input_dict,
        cali_dict=cali_dict,
        hbdk_dict=hbdk_dict,
        output_nodes=output_nodes,
        node_dict=node_dict,
        save_model=save_model,
        debug_mode=debug_mode,
        optimization=optimization,
        quant_config=quant_config,
        **kwargs,
    )


def build_model(
    onnx_model: Optional["ModelProto"] = None,
    march: str = "nash",
    cali_data: Optional[
        Union[Sequence["np.ndarray"], Dict[str, Sequence["np.ndarray"]]]
    ] = None,
    quant_config: Optional[Union[str, Dict[str, Any]]] = None,
    input_dict: Optional[Dict[str, Any]] = None,
    name_prefix: Optional[str] = None,
    verbose: Optional[bool] = True,
    **kwargs,
) -> Union[ModelBuilder, "ModelProto", None]:
    """Build model based on the provided input parameters.

    Model quantization (except HzLut) and compilation are no longer supported
    since hbdk4.
    See: https://horizonrobotics.feishu.cn/wiki/MpbuwZoIyiZmpEkLFGOcWxzrnyb

    Args:
        onnx_model: ONNX model object.
        cali_data: Calibration data.
        march: Architecture of BPU. Available values include ['nash', bayes'].
        onnx_file: Path to the ONNX model file.
        prototxt_file: Path to the prototxt file.
        caffemodel_file: Path to the caffemodel file.
        name_prefix: The output model name prefix.
        input_dict: A dict param including model input related parameters.
            Its keys are names of input nodes, and values are also dicts
            contain the paired parameters.
        cali_dict: A dict param including calibration related parameters.
        output_nodes: A list of specified model output node names,
            original output will be replaced.
        node_dict: A dict param including node related parameters.
            Its keys are the name of affected nodes, and values are
            lists of the action on all affected nodes.
        debug_methods: A list to specify debug methods during model compilation.
        optimization_methods: A list of optimization methods for model compilation.
        quant_config: This is path of a json file including quantization
            related parameters.
        return_builder: Whether to return the whole builder or just one model.
        check_mode: True to run in check mode, use random data for calibration.
            False to run in regular mode, use real data given by user.
        verbose: Whether to print model quant info after model built.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        model builder if return_builder is True, ptq model otherwise.
    """
    _ = BuildInfo("Horizon NN Model Convert")

    onnx_file = kwargs.pop("onnx_file", None)
    prototxt_file = kwargs.pop("prototxt_file", None)
    caffemodel_file = kwargs.pop("caffemodel_file", None)
    cali_dict = kwargs.pop("cali_dict", None)
    return_builder = kwargs.pop("return_builder", False)
    debug_methods = kwargs.pop("debug_methods", None)
    optimization_methods = kwargs.pop("optimization_methods", None)
    skip_step = kwargs.pop("skip_step", None)

    if onnx_model is None:
        if onnx_file is not None:
            onnx_model = parse(model_type="onnx", onnx_model_or_proto=onnx_file)
        elif prototxt_file is not None and caffemodel_file is not None:
            onnx_model = parse(
                model_type="caffe",
                prototxt_file=prototxt_file,
                caffemodel_file=caffemodel_file,
            )
    else:
        onnx_model = OnnxModel(onnx_model)

    if march == "nash-e" or march == "nash-m":
        march = "nash"

    # 将Python API的cali_data转成字典格式
    if cali_data is not None and cali_dict is None:
        cali_dict = {}
        if not isinstance(cali_data, Dict):
            assert len(onnx_model.graph.inputs) == 1
            cali_dict["calibration_data"] = {}
            cali_dict["calibration_data"][onnx_model.graph.inputs[0].name] = cali_data
        else:
            cali_dict["calibration_data"] = cali_data

    # 将name_prefix的默认值设为""
    if name_prefix is None:
        name_prefix = ""

    model_builder = ModelBuilder(
        onnx_model=onnx_model,
        march=march,
        return_hybrid_model=False,
        save_model=True,
        verbose=verbose,
        name_prefix=name_prefix,
        debug_mode=debug_methods,
        optimization=optimization_methods,
        skip_step=skip_step,
        cali_dict=cali_dict,
        input_dict=input_dict,
        user_quant_config=quant_config,
        **kwargs,
    )
    ptq_model = model_builder.build()

    if return_builder is False:
        return ptq_model

    return model_builder


def check_model(
    onnx_model: "ModelProto",
    march: str,
    input_dict: Optional[Dict[str, Any]] = None,
) -> "ModelProto":
    """使用随机参数快速验证模型转换流程是否成功.

    Args:
        onnx_model: onnx模型对象.
        march: 芯片架构.
        input_dict: 输入相关参数的字典.

    Returns:
        使用随机参数经过模型转换流程后的quant_model.
    """
    return build_model(
        march=march,
        onnx_model=onnx_model,
        input_dict=input_dict,
        return_builder=False,
        check_mode=True,
    )
