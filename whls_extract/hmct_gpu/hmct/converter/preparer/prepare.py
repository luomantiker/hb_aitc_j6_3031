import logging
from typing import TYPE_CHECKING, Sequence

from hmct.version import __version__

from .check_for_convert import check_for_convert
from .preprocess_for_convert import preprocess_for_convert

if TYPE_CHECKING:
    from hmct.common import InputDictParser
    from hmct.ir import OnnxModel


def prepare(
    onnx_model: "OnnxModel",
    input_dict_parser: "InputDictParser",
    op_convert: bool,
    opset_version: int,
    output_nodes: Sequence[str],
) -> "OnnxModel":
    """Prepare onnx model for model convert.

    Returns:
        Original onnx model after preparation.
    """
    # display onnx_model basic info
    display_model_infos(onnx_model)
    # check onnx opset version
    check_opset_version(onnx_model=onnx_model, op_convert=op_convert)

    # preprocess onnx model for model convert
    onnx_model = preprocess_for_convert(
        onnx_model=onnx_model,
        input_dict_parser=input_dict_parser,
        output_nodes=output_nodes,
    )

    if op_convert:
        onnx_model = convert_opset_version(onnx_model, opset_version)

    # check onnx model for model convert
    check_for_convert(
        onnx_model=onnx_model,
        input_batch_dict=input_dict_parser.get_input_batches(),
    )

    return onnx_model


def display_model_infos(onnx_model: "OnnxModel") -> None:
    info_str = []
    info_str.extend(
        (
            "Input ONNX Model Information:",
            f"ONNX IR version:          {onnx_model.ir_version}",
        )
    )
    # collect opset infos
    opset_infos = []
    for domain, version in onnx_model.opset_import.items():
        domain = "ai.onnx" if domain == "" else domain
        opset_info = domain + " v" + str(version)
        opset_infos.append(opset_info)
    info_str.append(f"Opset version:            {opset_infos}")
    # collect producer info
    if onnx_model.producer_name is None:
        producer_info = None
    else:
        producer_info = onnx_model.producer_name
        if onnx_model.producer_version is not None:
            producer_info = producer_info + " v" + onnx_model.producer_version
    info_str.extend(
        (
            f"Producer:                 {producer_info}",
            f"Domain:                   {onnx_model.model_domain}",
            f"Version:                  {onnx_model.model_version}",
        )
    )
    # collect graph outputs info
    info_str.append("Graph input:")
    for input_var in onnx_model.graph.inputs:
        input_name = input_var.name
        input_shape = input_var.shape
        input_dtype = (
            input_var.dtype.name if input_var.dtype is not None else input_var.dtype
        )
        if len(input_name) > 10:
            # avoid the input name too long
            input_name = "..." + input_name[-8:]
        info_str.append(
            "    {:<21} shape={}, dtype={}".format(
                input_name + ":",
                input_shape,
                input_dtype,
            ),
        )
    # collect graph outputs info
    info_str.append("Graph output:")
    for output_var in onnx_model.graph.outputs:
        output_name = output_var.name
        output_shape = output_var.shape
        output_dtype = (
            output_var.dtype.name if output_var.dtype is not None else output_var.dtype
        )
        if len(output_name) > 10:
            # avoid the output name too long
            output_name = "..." + output_name[-8:]
        info_str.append(
            "    {:<21} shape={}, dtype={}".format(
                output_name + ":",
                output_shape,
                output_dtype,
            ),
        )

    logging.info("\n".join(info_str))


def check_opset_version(onnx_model: "OnnxModel", op_convert: bool) -> None:
    domain = "" if "" in onnx_model.opset_import else "ai.onnx"
    opset_version = onnx_model.opset_import[domain]
    opset_supported = (10, 11) if int(__version__[0]) == 1 else (10, 20)

    if opset_version < opset_supported[0] and not op_convert:
        raise ValueError(
            f"The opset version of the model is {opset_version}, "
            f"the minimum supported version is {opset_supported[0]}."
        )
    if opset_version > opset_supported[1]:
        raise ValueError(
            f"The opset version of the model is {opset_version}, "
            f"the maximum supported version is {opset_supported[1]}."
        )


def convert_opset_version(onnx_model: "OnnxModel", target_version: int) -> "OnnxModel":
    domain = "" if "" in onnx_model.opset_import else "ai.onnx"
    opset_version = onnx_model.opset_import[domain]

    if opset_version < target_version:
        logging.info(
            f"The original model's opset version is {opset_version}, "
            f"try converting to opset {target_version}. "
        )
        onnx_model = onnx_model.convert_version(target_version=target_version)
    return onnx_model
