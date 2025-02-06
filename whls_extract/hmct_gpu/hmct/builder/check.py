import argparse
import logging
from typing import Union

from hmct.converter import parse

from .model_builder import BuildInfo, ModelBuilder

# ========================================================
# ---- Code below are exported model check interfaces ----
# ========================================================


def check_onnx(
    onnx_file,
    march,
    name_prefix="",
    input_dict=None,
    node_dict=None,
    save_model=False,
    **kwargs,
) -> Union["ModelProto", None]:  # noqa: F821
    """Check hybrid model building process from onnx model.

    Check process will use fake calibration method in the building process.
    So it's faster than build and is used for quick verification.

    Args:
        onnx_file: File path or content of onnx model.
        march: Architecture of BPU. Avaliable values include
            ['bayes', 'bernoulli2', 'bernoulli']
        name_prefix: The output model name or path prefix.
        input_dict: This is a dict param including input parameters.
            Its keys are names of input nodes, and values are also dicts
            contain the paired parameters.
        node_dict: This is a dict param including node related
            parameters. Its keys are the name of affected nodes, and
            values are dict of the actions on the corresponding node.
        save_model: Whether to save model generated during the
            check process. True to save and False not.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        model builder if return_builder is True, hybrid model otherwise.
    """
    _ = BuildInfo("Horizon NN Model Convert")
    cali_dict = kwargs.get("cali_dict", None)

    onnx_model = parse(model_type="onnx", onnx_model_or_proto=onnx_file)
    model_builder = ModelBuilder(
        onnx_model=onnx_model,
        march=march,
        check_mode=True,
        save_model=save_model,
        name_prefix=name_prefix,
        cali_dict=cali_dict,
        node_dict=node_dict,
        input_dict=input_dict,
    )

    return model_builder.build()


def check_caffe(
    prototxt_file,
    caffemodel_file,
    march,
    **kwargs,
) -> Union["ModelProto", None]:  # noqa: F821
    """Check hybrid model building process from caffe model.

    Check process will use fake calibration method in the building process.
    So it's faster than build and is used for quick verification.

    Args:
        prototxt_file: File of caffe prototxt.
        caffemodel_file: File of caffe model.
        march: Architecture of BPU. Avaliable value is
            ['bayes', 'bernoulli2', 'bernoulli']
        name_prefix: The output model name or path prefix.
        input_dict: This is a dict param including input parameters.
            Its keys are names of input nodes, and values are also dicts
            contain the paired parameters.
        save_model: Whether to save model generated during the
            check process. True to save and False not.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        model builder if return_builder is True, hybrid model otherwise.
    """
    kwargs["op_convert"] = True
    onnx_model = parse(
        model_type="caffe",
        prototxt_file=prototxt_file,
        caffemodel_file=caffemodel_file,
    )

    return check_onnx(onnx_model, march=march, **kwargs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        "-f",
        type=str,
        choices=["caffe", "onnx"],
        help="Neural network model framework.",
        required=True,
    )
    parser.add_argument(
        "--caffe_proto",
        "-cp",
        type=str,
        help="Input caffe model proto(.prototxt) file.",
    )
    parser.add_argument(
        "--caffe_model",
        "-cm",
        type=str,
        help="Input caffe model(.caffemodel) file.",
    )
    parser.add_argument(
        "--onnx_model",
        "-om",
        type=str,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--march",
        "-m",
        type=str,
        choices=["bernoulli", "bernoulli2", "bayes"],
        help="Target BPU micro architecture. Supported march: "
        "bernoulli; bernoulli2; bayes.",
        required=True,
    )
    parser.add_argument(
        "--calibration_type",
        "-t",
        type=str,
        choices=["max", "kl", "load"],
        help="Specify the calibration type.",
    )
    parser.add_argument(
        "--inputs",
        "-i",
        type=str,
        help="Names of the inputs, comma-separated.",
    )
    parser.add_argument(
        "--input_shapes",
        "-is",
        type=str,
        help="Shapes corresponding to inputs, colon-separated.",
    )
    parser.add_argument(
        "--op_convert",
        default=False,
        action="store_true",
        help="Convert the onnx model opset version.",
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action="store_true",
        help="Save intermediate model.",
    )

    return parser.parse_args()


def create_input_dict(input_names, input_shapes):
    if input_names is None or input_shapes is None:
        return None

    if not isinstance(input_shapes, list) or not isinstance(input_shapes[0], list):
        raise ValueError(
            "input_shapes and the value of input_shapes must be type of list",
        )
    if len(input_shapes) != len(input_names):
        raise RuntimeError(
            "Argument input_shapes and input_names are not equal "
            "in length. Or input_names is not specified, "
            "and the number of input node int the model does not "
            "match the length of input_shapes.",
        )

    input_dict = {}
    for name, shape in zip(input_names, input_shapes):
        input_dict[name] = {"input_shape": shape}

    return input_dict


def main():
    # parse arguments
    args = get_args()

    # parse input names and shapes
    input_names, input_shapes = None, None
    if args.inputs:
        input_names = [str(name) for name in args.inputs.split(",") if name]
    if args.input_shapes:
        input_shapes = [
            [int(val) for val in shape.split(",") if val]
            for shape in args.input_shapes.split(":")
        ]
    input_dict = create_input_dict(input_names, input_shapes)

    # parse calibration info
    cali_dict = None
    if args.calibration_type:
        cali_dict = {"calibration_type": args.calibration_type}

    if args.framework == "caffe":
        if args.caffe_proto is None or args.caffe_model is None:
            raise ValueError("The caffe proto or caffe model is not specified.")
        check_caffe(
            args.caffe_proto,
            args.caffe_model,
            args.march,
            save_model=args.save_model,
            input_dict=input_dict,
            cali_dict=cali_dict,
        )

    elif args.framework == "onnx":
        if args.onnx_model is None:
            raise ValueError("The onnx model is not specified.")
        check_onnx(
            args.onnx_model,
            args.march,
            save_model=args.save_model,
            input_dict=input_dict,
            op_convert=args.op_convert,
            cali_dict=cali_dict,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
