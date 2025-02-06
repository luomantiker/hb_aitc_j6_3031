# 模型兼容工具, 将低版本工具生成的模型转换成可在高版本工具环境下运行的模型
import argparse
import logging

from hmct.ir import OnnxModel, load_model, save_model


def convert_for_compatibility(onnx_model: OnnxModel) -> OnnxModel:
    """对给定模型做兼容性转换, 使其能够被当前版本horizon_nn运行."""
    onnx_model = add_horizon_domain(onnx_model)
    onnx_model = convert_mish_node(onnx_model)

    return onnx_model  # noqa: RET504


def add_horizon_domain(onnx_model: OnnxModel) -> OnnxModel:
    horizon_ops_exist = False
    for onnx_node in onnx_model.nodes:
        if onnx_node.op_type.startswith("Hz"):
            onnx_node.domain = "horizon"
            horizon_ops_exist = True
    # add horizon opset if horizon ops exist
    if horizon_ops_exist:
        onnx_model.opset_import["horizon"] = 1

    return onnx_model


def convert_mish_node(onnx_model: OnnxModel) -> OnnxModel:
    horizon_ops_exist = False
    for onnx_node in onnx_model.nodes:
        if onnx_node.op_type == "Mish":
            onnx_node.op_type = "HzMish"
            onnx_node.domain = "horizon"
            horizon_ops_exist = True
    # add horizon opset if horizon ops exist
    if horizon_ops_exist:
        onnx_model.opset_import["horizon"] = 1

    return onnx_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model",
        "-im",
        type=str,
        required=True,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--output_model",
        "-om",
        default="output_model.onnx",
        type=str,
        help="Output onnx model(.onnx) file.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    if args.onnx_model is None:
        raise ValueError("The input onnx model is not specified.")

    onnx_model = load_model(args.onnx_model)
    onnx_model = convert_for_compatibility(onnx_model)
    save_model(onnx_model, args.output_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
