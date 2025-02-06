# 模型简化工具, 可以将一个onnx模型进行结构简化, 生成一个简化后的模型
import argparse
import logging

from hmct.common import constant_folding, modify_model_by_cpp_func
from hmct.ir import load_model, save_model
from hmct.ir.horizon_onnx import quantizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model",
        "-om",
        type=str,
        required=True,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="simplified_model.onnx",
        type=str,
        help="Output onnx model file.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    if args.onnx_model is None:
        raise ValueError("The onnx model is not specified.")

    onnx_model = load_model(args.onnx_model)
    onnx_model = constant_folding(onnx_model)
    simplified_model = modify_model_by_cpp_func(onnx_model, quantizer.optimizer_tool)

    save_model(simplified_model, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
