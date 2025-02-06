import argparse
from collections import defaultdict
import logging
import os
from typing import Iterable, Optional, Union

from hmct.common import (
    Dataset,
    Loss,
    print_info_dict,
)
from hmct.executor import ORTExecutor
from hmct.ir import load_model


class ConsistencyChecker:
    """Compare the consistency between original_model and optimized_model."""

    def __init__(
        self,
        original_model: str,
        optimized_model: str,
        input_data: Optional[str] = None,
    ):
        self.original = load_model(original_model)
        self.optimized = load_model(optimized_model)
        self.input_data = self.prepare_input_data(input_data)
        self.add_output(self.original)
        self.add_output(self.optimized)

    def prepare_input_data(self, input_data: Optional[str] = None) -> Dataset:
        if isinstance(input_data, str):
            input_data = {
                name: os.path.join(input_data, name) for name in os.listdir(input_data)
            }

        if input_data is not None:
            return Dataset(input_data)
        return Dataset(
            input_shapes=self.original.graph.input_shapes,
            input_dtypes=self.original.graph.input_dtypes,
        )

    def add_output(self, model):
        conv_nodes = model.graph.type2nodes["Conv"]
        # 将Conv的输出添加到模型输出中
        for conv_node in conv_nodes:
            # 部分算子会在优化阶段被融合进Conv
            # TODO(jilei.hou): 这里还需要考虑Add/Sub等融合进Conv的情况
            op_type = ["BatchNormalization", "Relu", "Clip"]
            stage = 0
            node = conv_node
            while node:
                if len(node.next_ops) != 1:
                    current_output = node.outputs
                    if current_output[0] not in model.graph.outputs:
                        model.graph.append_output(current_output[0])
                    break
                if node.op_type in op_type[1:3]:
                    stage = 1
                if node.next_op.op_type in op_type[stage:]:
                    # The child node can be fused.
                    node = node.next_op
                else:
                    current_output = node.outputs
                    if current_output[0] not in model.graph.outputs:
                        model.graph.append_output(current_output[0])
                    break

    def similarity_info(
        self,
        metrics: Union[str, Iterable[str]] = ("cosine-similarity", "mse", "chebyshev"),
        decimal: int = 8,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        # forward model to get model outputs
        original_executor = ORTExecutor(self.original).create_session()
        optimized_executor = ORTExecutor(self.optimized).create_session()
        original_outputs = original_executor.inference(
            self.input_data.get_data(self.original.graph.input_names)
        )
        optimized_outputs = optimized_executor.inference(
            self.input_data.get_data(self.optimized.graph.input_names)
        )
        # collect similarity info
        similarity_info = defaultdict(dict)
        for name in original_outputs:
            if name not in optimized_outputs:
                continue
            for metric in metrics:
                similarity_info[name][metric] = Loss.create(metric).run(
                    original_outputs[name],
                    optimized_outputs[name],
                )
                similarity_info[name][metric] = round(
                    float(similarity_info[name][metric]),
                    decimal,
                )
        # visualize similarity info
        print_info_dict(similarity_info, title="similarity info", key_type="Output")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_model",
        "-ori",
        type=str,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--optimized_model",
        "-opt",
        type=str,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--input_data",
        "-i",
        type=str,
        help="Input data path with numpy binary file.",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    logging.info(
        f"original_model file: {args.original_model} \n"
        f"optimized_model file: {args.optimized_model} \n"
        f"input data path: {args.input_data}",
    )
    logging.info("Compare the consistency between original_model and optimized_model.")

    checker = ConsistencyChecker(
        args.original_model, args.optimized_model, args.input_data
    )
    checker.similarity_info()


if __name__ == "__main__":
    main()
