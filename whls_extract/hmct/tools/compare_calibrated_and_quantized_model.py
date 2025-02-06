import argparse
from collections import defaultdict
import logging
import os
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from hmct.common import (
    Dataset,
    Loss,
    print_info_dict,
)
from hmct.executor import ORTExecutor
from hmct.ir import DataType, OnnxVariable, load_model


def convert_input_datas_by_graph_inputs(
    input_data: Dict[str, np.ndarray], graph_inputs: Tuple[OnnxVariable, ...]
) -> Dict[str, np.ndarray]:
    """Convert the data layout to match graph inputs."""
    for input_node in graph_inputs:
        org_shape = input_data[input_node.name].shape
        if len(org_shape) == 4 and input_node.dtype != DataType.FLOAT32:
            input_shape = list(org_shape[1:])
            expected_shape = list(input_node.shape[1:])
            if [
                input_shape[1],
                input_shape[2],
                input_shape[0],
            ] == expected_shape:
                # nchw to nhwc
                input_data[input_node.name] = input_data[input_node.name].transpose(
                    [0, 2, 3, 1]
                )
            elif [
                input_shape[2],
                input_shape[0],
                input_shape[1],
            ] == expected_shape:
                # nhwc to nchw
                input_data[input_node.name] = input_data[input_node.name].transpose(
                    [0, 3, 1, 2]
                )
            else:
                pass
            if input_node.dtype == DataType.INT8:
                input_data[input_node.name] = input_data[input_node.name].astype(
                    np.int8
                )

    return input_data


class ConsistencyChecker:
    """Compare the consistency between calibrated_model and quantized_model."""

    def __init__(
        self,
        calibrated_model: str,
        quantized_model: str,
        input_data: Optional[str] = None,
    ):
        self.calibrated_model = load_model(calibrated_model)
        self.quantized_model = load_model(quantized_model)
        self.input_data = self.prepare_input_data(input_data)

    def prepare_input_data(self, input_data: Optional[str] = None) -> Dataset:
        if isinstance(input_data, str):
            input_data = {
                name: os.path.join(input_data, name) for name in os.listdir(input_data)
            }

        if input_data is not None:
            return Dataset(input_data)
        return Dataset(
            input_shapes=self.calibrated_model.graph.input_shapes,
            input_dtypes=self.calibrated_model.graph.input_dtypes,
        )

    def similarity_info(
        self,
        metrics: Union[str, Iterable[str]] = ("cosine-similarity", "mse", "chebyshev"),
        decimal: int = 8,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        # forward model to get model outputs
        calibrated_model_executor = ORTExecutor(self.calibrated_model).create_session()
        quantized_model_executor = ORTExecutor(self.quantized_model).create_session()
        calibrated_model_outputs = calibrated_model_executor.inference(
            convert_input_datas_by_graph_inputs(
                self.input_data.get_data(self.calibrated_model.graph.input_names),
                self.calibrated_model.graph.inputs,
            )
        )
        quantized_model_outputs = quantized_model_executor.inference(
            convert_input_datas_by_graph_inputs(
                self.input_data.get_data(self.quantized_model.graph.input_names),
                self.quantized_model.graph.inputs,
            )
        )

        # collect similarity info
        similarity_info = defaultdict(dict)
        for name in calibrated_model_outputs:
            if name not in quantized_model_outputs:
                continue
            for metric in metrics:
                similarity_info[name][metric] = Loss.create(metric).run(
                    calibrated_model_outputs[name],
                    quantized_model_outputs[name],
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
        "--calibrated_model",
        "-c",
        type=str,
        help="Input calibrated model(.onnx) file.",
    )
    parser.add_argument(
        "--quantized_model",
        "-q",
        type=str,
        help="Input quantized model(.onnx) file.",
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
        f"calibrated model file: {args.calibrated_model} \n"
        f"quantized model file: {args.quantized_model} \n"
        f"input data path: {args.input_data}",
    )
    logging.info(
        "Compare the consistency between calibrated_model and quantized_model."
    )

    checker = ConsistencyChecker(
        args.calibrated_model,
        args.quantized_model,
        args.input_data,
    )
    checker.similarity_info()


if __name__ == "__main__":
    main()
