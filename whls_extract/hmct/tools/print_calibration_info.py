import argparse
import logging

import numpy as np

from hmct.ir import load_model
from hmct.reporter import (
    calculate_quant_type,
    calculate_similarity,
    print_model_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cosine similarity calculation tool.")
    parser.add_argument("--version", action="version", version="0.0.1")
    parser.add_argument(
        "calibration_model",
        type=str,
        help="The calibration model argument.",
    )
    parser.add_argument(
        "--calibration_data",
        "-c",
        type=str,
        action="append",
        required=True,
        help=(
            "Numpy binary file with data type float32, "
            "calibration data for cosine similarity calculation."
        ),
    )
    return parser.parse_args()


def cmd_main():
    """Main function to perform cosine similarity calculation.

    It loads the calibration model and calibration data,
    then calculates and prints the cosine similarity of model nodes.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    calibration_data = args.calibration_data
    calibration_model = args.calibration_model
    logging.info(
        f"model file: {calibration_model}, calibration data: {calibration_data}",
    )
    model = load_model(calibration_model)
    input_names = model.graph.input_names
    if len(calibration_data) != len(input_names):
        raise ValueError(
            f"number of calibration data provided: {len(calibration_data)} "
            f"does not match number of model input {len(input_names)}",
        )

    data_dict = {
        name: np.load(cal_data) for name, cal_data in zip(input_names, calibration_data)
    }

    model_quant_type_info = calculate_quant_type(model)
    node_similarity_info, output_similarity_info = calculate_similarity(
        calibrated_model=model,
        calibration_data_dict=data_dict,
    )
    print_model_info(
        node_similarity_info=node_similarity_info,
        output_similarity_info=output_similarity_info,
        model_quant_type_info=model_quant_type_info,
    )


if __name__ == "__main__":
    cmd_main()
