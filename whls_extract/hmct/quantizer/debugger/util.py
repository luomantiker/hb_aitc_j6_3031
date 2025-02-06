import os
from typing import List

from hmct.common import Dataset
from hmct.ir import load_model


def set_calibration_data(calibrated_data):
    processed_data = None
    if isinstance(calibrated_data, str):
        cali_data_dict = {}
        for input_name in os.listdir(calibrated_data):
            cali_data_dict[input_name] = calibrated_data + "/" + input_name
        processed_data = Dataset(input_data=cali_data_dict)
    elif isinstance(calibrated_data, Dataset):
        processed_data = calibrated_data
    return processed_data


def check_parameters(**kwargs):
    # calibrated_model和calibration_data的默认值设置为0
    # 目的是区别客户是否真的设置这两个参数

    # check calibrated_model
    calibrated_model = kwargs.get("model_or_file", 0)
    if calibrated_model is None:
        raise ValueError("There is no calibrated_model.")

    # check calibration_data
    calibration_data = kwargs.get("calibrated_data", 0)
    if calibration_data is None:
        raise ValueError("There is no calibration_data.")

    # check data_num
    data_num = kwargs.get("data_num", -1)
    if data_num != -1:
        processed_data = set_calibration_data(calibration_data)
        total_data_num = processed_data.number_of_samples
        if data_num is None:
            data_num = total_data_num
        if data_num > total_data_num:
            raise ValueError(
                f"There are only {total_data_num} data, but data_num "
                f"is set to {data_num}!"
            )

    # check nodes_list
    nodes_list = kwargs.get("nodes_list", None)
    if nodes_list is not None:
        if not isinstance(nodes_list, (str, List)):
            raise TypeError("nodes_list type must be str or list.")
        if isinstance(nodes_list, str):
            nodes_list = [nodes_list]
        elif isinstance(nodes_list, List) and all(
            isinstance(sublist, List) for sublist in nodes_list
        ):
            nodes_list = [item for sublist in nodes_list for item in sublist]

        model = load_model(calibrated_model)
        node_names = model.graph.node_names
        for node in nodes_list:
            if node not in node_names:
                raise ValueError(f"{node} does not exist in the model.")

    # check metrics
    metrics = kwargs.get("metrics", "cosine-similarity")
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if metric not in ["mse", "mre", "cosine-similarity", "sqnr", "chebyshev"]:
            raise ValueError(f"metrics contain unavailable method: {metric}")
