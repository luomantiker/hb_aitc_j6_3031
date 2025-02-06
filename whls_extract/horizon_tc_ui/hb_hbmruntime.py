# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
from typing import Dict, List, Union

import numpy as np

from horizon_tc_ui.hbm_handle import Hbm, HBMHandle


class HB_HBMRuntime:
    def __init__(self, model_file: str, func_index: int = 0) -> None:
        if not os.path.exists(model_file):
            raise ValueError(f"Input hbm model is not exists: {model_file}")
        if not model_file.endswith('.hbm'):
            raise ValueError(
                f"model {model_file} is invalid. Only models with .hbm "
                "suffixes are supported")

        self.hbm_handle = HBMHandle(hbm_path=model_file)
        model_graphs = self.hbm_handle.get_graphs()
        if len(model_graphs) > 1:
            raise ValueError(f'Input model {model_file} '
                             'is a pack model and not supported now')
        self.hbm_model = self.hbm_handle.model
        self.function = self.hbm_model.functions[func_index]

        self.model_name = self.function.name
        self.node_output = {}

        self._desc = None
        self._input_types = None
        self._output_types = None
        self._layout = None

    @property
    def model(self) -> Hbm:
        return self.hbm_model

    @property
    def input_num(self) -> int:
        return len(self.function.flatten_inputs)

    @property
    def output_num(self) -> int:
        return len(self.function.flatten_outputs)

    @property
    def input_names(self) -> List[str]:
        return [input.name for input in self.function.flatten_inputs]

    @property
    def output_names(self) -> List[str]:
        return [output.name for output in self.function.flatten_outputs]

    @property
    def input_types(self) -> List[np.dtype]:
        return [
            np.dtype(input.type.np_dtype)
            for input in self.function.flatten_inputs
        ]

    @property
    def output_types(self) -> List[np.dtype]:
        return [
            np.dtype(output.type.np_dtype)
            for output in self.function.flatten_outputs
        ]

    @property
    def output_data_types(self) -> List[str]:
        if self._output_types:
            return self._output_types
        self._output_types = []
        for output in self.function.flatten_outputs:
            element_type = output.type.np_dtype.type.__name__
            self._output_types.append(element_type)

        return self._output_types

    @property
    def input_shapes(self) -> List[list]:
        input_shapes = []
        for input in self.function.flatten_inputs:
            input_shape = list(input.type.shape)
            input_shapes.append(input_shape)
        return input_shapes

    @property
    def output_shapes(self) -> List[list]:
        output_shapes = []
        for output in self.function.flatten_outputs:
            output_shape = list(output.type.shape)
            output_shapes.append(output_shape)
        return output_shapes

    @property
    def layout(self) -> List[str]:
        return []

    @property
    def desc(self) -> dict:
        if not self._desc:
            _desc = self.hbm_handle.desc()
            try:
                self._desc = json.loads(
                    _desc["models"][self.model_name]["desc"])
            except Exception as e:
                logging.error(f"Failed to parse model desc: {e}")
                self._desc = {}
        return self._desc

    def run_arm(
        self,
        output_names: Union[List[str], None] = None,
        input_info: Union[Dict[str, np.ndarray], None] = None
    ) -> List[np.ndarray]:
        logging.warning("Hbm run arm is not supported now")
        return []

    def run_sim(
        self,
        output_names: Union[List[str], None] = None,
        input_info: Union[Dict[str, np.ndarray], None] = None
    ) -> List[np.ndarray]:
        if not output_names:
            output_names = self.output_names

        if not input_info:
            raise ValueError("Please provide input_info parameter.")

        result = self.hbm_handle.model[0].feed(feed_dict=input_info)
        return [
            result[output_name] for output_name in output_names
            if output_name in self.output_names
        ]

    def run(self,
            output_names: Union[List[str], None] = None,
            input_info: Union[Dict[str, np.ndarray], None] = None,
            **extra_args: dict) -> List[np.ndarray]:
        """
        Executes the model using provided input data and returns the specified output data.

        This method allows for additional configuration through `extra_args` and supports
        a `board_ip` parameter for inference on board.

        Parameters:
        - output_names (Union[List[str], None]): A list of output names for which the output data is requested.
                                    If empty or not provided, all model outputs will be returned.
        - input_info (Dict[str, np.ndarray]): A dictionary mapping input names to their corresponding
                                            numpy array data.
        - **extra_args (dict): Optional additional arguments for future use or for maintaining backward compatibility.

        Returns:
        - List[np.ndarray]: A list of numpy arrays corresponding to the requested output names. The order
                            of the outputs in the list matches the order of `output_names`.

        Raises:
        - ValueError: If both `output_name` (deprecated) and `output_names` are provided, to avoid confusion.

        Note:
        - The `output_name` parameter is deprecated and will be removed in future versions. Please use `output_names`.
        """  # noqa

        if extra_args.get("board_ip"):
            return self.run_arm(output_names=output_names,
                                input_info=input_info)
        else:
            return self.run_sim(output_names=output_names,
                                input_info=input_info)

    def run_direct(self, *args, **kwargs) -> List[np.ndarray]:
        return self.run(*args, **kwargs)
