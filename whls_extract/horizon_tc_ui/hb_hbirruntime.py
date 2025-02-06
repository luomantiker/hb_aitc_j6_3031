# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from hbdk4.compiler import load

from horizon_tc_ui.utils.tool_utils import get_hw_index


class HB_HBIRRuntime():
    def __init__(self, model_file: str, func_index: int = 0) -> None:
        if not os.path.exists(model_file):
            raise ValueError(f"{model_file} does not exist !!!")
        if not model_file.endswith('.bc'):
            raise ValueError(
                f"model {model_file} is invalid. Only models with .bc "
                "suffixes are supported")

        self.hbir_model = load(model_file)
        # self.sess = self.model.sess
        self.function = self.model.functions[func_index]
        self.node_output = {}

        self._input_types = None
        self._output_types = None
        self._layout = None
        self.current_phase = None
        self.check_current_phase()
        self.intermediate_outputs = {}

    @property
    def model(self):
        return self.hbir_model

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
        return [input.type.np_dtype for input in self.function.flatten_inputs]

    @property
    def output_types(self) -> List[np.dtype]:
        return [
            output.type.np_dtype for output in self.function.flatten_outputs
        ]

    @property
    def input_type_rt(self) -> List[str]:
        input_type_rts = self.desc.get("INPUT_TYPE_RT", "").split(";")
        if self.desc.get("SEPARATE_BATCH") == "True":
            input_batch = int(self.desc.get("INPUT_BATCH", "").split(";")[0])
            input_type_rts = input_type_rts * input_batch
        return input_type_rts

    @property
    def output_data_types(self) -> List[str]:
        if self._output_types:
            return self._output_types
        self._output_types = []
        for output in self.function.flatten_outputs:
            element_type = output.type.tensor.element_type.__str__()
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
        if self._layout:
            return self._layout
        self._layout = []
        for idx, input_type_rt in enumerate(self.input_type_rt):
            shape_item = self.input_shapes[idx]
            if len(shape_item) != 4 or input_type_rt == "featuremap":
                self._layout.append("")
                continue
            if shape_item[1] == 3 or shape_item[1] == 1:
                self._layout.append("NCHW")
                continue
            if shape_item[3] == 3 or shape_item[3] == 1:
                self._layout.append("NHWC")
                continue
            self._layout.append("")
        return self._layout

    @property
    def desc(self) -> dict:
        desc = {}
        try:
            desc = json.loads(self.function.desc)
        except Exception as e:
            logging.warning(f"Model desc parse failed, error log: {e}")
        finally:
            return desc

    def check_current_phase(self) -> None:
        for op in self.function.operations:
            if op.type == 'qnt.const_fake_quant':
                self.current_phase = 'export'

    def get_input_type(self) -> int:
        # TODO(wenhao.ma) Add check input type logic
        return 3

    def get_provider(self) -> str:
        return "CPUExecutionProvider"

    def get_hw(self, index: int = 0, layout: str = "") -> Tuple[int, int]:
        if index >= self.input_num:
            raise ValueError(
                f"wrong index: {index}. Model has {self.input_num} inputs")
        if not layout:
            layout = self.layout[index]
        h_index, w_index = get_hw_index(layout)
        return self.input_shapes[index][h_index], self.input_shapes[index][
            w_index]

    def get_input_index(self, input_name: str) -> int:
        for idx, name in enumerate(self.input_names):
            if name == input_name:
                return idx
        return -1

    def register_dump_nodes(self):
        def callback(op, results, operands):
            for idx, output in enumerate(op.outputs):
                if not output.name:
                    continue

                node_info = op.name + '__' + output.name
                self.intermediate_outputs.update({node_info: results[idx]})
            return True

        self.function.register_callback(callback)

    def feed(self, feed_dict: dict) -> Any:
        return self.function.feed(inputs=feed_dict)

    def run(self,
            output_names: Union[List[str], None] = None,
            input_info: Union[Dict[str, np.ndarray], None] = None,
            **extra_args: dict) -> List[np.ndarray]:
        """
        Executes the model using provided input data and returns the specified output data.

        This method allows for additional configuration through `extra_args`.

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
        deprecated_output_name = extra_args.get('output_name')
        if deprecated_output_name and not output_names:
            output_names = deprecated_output_name
            logging.warning("The output_name parameter has been deprecated, "
                            "please use output_names")
        elif deprecated_output_name and output_names:
            raise ValueError("The output_name parameter is deprecated, "
                             "do not use both output_name and output_names.")
        elif not deprecated_output_name and not output_names:
            output_names = self.output_names

        if not input_info:
            raise ValueError("Please provide input_info parameter.")

        result = self.function.feed(inputs=input_info)
        return [
            result[output_name] for output_name in output_names
            if output_name in self.output_names
        ]

    def run_direct(self,
                   output_names: Union[List[str], None] = None,
                   input_info: Union[Dict[str, np.ndarray], None] = None,
                   **extra_args: dict) -> List[np.ndarray]:
        """
        Directly executes the model with the provided input data and returns the specified outputs.

        This method bypasses any additional configurations or deprecated parameters, offering a more straightforward
        execution path. It also provides a warning if a deprecated parameter is used and raises an error if both
        deprecated and current parameters are provided together.

        Parameters:
        - output_names (Union[List[str], None]): A list of output names for which the output data is requested. If empty
                                    or not provided, all model outputs will be returned.
        - input_info (Dict[str, np.ndarray]): A dictionary mapping input names to their corresponding numpy
                                            array data.
        - **extra_args (dict): Optional additional arguments for future use.

        Returns:
        - List[np.ndarray]: A list of numpy arrays corresponding to the requested output names. The list's order
                            corresponds to the order of `output_names`.

        Raises:
        - ValueError: If both `output_name` (deprecated) and `output_names` are provided to avoid confusion.

        Note:
        - The `output_name` parameter within `extra_args` is deprecated. Users are encouraged to use `output_names`.
        - If no `output_names` are provided, the method defaults to using all output names from the model.
        """  # noqa

        deprecated_output_name = extra_args.get('output_name')
        if deprecated_output_name and not output_names:
            output_names = deprecated_output_name
            logging.warning("The output_name parameter has been deprecated, "
                            "please use output_names")
        elif deprecated_output_name and output_names:
            raise ValueError("The output_name parameter is deprecated, "
                             "do not use both output_name and output_names.")
        elif not deprecated_output_name and not output_names:
            output_names = self.output_names

        if not input_info:
            raise ValueError("Please provide input_info parameter.")

        result = self.function.feed(inputs=input_info)
        return [
            result[output_name] for output_name in output_names
            if output_name in self.output_names
        ]
