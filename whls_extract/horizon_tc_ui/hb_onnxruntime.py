# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import copy
import logging
from typing import Dict, List, Union

import numpy as np
import onnx
from onnx import TensorProto
from horizon_nn.api import ORTExecutor

from horizon_tc_ui.utils.tool_utils import get_hw_index


class HB_ONNXRuntime():
    def __init__(self,
                 model_file=None,
                 onnx_model=None,
                 input_batch=None) -> None:
        if model_file is None and onnx_model is None:
            raise ValueError("please provide either model_file or onnx_model")
        if onnx_model is None:
            self.onnx_model = onnx.load(model_file)
        else:
            self.onnx_model = onnx_model

        if input_batch is not None:
            logging.info('If you need to modify the batch dimension, '
                         'please manually call the set_dim_param '
                         'method to update its value.')

        self.executor = ORTExecutor(self.onnx_model).create_session()
        self._input_shapes = None
        self._input_names = None
        self._input_types = None
        self._input_num = None
        self._output_num = None
        self._output_shapes = None
        self._output_names = None
        self._output_types = None
        self._layout = None
        self._desc = {}
        self.original_input_shapes = copy.deepcopy(self.input_shapes)

        self.dtype_map = {
            TensorProto.DataType.FLOAT: "float32",
            TensorProto.DataType.UINT8: "uint8",
            TensorProto.DataType.INT8: "int8",
            TensorProto.DataType.UINT16: "uint16",
            TensorProto.DataType.INT16: "int16",
            TensorProto.DataType.INT32: "int32",
            TensorProto.DataType.INT64: "int64",
            TensorProto.DataType.STRING: "string",
            TensorProto.DataType.BOOL: "bool",
            TensorProto.DataType.FLOAT16: "float16",
            TensorProto.DataType.DOUBLE: "float64",
            TensorProto.DataType.UINT32: "uint32",
            TensorProto.DataType.UINT64: "uint64",
            TensorProto.DataType.BFLOAT16: "bfloat16",
        }
        self.func_map = {
            TensorProto.DataType.FLOAT: self.data_pre_to_float,
            TensorProto.DataType.UINT8: self.data_pre_to_uint8,
            TensorProto.DataType.INT8: self.data_pre_to_int8,
        }

    @property
    def model(self):
        return self.onnx_model

    @property
    def inputs(self) -> list:
        return self.executor.get_inputs()

    @property
    def outputs(self) -> list:
        return self.executor.get_outputs()

    @property
    def input_names(self):
        if self._input_names is not None:
            return self._input_names
        self._input_names = []
        for input_item in self.inputs:
            self._input_names.append(input_item.name)
        return self._input_names

    @property
    def input_shapes(self):
        if self._input_shapes is not None:
            return self._input_shapes
        self._input_shapes = []
        for input_item in self.inputs:
            shape_item = []
            for shape_dim in input_item.shape:
                shape_item.append(shape_dim)
            self._input_shapes.append(shape_item)
        return self._input_shapes

    @property
    def input_types(self) -> List[np.dtype]:
        if self._input_types is not None:
            return self._input_types
        self._input_types = []
        for input_item in self.inputs:
            self._input_types.append(np.dtype(self.dtype_map[input_item.type]))
        return self._input_types

    @property
    def output_names(self):
        if self._output_names is not None:
            return self._output_names
        self._output_names = []
        for output_item in self.outputs:
            self._output_names.append(output_item.name)
        return self._output_names

    @property
    def output_shapes(self):
        if self._output_shapes is not None:
            return self._output_shapes
        self._output_shapes = []
        for output_item in self.outputs:
            shape_item = []
            for shape_dim in output_item.shape:
                shape_item.append(shape_dim)
            self._output_shapes.append(shape_item)
        return self._output_shapes

    @property
    def output_types(self):
        if self._output_types is not None:
            return self._output_types
        self._output_types = []
        for output_item in self.outputs:
            self._output_types.append(output_item.type)
        return self._output_types

    @property
    def layout(self) -> List[str]:
        if self._layout is not None:
            return self._layout
        self._layout = []
        for shape_item in self.input_shapes:
            if len(shape_item) != 4:
                self._layout.append("")
                continue
            elif shape_item[0] != 1:
                self._layout.append("")
                continue
            elif shape_item[1] != 3 and shape_item[3] != 3:
                self._layout.append("")
                continue
            elif shape_item[1] == 3 and (shape_item[2] % 2 == 0
                                         and shape_item[3] % 2 == 0):  # noqa
                self._layout.append("NCHW")
                continue
            elif shape_item[3] == 3 and (shape_item[1] % 2 == 0
                                         and shape_item[2] % 2 == 0):  # noqa
                self._layout.append("NHWC")
                continue
            else:
                self._layout.append("")
        return self._layout

    @property
    def input_num(self):
        if self._input_num is not None:
            return self._input_num
        self._input_num = len(self.input_names)
        return self._input_num

    @property
    def output_num(self):
        if self._output_num is not None:
            return self._output_num
        self._output_num = len(self.output_names)
        return self._output_num

    @property
    def desc(self):
        return self._desc

    def get_model(self):
        return self.onnx_model

    def get_inputs(self):
        return self.executor.get_inputs()

    def get_input_type(self, index=0):
        if index > len(self.onnx_model.graph.input):
            raise ValueError(f"onnx_model does not have input {index} !!!")
        return self.onnx_model.graph.input[index].type.tensor_type.elem_type

    def get_outputs(self):
        return self.executor.get_outputs()

    def get_hw(self, index=0, layout=None):
        if index >= self.input_num:
            raise ValueError(
                f"wrong index: {index}. Model has {self.input_num} inputs")
        if not layout:
            layout = self.layout[index]
        h_index, w_index = get_hw_index(layout)
        return self.input_shapes[index][h_index], self.input_shapes[index][
            w_index]

    def get_input_shape(self, layout=None, *, input_index=0):
        return self.input_shapes

    def set_providers(self, providers):
        self.executor.to(providers)

    def get_provider(self):
        if 'cuda' in ORTExecutor.get_support_devices():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"

    def set_dim_param(self, input_id: int, index_id: int,
                      expect_val: Union[int, str]) -> None:
        _input = self.onnx_model.graph.input[input_id]
        if isinstance(expect_val, int):
            _input.type.tensor_type.shape.dim[index_id].dim_value = expect_val
        else:
            _input.type.tensor_type.shape.dim[index_id].dim_param = expect_val
        self._input_shapes = None
        self._output_shapes = None
        self.executor = ORTExecutor(self.onnx_model).create_session()

    def data_pre_to_float(self, input_data, name):
        # TODO(ruxin.song): Compatibility only
        dtype_supported = ["uint8", "int8", "float32"]
        if input_data.dtype not in dtype_supported:
            raise ValueError(
                f"input[{name}] model input type is float, only support to "
                f"provide {dtype_supported} type input,"
                f"input illegal: {input_data.dtype}")

        if input_data.dtype in ["uint8", "int8"]:
            logging.warning(
                f"input[{name}] model input type is float, input data type is"
                f" {input_data.dtype}, will be convert.")
            input_data = input_data.astype(np.float32)

        if hasattr(self.onnx_model.graph, 'nodes'):
            nodes = self.onnx_model.graph.nodes
        else:
            nodes = self.onnx_model.graph.node

        if input_data.dtype in ["uint8", "float32"]:
            hz_preprocess_exist = any(node.op_type == "HzPreprocess"
                                      for node in nodes)
            if hz_preprocess_exist:
                input_data = input_data - 128.0

        return input_data

    def data_pre_to_uint8(self, input_data, name):
        dtype_supported = ["uint8", "int8"]
        if input_data.dtype not in dtype_supported:
            raise ValueError(f"input[{name}] model input type is uint8"
                             f"input illegal: {input_data.dtype}")
        if input_data.dtype in ["int8"]:
            logging.warning(
                f"input[{name}] model input type is uint8, input data type is"
                f" {input_data.dtype}, will be convert.")
            input_data = (input_data + 128).astype(np.uint8)
        return input_data

    def data_pre_to_int8(self, input_data, name):
        if input_data.dtype not in ["uint8", "int8", "float64", "float32"]:
            raise ValueError(
                f"input[{name}] model input type is int8, only support to "
                f"provide uint8,int8 type input,"
                f"input illegal: {input_data.dtype}")

        if input_data.dtype in ["uint8"]:
            logging.warning(
                f"input[{name}] model input type is int8, input data type is"
                f" {input_data.dtype}, will be convert.")
            input_data = (input_data - 128).astype(np.int8)

        return input_data

    def data_preprocess(self, input_info):
        if hasattr(self.onnx_model.graph, 'inputs'):
            model_input = self.onnx_model.graph.inputs
        else:
            init_names = [
                init.name for init in self.onnx_model.graph.initializer
            ]
            model_input = [
                i for i in self.onnx_model.graph.input
                if i.name not in init_names
            ]
        for idx, name in enumerate(self.input_names):
            if hasattr(model_input[idx], 'dtype'):
                dtype = model_input[idx].dtype.value
            else:
                dtype = model_input[idx].type.tensor_type.elem_type
            input_data = input_info[name]
            if dtype not in self.func_map.keys(
            ) and input_data.dtype != self.dtype_map[dtype]:
                raise ValueError(
                    f"input[{name}] model expects type {dtype},"
                    f"not support input with data type {input_data.dtype}")
            if dtype in self.func_map.keys():
                input_info[name] = self.func_map[dtype](input_data, name)
        return input_info

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

        input_info = self.data_preprocess(input_info)
        result = self.executor.forward(input_info, output_names)
        result = list(result.values())
        return result

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

        result = self.executor.forward(input_info, output_names)
        result = list(result.values())
        return result
