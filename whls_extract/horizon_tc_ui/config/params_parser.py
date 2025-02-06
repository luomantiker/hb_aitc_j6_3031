# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
import copy
import logging
import os
import re
from typing import Any

import yaml
from schema import Schema

from horizon_tc_ui.config import mapper_consts as mconsts
from horizon_tc_ui.config.config_info import ConfigInfo
from horizon_tc_ui.config.schema_yaml import schema_yaml
from horizon_tc_ui.parser.caffe_parser import CaffeProto
from horizon_tc_ui.parser.onnx_parser import OnnxModel
from horizon_tc_ui.utils.tool_utils import (get_item_from_string,
                                            get_list_from_txt)


# TODO(ruxin.song): Reusable, separate file
def dict_add_key(data: dict, key: str, value: object) -> dict:
    if key not in data:
        data[key] = {}
    return data


class ParamsParser():
    """
    Used to parse user yaml file parameters
    """
    def __init__(self, yaml_path: str) -> None:
        # init member variables
        self.yaml_path = yaml_path
        self.yaml_info = {}
        self.yaml_file_data = {}
        self.model = None
        self.model_type = None  # ["caffe", "onnx"]
        self.is_checked = False

        # verify schema struct
        with open(self.yaml_path, 'r', encoding='UTF-8') as f:
            self.yaml_file_data = yaml.safe_load(f)

        # TODO(ruxin.song): To be optimized
        # adaptation of schema struct
        params = [
            "input_parameters", "calibration_parameters",
            "compiler_parameters", "custom_op"
        ]
        for param in params:
            self.yaml_file_data = dict_add_key(self.yaml_file_data, param,
                                               None)
        schema_dict = copy.deepcopy(schema_yaml)
        self.yaml_info = Schema(schema_dict).validate(self.yaml_file_data)

        # compatibility support
        self._mp_conf = self.yaml_info

        self.model_parameters = self.yaml_info['model_parameters']
        self.input_parameters = self.yaml_info['input_parameters']
        self.calibration_parameters = self.yaml_info['calibration_parameters']
        self.compiler_parameters = self.yaml_info['compiler_parameters']
        self.custom_op_parameters = self.yaml_info['custom_op']

        # generate conf info
        self.conf = ConfigInfo()

    def validate_parameters(self) -> None:
        self._validate_model_parameters()
        self._validate_input_parameters()
        self._validate_calibration_parameters()
        self._validate_compiler_parameters()
        self._validate_custom_op_parameters()
        self._validate_deprecated_params()
        return None

    def validate_bc_compile_mode_parameters(self) -> None:
        self._validate_compiler_parameters()
        self._validate_remove_node_type()
        self._validate_remove_node_name()
        self._validate_working_dir()
        self._validate_output_model_file_prefix()
        self._validate_march()
        self._validate_enable_vpu()

    def _validate_model_parameters(self) -> None:
        self._validate_model_file()
        self._validate_march()
        self._validate_working_dir()
        self._validate_output_model_file_prefix()
        self._validate_output_nodes()
        self._validate_set_node_data_type()
        self._validate_model_debug_mode()
        self._validate_node_info()
        self._validate_remove_node_type()
        self._validate_remove_node_name()
        self._validate_enable_vpu()
        return None

    def _validate_input_parameters(self) -> None:
        self._validate_input_name()
        self._validate_input_shape()
        self._validate_fast_perf()
        self._validate_input_batch()
        self._validate_separate_batch()
        self._validata_separate_name()
        self._validate_input_type()
        self._validate_input_layout()
        self._validate_input_type_association()
        self._validate_std_value()
        self._validate_scale_value()
        self._validate_mean_value()
        self._validate_norm_type()
        self._validate_odd_shape()
        return None

    def _validate_calibration_parameters(self) -> None:
        self._validate_calibration_type()
        self._validate_calibration_optimization()
        self._validate_cal_data_dir()
        self._validate_cal_data_type()
        self._validate_per_channel()
        self._validate_max_percentile()
        self._validate_run_on_cpu()
        self._validate_run_on_bpu()
        self._validate_cali_extra_param()
        self._validate_quantization_config()
        return None

    def _validate_compiler_parameters(self) -> None:
        self._validate_optimize_level()
        self._validate_input_source()
        self._validate_compile_debug_mode()
        self._validate_ability_entry()
        self._validate_core_num()
        self._validate_compile_mode()
        self._validate_balance_factor()
        self._validate_max_time_per_fc()
        self._validate_jobs()
        self._validate_advice()
        self._validate_hbdk3_compatible_mode()
        self._validate_extra_params()
        return None

    def _validate_custom_op_parameters(self) -> None:
        self._validate_custom_op()
        return None

    def _validate_model_file(self) -> None:
        onnx_model = self.model_parameters['onnx_model']
        caffe_model = self.model_parameters['caffe_model']
        prototxt = self.model_parameters['prototxt']
        if onnx_model and bool(caffe_model or prototxt):
            raise ValueError("Both onnx and caffe models are not supported")
        if not (onnx_model or caffe_model or prototxt):
            raise ValueError(
                "Missing model file input."
                "Please input caffe_model or onnx_model in yaml config file")

        self.conf.model_type = "onnx" if onnx_model else "caffe"
        if self.conf.model_type == "onnx":
            self.conf.onnx_model = self.file_check(onnx_model, ".onnx")
            self.conf.model = OnnxModel(model_file=self.conf.onnx_model)
            self.load_model = self.conf.model.model
            logging.info(f"Using onnx model file: {self.conf.onnx_model}")
        if self.conf.model_type == "caffe":
            self.conf.caffemodel_file = self.file_check(
                caffe_model, ".caffemodel")
            self.conf.prototxt_file = self.file_check(prototxt, ".prototxt")
            self.conf.model = CaffeProto(self.conf.prototxt_file)
            self.load_model = self.conf.model
            logging.info(f"Using caffe model file: {self.conf.caffemodel_file}"
                         f" and prototxt file: {self.conf.prototxt_file}")
        self.conf.input_num = self.conf.model.input_num()
        logging.info(
            f"Model has {self.conf.input_num} inputs according to model file")
        return None

    def _validate_march(self) -> None:
        self.conf.march = self.model_parameters['march']
        if self.conf.march not in mconsts.march_list:
            raise ValueError(f"User input march invalid: '{self.conf.march}' ."
                             f" It should in list {mconsts.march_list}")
        return None

    def _validate_working_dir(self) -> None:
        self.conf.working_dir = self.model_parameters["working_dir"]
        self.conf.working_dir = self._get_abspath(self.conf.working_dir)
        if not os.path.exists(self.conf.working_dir):
            logging.info("working_dir does not exist. "
                         f"Creating working_dir: {self.conf.working_dir}")
            os.makedirs(self.conf.working_dir, exist_ok=True)
        return None

    def _validate_output_model_file_prefix(self) -> None:
        prefix = self.model_parameters["output_model_file_prefix"]
        self.conf.output_model_file_prefix_full = os.path.join(
            self.conf.working_dir, prefix)
        self.conf.output_model_file_prefix = prefix
        return None

    def _validate_output_nodes(self) -> None:
        self.conf.output_nodes = get_list_from_txt(
            self.model_parameters['output_nodes'])
        return None

    def _validate_remove_node_type(self) -> None:
        self.conf.remove_node_type = get_list_from_txt(
            self.model_parameters['remove_node_type'])
        unsupport_type_list = [
            str(x) for x in self.conf.remove_node_type
            if x not in mconsts.removal_list
        ]
        if unsupport_type_list:
            raise ValueError(
                f'Unsupport remove {", ".join(unsupport_type_list)} now')
        return None

    def _validate_remove_node_name(self) -> None:
        self.conf.remove_node_name = get_list_from_txt(
            self.model_parameters['remove_node_name'])
        return None

    def _validate_enable_vpu(self):
        self.conf.enable_vpu = self.model_parameters.get('enable_vpu', False)
        self.bool_type_check('enable_vpu', self.conf.enable_vpu)
        return None

    def _validate_set_node_data_type(self) -> None:
        if self.model_parameters.get("set_node_data_type"):
            logging.warning(
                "The parameter 'set_node_data_type' will be deprecated "
                "and the parameter 'node_info' "
                "will provide the relevant functionality. "
                "Please refer to the documentation "
                "for the use of the parameter 'node_info'.")
        return None

    def __get_node_dict_by_str(self, param_value: str):
        _dict = {}
        value_list = []
        if ";" in param_value:
            value_list += param_value.split(';')
        else:
            value_list.append(param_value)

        for value in value_list:
            if not value:
                continue
            if ":" not in value or len(value.split(':')) != 2:
                raise ValueError(f"The format you gave is {param_value}, "
                                 "currently we only support "
                                 "Conv_0:int16;Conv_1:int16")

            node_name, data_type = value.split(':')
            _dict[node_name.strip()] = {"OutputType": data_type.strip()}

        return _dict

    def __get_node_dict_by_dict(self, param_value: dict):
        for node, value_dict in param_value.items():
            if not isinstance(value_dict, dict):
                raise ValueError(
                    f"The format you gave is {value_dict}"
                    f"({type(value_dict)}), currently we only support "
                    "{Conv_0:{'InputType0':'int16','OutputType':'int16'}}")

            for key, value in value_dict.items():
                if key not in ["ON", "OutputType", "InputType"] and \
                        not re.search(r'^InputType\d+$', key):
                    raise ValueError(
                        f"The format you gave is {key},"
                        "currently we only support ON、OutputType and "
                        r"InputType\d+")

                if key == "ON" and value not in mconsts.run_on_list:
                    raise ValueError("Currently only support is "
                                     f"{mconsts.run_on_list}")
        return param_value

    def __get_node_dict(self, node_dict: dict, param_value: str or dict):
        _dict = {}
        if isinstance(param_value, str):
            _dict = self.__get_node_dict_by_str(param_value)

        if isinstance(param_value, dict):
            _dict = self.__get_node_dict_by_dict(param_value)

        if _dict:
            for key, value in _dict.items():
                if key in node_dict:
                    node_dict[key].update(value)
                else:
                    node_dict[key] = value

        return node_dict

    def _validate_node_info(self):
        self.conf.node_dict = {}

        self.conf.node_info = self.model_parameters.get("node_info")
        self.conf.node_dict = self.__get_node_dict(self.conf.node_dict,
                                                   self.conf.node_info)

        if self.conf.march != "bernoulli2" and get_list_from_txt(
            self.calibration_parameters.get('run_on_cpu')) \
                and self.conf.node_info:
            logging.warning(
                "You configured both parameter run_on_cpu "
                "and parameter node_info. "
                "We will use the configuration of parameter node_info "
                "as the highest priority")

        if self.conf.march != "bernoulli2" and get_list_from_txt(
            self.calibration_parameters.get('run_on_bpu')) \
                and self.conf.node_info:
            logging.warning(
                "You configured both parameter run_on_bpu "
                "and parameter node_info. "
                "We will use the configuration of parameter node_info "
                "as the highest priority")

        logging.debug(f"node_dict: {self.conf.node_dict}")

    def _validate_model_debug_mode(self) -> None:
        self.conf.model_debug_mode = get_list_from_txt(
            self.model_parameters["debug_mode"])
        return None

    def _validate_input_name(self) -> None:
        if self.calibration_parameters.get('optimization') \
           == 'run_fast':
            return None
        model_input_names = self.conf.model.get_input_names()
        yaml_input_names = get_list_from_txt(
            self.input_parameters['input_name'])
        # implicit configuration
        if not yaml_input_names:
            if len(model_input_names) > 1:
                raise ValueError(
                    "Model has more than one input! "
                    "It is required to explicitly give input names "
                    "to ensure sequence is correct.")
            self.conf.input_names = model_input_names
            logging.info(
                "Model name not given in yaml_file, "
                f"using model name from model file: {model_input_names}")
            return None

        # explicit configuration
        if len(yaml_input_names) != len(model_input_names):
            raise ValueError(
                f"Wrong num of input names received. "
                f"Num of input name given: {len(yaml_input_names)}, "
                f"while model file has {len(model_input_names)} inputs")
        if len(yaml_input_names) != len(set(yaml_input_names)):
            raise ValueError(f"Input names duplicated: '{yaml_input_names}' ")

        for name in yaml_input_names:
            if name not in model_input_names:
                raise ValueError(
                    f"Input name does not exist in model file: {name}. "
                    f"name list: {model_input_names}")
        self.conf.input_names = yaml_input_names
        return None

    def _validate_input_shape(self) -> None:
        if self.calibration_parameters.get('optimization') \
           == 'run_fast':
            return None
        yaml_input_shape_txt = get_list_from_txt(
            self.input_parameters['input_shape'])
        model_file_shape = []
        for name in self.conf.input_names:
            model_file_shape.append(self.conf.model.get_input_shape(name))

        yaml_input_shape_len = len(yaml_input_shape_txt)
        if yaml_input_shape_txt and yaml_input_shape_len != \
           self.conf.input_num:
            raise ValueError(
                f"Num of input shape given: {yaml_input_shape_len}, "
                f"while model file has {self.conf.input_num} inputs")

        # explicit configuration
        if yaml_input_shape_txt:
            self.conf.input_shapes = []
            try:
                for shape_index, shape_item in enumerate(yaml_input_shape_txt):
                    self.conf.input_shapes.append(
                        list(map(int,
                                 shape_item.strip().lower().split('x'))))
            except Exception:
                raise ValueError("Input shape parse failed. "
                                 f"Input index {shape_index}: {shape_item}")
        # implicit configuration
        else:
            for origin_input_shape in model_file_shape:
                for dim_value in origin_input_shape:
                    if int(dim_value) == 0:
                        raise ValueError(
                            "The input_shape in the model is a dynamic shape. "
                            "Please configure 'input_shape' "
                            "option in 'input_parameters'")
            self.conf.input_shapes = model_file_shape
            logging.info("Model input shape not given in yaml_file, "
                         f"using shape from model file: {model_file_shape}")

        for index, shape_item in enumerate(self.conf.input_shapes):
            if len(shape_item) != 4:
                logging.info(
                    f"Input shape {shape_item} has length: {len(shape_item)}, "
                    "make sure it is a featuremap input")
            if self.conf.input_shapes[index] != model_file_shape[index]:
                logging.warning(
                    f"For input {index}: user input shape: "
                    f"{self.conf.input_shapes[index]} is different "
                    f"from model file input shape: {model_file_shape[index]}. "
                    "Using user input info")
        return None

    def _validate_input_batch(self) -> None:
        self.conf.input_batches = get_list_from_txt(
            self.input_parameters['input_batch'])
        if not self.conf.input_batches:
            return None

        if len(self.conf.input_batches) != 1:
            raise ValueError("input_batch option can only receive one input. "
                             f"There are {len(self.conf.input_batches)} given")

        for input_shape_item in self.conf.input_shapes:
            if input_shape_item[0] != 1:
                raise ValueError("The first dimension of input_shape must be 1"
                                 f", got {input_shape_item[0]}")

        return None

    def _validate_separate_batch(self) -> None:
        if not self.conf.input_batches or self.conf.input_batches[0] == 1:
            return None

        self.conf.separate_batch = self.input_parameters.get(
            'separate_batch', False)
        self.bool_type_check('separate_batch', self.conf.separate_batch)
        return None

    def _validata_separate_name(self) -> None:
        self.conf.separate_names = get_list_from_txt(
            self.input_parameters.get('separate_name', ''))
        if not self.conf.separate_names:
            return None
        if self.conf.separate_batch:
            raise ValueError("separate_batch and separate_name are not \
                             supported at the same time")

        # check if separate_name is in input_names
        for name in self.conf.separate_names:
            if name not in self.conf.input_names:
                raise ValueError(f"Separate name {name} is not in input names")

        return None

    def _validate_input_type(self) -> None:
        # check input_type_rt
        self.conf.input_type_rt = get_list_from_txt(
            self.input_parameters['input_type_rt'])
        self._validate_num(self.conf.input_type_rt, "input_type_rt")
        self.mconsts_check(self.conf.input_type_rt, mconsts.input_type_rt_list,
                           "input_type_rt")

        # check input_type_train
        self.conf.input_type_train = get_list_from_txt(
            self.input_parameters['input_type_train'])
        self._validate_num(self.conf.input_type_train, "input_type_train")
        self.mconsts_check(self.conf.input_type_train,
                           mconsts.input_type_train_list, "input_type_train")

        # check input_space_and_range
        self.conf.input_space_and_range = get_list_from_txt(
            self.input_parameters['input_space_and_range'])
        if self.conf.input_space_and_range:
            self._validate_num(self.conf.input_space_and_range,
                               'input_space_and_range')
        else:
            self.conf.input_space_and_range = ["regular"] * self.conf.input_num
        self.mconsts_check(self.conf.input_space_and_range,
                           mconsts.input_space_and_range,
                           'input_space_and_range')
        for idx, value in enumerate(self.conf.input_space_and_range):
            # nv12 data don't need layout info
            if value == 'bt601_video' and \
               self.conf.input_type_rt[idx] != 'nv12':
                raise ValueError(
                    "Input_type_rt and input_space_and_range combination "
                    f"invalid. input_space_and_range: {value} "
                    f"and input_type_rt {self.conf.input_type_rt[idx]}")
        return None

    def _validate_input_layout(self) -> None:
        self.conf.input_layout_train = get_list_from_txt(
            self.input_parameters['input_layout_train'])
        self.conf.input_layout_rt = get_list_from_txt(
            self.input_parameters['input_layout_rt'])

        if self.conf.input_layout_rt:
            logging.warning(
                'Please note that input_layout_rt is deprecated '
                'and configuring this parameter will not have any effect')

        if self.conf.input_layout_train:
            self._validate_num(self.conf.input_layout_train,
                               'input_layout_train')
            self.mconsts_check(self.conf.input_layout_train,
                               mconsts.layout_list, 'input_layout_train')

            for idx, item in enumerate(self.conf.input_layout_train):
                if len(self.conf.input_shapes[idx]) != 4:
                    logging.warning("Input shape is not four-dimensional, "
                                    "input_layout_train should be featuremap")
        else:
            for input_type_rt in self.conf.input_type_rt:
                if input_type_rt != 'featuremap':
                    raise ValueError('This model has inputs that are not '
                                     'featuremap, please configure '
                                     'input_layout_train value')
        return None

    def _validate_input_type_association(self) -> None:
        for idx in range(self.conf.input_num):
            train_type = self.conf.input_type_train[idx]
            rt_type = self.conf.input_type_rt[idx]

            # yuv444+NCHW+bernoulli2 is not supported
            if rt_type in ['yuv444', 'yuv444_128'] and \
               self.conf.input_layout_train == "NCHW" \
               and self.conf.march == 'bernoulli2':
                raise ValueError(
                    f"Input {idx} has input_type_rt {rt_type} with "
                    "input_layout_train NCHW "
                    "is not supported on bernoulli2 for now.")

            if train_type == rt_type:
                continue
            if train_type not in mconsts.legal_trans_dict.keys(
            ) or rt_type not in mconsts.legal_trans_dict[train_type]:
                raise ValueError(
                    f"Input {idx} has input_type_train '{train_type}' is not "
                    f"supported transform to input_type_rt '{rt_type}' for now"
                )
        self.conf.from_color = self.convert_input_type_to_color(
            self.conf.input_type_train)
        self.conf.to_color = self.convert_input_type_to_color(
            self.conf.input_type_rt)
        return None

    def _validate_norm_type(self) -> None:
        self.conf.norm_type = get_list_from_txt(
            self.input_parameters['norm_type'])
        logging.warning("norm_type parameter is deprecated and will be "
                        "determined by the configuration of "
                        "mean/scale/std parameter")

        scale_values = get_list_from_txt(
            self.input_parameters.get('scale_value', None))
        std_values = get_list_from_txt(
            self.input_parameters.get('std_value', None))

        if scale_values and std_values:
            raise ValueError(
                "Only one of scale_value and std_value can be configured")

        self.conf.norm_type = ["no_preprocess"] * self.conf.input_num
        for idx in range(self.conf.input_num):
            mean = self.conf.mean[idx]
            scale = self.conf.scale[idx]
            std = self.conf.std[idx]
            if mean and scale:
                self.conf.norm_type[idx] = 'data_mean_and_scale'
            elif mean and std:
                self.conf.norm_type[idx] = 'data_mean_and_std'
            elif scale:
                self.conf.norm_type[idx] = 'data_scale'
            elif std:
                self.conf.norm_type[idx] = 'data_std'
            elif mean:
                self.conf.norm_type[idx] = 'data_mean'

            if self.conf.norm_type[idx] != 'no_preprocess' and \
                    self.conf.input_type_rt[idx] == 'featuremap':
                raise ValueError(
                    f"Input_type_rt {idx + 1} is featuremap, "
                    "configuration of mean/scale/std is not supported")

        self._validate_num(self.conf.norm_type, "norm_type")
        return None

    def _validate_mean_value(self) -> None:
        mean_values = get_list_from_txt(self.input_parameters['mean_value'])
        if not mean_values:
            mean_values = [None] * self.conf.input_num
        self._validate_num(mean_values, "mean_value")

        self.conf.mean = []
        for idx in range(self.conf.input_num):
            mean_value = mean_values[idx]
            try:
                self.conf.mean.append(
                    get_item_from_string(mean_value, func=float))
            except Exception:
                raise ValueError(f"Wrong mean_value format {mean_value}, "
                                 f"please refer to user manual")
        return None

    def _validate_scale_value(self) -> None:
        scale_values = get_list_from_txt(self.input_parameters['scale_value'])
        if not scale_values:
            scale_values = [None] * self.conf.input_num
        self._validate_num(scale_values, "scale_value")

        self.conf.scale = []
        for idx in range(self.conf.input_num):
            scale_value = scale_values[idx]
            try:
                self.conf.scale.append(
                    get_item_from_string(scale_value, func=float))
            except Exception:
                raise ValueError(f"Wrong scale_value format {scale_value}, "
                                 f"please refer to user manual")
        return None

    def _validate_std_value(self) -> None:
        std_values = get_list_from_txt(self.input_parameters['std_value'])
        if not std_values:
            std_values = [None] * self.conf.input_num
        self._validate_num(std_values, "std_value")

        self.conf.std = []
        for idx in range(self.conf.input_num):
            std_value = std_values[idx]
            try:
                self.conf.std.append(
                    get_item_from_string(std_value, func=float))
            except Exception:
                raise ValueError(f"Wrong std_value format {std_value}, "
                                 f"please refer to user manual")
        return None

    def _validate_odd_shape(self) -> None:
        for index, input_type_rt in enumerate(self.conf.input_type_rt):
            if not input_type_rt == 'nv12':
                continue

            shape = self.conf.input_shapes[index]
            if self.conf.input_layout_train[index] == "NCHW":
                h, w = shape[2], shape[3]  # NCHW
            else:
                h, w = shape[1], shape[2]  # NHWC
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(f'Invalid nv12 input shape: {shape}, '
                                 'nv12 type does not support odd size')
        return None

    def _validate_calibration_type(self) -> None:
        self.conf.calibration_type = self.calibration_parameters[
            'calibration_type']
        check_list = mconsts.autoq_caltype_list + mconsts.preq_caltype_list
        if self.conf.calibration_type not in check_list:
            raise ValueError(
                "User input calibration_type invalid, "
                f"'{self.conf.calibration_type}' should in list {check_list}")
        return None

    def _validate_calibration_optimization(self) -> None:
        self.conf.calibration_optimization = get_list_from_txt(
            self.calibration_parameters['optimization'])
        if not self.conf.calibration_optimization:
            self.conf.calibration_optimization = None
        return None

    def _validate_cal_data_dir(self) -> None:
        if self.conf.calibration_type in mconsts.preq_caltype_list:
            logging.info(
                f"Parameter calibration_type is {self.conf.calibration_type}. "
                "cal_data_dir check skipped")
            self.conf.cal_data_dir = []
            return None

        self.conf.cal_data_dir = get_list_from_txt(
            self.calibration_parameters['cal_data_dir'])
        if len(self.conf.cal_data_dir) not in [self.conf.input_num, 0]:
            raise ValueError("Wrong cal_data_dir num received.")
        if len(self.conf.cal_data_dir) == 0:
            self.conf.calibration_type = "skip"
            logging.info("Calibration dataset not configured, "
                         "calibration method modified to skip")
            return None

        for cal_index in range(self.conf.input_num):
            self.conf.cal_data_dir[cal_index] = self._get_abspath(
                self.conf.cal_data_dir[cal_index])
            if not os.path.exists(self.conf.cal_data_dir[cal_index]):
                raise ValueError("Cal_data_dir does not exist: "
                                 f"{self.conf.cal_data_dir[cal_index]}")
        return None

    def _validate_cal_data_type(self) -> None:
        self.conf.cal_data_type = get_list_from_txt(
            self.calibration_parameters['cal_data_type'])

        if self.conf.cal_data_type:
            logging.warning("Note that this parameter is deprecated, "
                            "Please switch the calibration dataset to npy.")
        else:
            return None

        if self.conf.calibration_type in ['skip']:
            logging.info('The skip or load calibration method is configured, '
                         'so the cal_data_type parameter is not used')
            return None

        self._validate_num(self.conf.cal_data_type, 'cal_data_type')

        for index, type in enumerate(self.conf.cal_data_type):
            if type not in mconsts.cal_data_type_list:
                raise ValueError(
                    f"User input cal_data_type invalid, '{type}' "
                    f"should in list {mconsts.cal_data_type_list}")
            dir_suffix = 'uint8'
            if self.conf.cal_data_dir[index].endswith('_f32'):
                dir_suffix = 'float32'
            if dir_suffix != self.conf.cal_data_type[index]:
                logging.warning(
                    f"The calibration dir name suffix is not the same as the "
                    f"value {self.conf.cal_data_type[index]} of the parameter "
                    f"cal_data_type, the parameter setting will prevail")
                continue
            logging.info(
                f"The calibration dir name suffix is the same as the value "
                f"{self.conf.cal_data_type[index]} of the"
                " cal_data_type parameter "
                f"and will be read with the value of cal_data_type.")

        return None

    def _validate_per_channel(self) -> None:
        _per_channel = self.calibration_parameters.get('per_channel', False)
        self.conf.per_channel = _per_channel
        self.bool_type_check('per_channel', _per_channel)
        return None

    def _validate_max_percentile(self) -> None:
        self.conf.max_percentile = self.calibration_parameters[
            'max_percentile']
        if not self.conf.max_percentile:
            return None
        if not 0 <= self.conf.max_percentile <= 1:
            raise ValueError("Invalid max_percentile: "
                             f"{self.conf.max_percentile}, "
                             f"avaliable range: 0~1")
        return None

    def _validate_run_on_cpu(self) -> None:
        self.conf.run_on_cpu = get_list_from_txt(
            self.calibration_parameters['run_on_cpu'])
        return None

    def _validate_run_on_bpu(self) -> None:
        self.conf.run_on_bpu = get_list_from_txt(
            self.calibration_parameters['run_on_bpu'])
        return None

    def _validate_cali_extra_param(self) -> None:
        cali_extra_param = {}
        for key, value in self._mp_conf['calibration_parameters'].items():
            if key not in mconsts.calibration_parameters:
                cali_extra_param[key] = value
        self.conf.cali_extra_param = cali_extra_param
        return None

    def _validate_quantization_config(self) -> None:
        self.conf.quant_config = self.calibration_parameters.get(
            "quant_config")  # noqa
        return None

    def _validate_num(self, obj: list, name: str) -> None:
        if len(obj) != self.conf.input_num:
            raise ValueError(f"Wrong {name} num received. "
                             f"Num of {name} given: {len(obj)} "
                             "is not equal to input num "
                             f"{self.conf.input_num}")
        return None

    def _validate_optimize_level(self) -> None:
        self.conf.optimize_level = self.compiler_parameters['optimize_level']
        if self.conf.optimize_level not in mconsts.optimize_level_hbdk4:
            raise ValueError("User input optimize_level invalid, "
                             f"'{self.conf.optimize_level}' should in list "
                             f"{mconsts.optimize_level_hbdk4}")
        return None

    def _validate_input_source(self) -> None:
        input_source_input = self._mp_conf['compiler_parameters'].get(
            "input_source", {})
        if not isinstance(input_source_input, dict):
            raise ValueError("Invalid input_source format received. "
                             "input_source should be a dict")
        self.input_source = {}

        input_names = self.conf.input_names
        for input_index, input_name in enumerate(input_names):
            input_type_rt = self.conf.input_type_rt[input_index]
            input_source_item = input_source_input.get(input_name)
            support_dict = mconsts.input_source_support_dict

            # if input_source not received,
            # give default value according to input_type_rt
            if not input_source_item:
                self.input_source[input_name] = "ddr"
                if input_type_rt in support_dict['pyramid']:
                    self.input_source[input_name] = "pyramid"
                logging.warning(
                    f"Input node {input_name}'s input_source not set, "
                    f"it will be set to {input_source_item} by default")
                continue

            # if input_source received, check it if in input_source_range
            if input_source_item not in mconsts.input_source_range:
                raise ValueError("Invalid input_source received. input_source "
                                 f"{input_source_item} should in list "
                                 f"{mconsts.input_source_range}")

            # check input_type_rt if input_source is supported
            if input_type_rt not in support_dict[input_source_item]:
                raise ValueError(
                    "Wrong input_source received. input type rt : "
                    f"{input_type_rt} does not support "
                    f"input_source {input_source_item}")

            self.input_source[input_name] = input_source_item

        self.conf.input_source = self.input_source
        # self.input_source.update({'_default_value': 'ddr'})
        return None

    def _validate_compile_debug_mode(self) -> None:
        _debug = self.compiler_parameters['debug']
        self.bool_type_check('debug', value=_debug)
        self.conf.compile_debug_mode = _debug
        return None

    def _validate_ability_entry(self) -> None:
        self.conf.ability_entry = self.compiler_parameters['ability_entry']
        return None

    def _validate_core_num(self) -> None:
        self.conf.core_num = self.compiler_parameters['core_num']
        if self.conf.core_num not in mconsts.core_num_range:
            raise ValueError("Wrong core num setting given, "
                             f"{self.conf.core_num} should in range "
                             f"{mconsts.core_num_range}")
        return None

    def _validate_compile_mode(self) -> None:
        self.conf.compile_mode = self.compiler_parameters['compile_mode']
        if self.conf.compile_mode not in mconsts.compile_mode_list:
            raise ValueError(
                f"Invalid compile model received. {self.conf.compile_mode} "
                f"is invalid, it should in list {mconsts.compile_mode_list}")
        return None

    def _validate_balance_factor(self) -> None:
        compile_mode = self.compiler_parameters.get('compile_mode')
        self.conf.balance_factor = self.compiler_parameters.get(
            'balance_factor')
        if compile_mode in mconsts.balance_factor_mapping:
            if self.conf.balance_factor:
                logging.warning(
                    "Parameter compile_mode is set to %s, "
                    "balance_factor will not take effect", compile_mode)
            else:
                self.conf.balance_factor = mconsts.balance_factor_mapping[
                    compile_mode]
                logging.info(
                    "Parameter compile_mode is set to %s, "
                    "balance_factor will set to %s.", compile_mode,
                    self.conf.balance_factor)
            return None
        elif compile_mode == "balance" and self.conf.balance_factor is None:
            raise ValueError("Parameter compile_mode is set to balance, "
                             "please set balance_factor to use this mode")
        if 0 > self.conf.balance_factor or self.conf.balance_factor > 100:
            raise ValueError(
                f"Invalid balance_factor received: {self.conf.balance_factor}"
                ", value range is 0-100")

    def _validate_max_time_per_fc(self) -> None:
        self.conf.max_time_per_fc = self.compiler_parameters['max_time_per_fc']
        if self.conf.max_time_per_fc and \
           (self.conf.max_time_per_fc < 1000 or self.conf.max_time_per_fc > 4294967295): # noqa
            raise ValueError("Parameter max_time_per_fc value check failed. "
                             "Please set it 0 or 1000-4294967295")
        return None

    def _validate_jobs(self):
        self.conf.jobs = self._mp_conf['compiler_parameters'].get('jobs')

    def _validate_advice(self):
        self.conf.advice = self._mp_conf['compiler_parameters'].get('advice')
        if self.conf.advice and not str(self.conf.advice).isdigit():
            raise ValueError("The parameter advice, "
                             "must be a positive integer")

    def _validate_hbdk3_compatible_mode(self) -> None:
        self.conf.hbdk3_compatible_mode = self.compiler_parameters.get(
            'hbdk3_compatible_mode', False)
        self.bool_type_check('hbdk3_compatible_mode',
                             self.conf.hbdk3_compatible_mode)
        return None

    def _validate_extra_params(self) -> None:
        self.conf.compile_extra_params = self.compiler_parameters.get(
            'extra_params', {})
        return None

    def _validate_custom_op(self) -> None:
        return None

    def _validate_deprecated_params(self):
        _layer_out_dump = self.model_parameters.get('layer_out_dump', False)
        _preprocess_on = self.calibration_parameters.get(
            'preprocess_on', False)
        self.bool_type_check('layer_out_dump', _layer_out_dump)
        self.bool_type_check('preprocess_on', _preprocess_on)
        self.conf.layer_out_dump = _layer_out_dump
        if self.model_parameters.get('log_level'):
            logging.warning("User input 'log_level' is deprecated，"
                            "Console log level is set as info, "
                            "and logfile log level is set as debug.")
        self.conf.log_level = logging.DEBUG
        if _preprocess_on:
            logging.warning(
                'Please note that preprocess_on is deprecated '
                'and configuring this parameter will not have any effect')

    @staticmethod
    def diff_dynamic_shape(before: list, after: list) -> bool:
        """
        Compare whether the difference between two Shapes in ['?', 0, -1]

        Returns:
            bool: Returns False if the modified range is not in ['?', 0, -1]
        """
        diff = list(
            map(
                lambda i: before[i] if before[i] not in ['?', 0, -1] and
                before[i] != after[i] else None, range(len(before))))
        diff = list(filter(lambda x: x is not None, diff))
        logging.debug(f'Shape diff result: {diff}')
        return len(diff) == 0

    def _validate_fast_perf(self) -> None:
        """
        validate fast perf input parameters
        """
        if self.calibration_parameters.get('optimization') \
           != 'run_fast':
            return None
        model_input_names = self.conf.model.get_input_names()
        yaml_input_names = get_list_from_txt(
            self.input_parameters.get('input_name'))
        yaml_input_shapes_txt = get_list_from_txt(
            self.input_parameters.get('input_shape'))
        yaml_input_shapes = []
        try:
            for shape_item in yaml_input_shapes_txt:
                yaml_input_shapes.append(
                    list(map(int,
                             str(shape_item).strip().lower().split('x'))))
        except Exception:
            raise ValueError('Failed to parse the input_shape, '
                             'please double check your input')
        model_file_shapes = [
            self.conf.model.get_input_shape(name) for name in model_input_names
        ]
        # validate user input_name and input_shape
        if len(yaml_input_names) > len(model_input_names):
            raise ValueError('Your number of input names '
                             f'{len(yaml_input_names)} is not '
                             'equal to the number of model input names '
                             f'{len(model_input_names)}')
        parsed_shapes = copy.deepcopy(model_file_shapes)
        for idx_yaml, yaml_name in enumerate(yaml_input_names):
            try:
                idx_model = model_input_names.index(yaml_name)
            except ValueError:
                raise ValueError(f'Your input name {yaml_name} '
                                 'not in model inputs')
            yaml_shape = yaml_input_shapes[idx_yaml]
            model_shape = parsed_shapes[idx_model]
            if len(yaml_shape) != len(model_shape):
                raise ValueError(
                    f'Input shape {yaml_shape} '
                    f'length not equal to model input shape {model_shape}')
            if not self.diff_dynamic_shape(model_shape, yaml_shape):
                raise ValueError(
                    f'Your input shape {yaml_shape} but model input shape is '
                    f'{model_shape}, we only supports modifying '
                    'the dim value of a dynamic batch')
            parsed_shapes[idx_model] = yaml_shape

        for idx_shape, current_shape in enumerate(parsed_shapes):
            for idx_dim, dim_value in enumerate(current_shape):
                if dim_value not in ['?', -1, 0]:
                    continue
                if idx_dim != 0:
                    raise ValueError(
                        f'The input {model_input_names[idx_shape]} '
                        'has the dynamic input_shape '
                        f'{model_file_shapes[idx_shape]} but the '
                        'dynamic batch dim is not the first dim, '
                        'please configure the input_shape option '
                        'and specify all the dynamic dims of this input')
                logging.warning(
                    f'The input {model_input_names[idx_shape]} has '
                    'dynamic input_shape, the first dim of the '
                    f'{model_file_shapes[idx_shape]} will be set to 1')
                parsed_shapes[idx_shape][idx_dim] = 1

        self.conf.input_shapes = parsed_shapes
        self.conf.input_names = model_input_names

    def _get_abspath(self, path: str) -> str:
        if not os.path.isabs(path):
            yaml_base_dir = os.path.dirname(self.yaml_path)
            path = os.path.abspath(os.path.join(yaml_base_dir, path))
            logging.debug(f"Using abs path {path}")
        return path

    def file_check(self, file_path: str, suffix: str = "") -> str:
        file_path = self._get_abspath(file_path)
        # check file suffix
        if not file_path.endswith(suffix):
            raise ValueError(
                f"file invalid: {file_path}. It should be a '{suffix}' file")
        # check whether the file exists
        if suffix and not os.path.isfile(file_path):
            raise ValueError(f"Can not find file given: {file_path},"
                             "please check the parameter")
        return file_path

    def mconsts_check(self, obj: list, expect: list, name: str) -> None:
        for value in obj:
            if value in expect:
                continue
            raise ValueError(f"Invalid {name} received. "
                             f"{name}: '{value}' should in list {expect}")
        return None

    def convert_input_type_to_color(self, input_type_list: list) -> list:
        res = []
        for idx, input_type in enumerate(input_type_list):
            space_and_range = self.conf.input_space_and_range[idx]
            if input_type == 'nv12' and space_and_range == 'bt601_video':
                input_type = 'yuv420sp_bt601_video'

            if input_type in mconsts.color_map:
                input_type = mconsts.color_map[input_type]
            else:
                input_type = input_type.upper()
            res.append(input_type)
        return res

    def bool_type_check(self, name: str, value: Any) -> None:
        """Use to check bool parameter

        Args:
            name (str): parameter name
            value (Any): parameter value

        Raises:
            ValueError: if input parameter is not bool type
        """
        if not isinstance(value, bool):
            logging.error(f'Wrong parameter `{name}` setting given, '
                          'expect type bool '
                          f'but receive type {type(value).__name__}')
            exit(1)
