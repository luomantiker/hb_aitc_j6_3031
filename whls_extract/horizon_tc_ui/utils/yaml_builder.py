# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
import shutil
import time
from collections import Counter
from typing import Literal, Tuple

import yaml
from schema import Schema

from horizon_tc_ui import HB_HBIRRuntime, HB_ONNXRuntime
from horizon_tc_ui import __file__ as tc_ui_root_file
from horizon_tc_ui.config.mapper_conf_parser import conf_schema_dict
from horizon_tc_ui.parser.caffe_parser import CaffeParser


class YamlBuilder:
    def __init__(self,
                 mode: Literal["fast_perf", "check"],
                 proto: str,
                 model: str,
                 model_type: Literal["onnx", "caffe", "bc"],
                 march: str,
                 input_shape: Tuple[Tuple[str, str]] = (),
                 hbdk_version: int = 4,
                 yaml_path: str = "") -> None:
        if mode not in ['fast_perf', 'check']:
            raise ValueError("mode only supports fast_perf, check")
        if model_type not in ["onnx", "caffe", "bc"]:
            raise ValueError("model_type only supports onnx/caffe/bc")

        self.mode = mode
        self.proto_path = proto
        self.model_path = model
        self.model_type = model_type
        self.march = march
        self.input_shape = input_shape
        self.hbdk_version = hbdk_version
        tc_ui_path = os.path.abspath(os.path.dirname(tc_ui_root_file))
        self.template_path = os.path.join(tc_ui_path, 'template/')
        self.workspace = os.path.join(os.getcwd(), '.' + self.mode)
        self.model = None
        self.config = {}
        self.model_name = None
        self.yaml_path = yaml_path
        self.input_num = None
        self.model_input_shapes = []
        self.input_names = []
        self.input_shapes = []

        self.validate()
        return None

    def validate(self) -> None:
        if self.model_type == 'caffe' and not self.proto_path:
            raise ValueError('model type is caffe but proto file missing')
        return None

    def get_template_config(self) -> None:
        fast_perf_template_path = os.path.join(self.template_path,
                                               f"{self.mode}_template.yaml")
        with open(fast_perf_template_path, 'r') as stream:
            self.config = yaml.safe_load(stream)
        logging.debug(self.config)
        logging.info(f"{self.mode} template yaml load success")
        return None

    def prepare_env(self) -> None:
        model_file_name = os.path.basename(self.model_path)
        self.model_name = os.path.splitext(model_file_name)[0]
        if not self.yaml_path:
            self.yaml_path = os.path.join(self.workspace,
                                          f'{self.model_name}_config.yaml')
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.workspace, exist_ok=True)

    def model_parser(self) -> None:
        if self.model_type == 'onnx':
            self.model = HB_ONNXRuntime(model_file=self.model_path)
        elif self.model_type == 'bc':
            self.model = HB_HBIRRuntime(model_file=self.model_path)
        else:
            self.model = CaffeParser(model_file=self.model_path,
                                     model_proto=self.proto_path)

        self.input_names = self.model.input_names
        self.input_shapes = self.model.input_shapes
        self.input_num = len(self.input_names)
        return None

    def check_input(self) -> None:
        # input_shapes
        for set_name, set_shape in self.input_shape:
            if set_name not in self.input_names:
                raise ValueError(f'Your input name {set_name} '
                                 'not in model inputs, '
                                 'please double check your input')
            index = self.input_names.index(set_name)
            self.input_shapes[index] = [int(d) for d in set_shape.split('x')]
        check_shapes = self.input_shapes
        for idx_shape, current_shape in enumerate(check_shapes):
            shape_res = Counter(
                [isinstance(dim, int) for dim in current_shape])
            if shape_res[False] == 0:
                continue
            if shape_res[False] == 1 and not isinstance(current_shape[0], int):
                logging.warning(f'The input {self.input_names[idx_shape]} has '
                                'dynamic input_shape, the first dim of the '
                                f'{self.input_shapes[idx_shape]} '
                                'will be set to 1')
                self.input_shapes[idx_shape][0] = 1
            else:
                raise ValueError(
                    f'The input {self.input_names[idx_shape]} '
                    'has the dynamic input_shape '
                    f'{self.input_shapes[idx_shape]} but the '
                    'dynamic batch dim is not the first dim, '
                    'please configure the input-shape option '
                    'and specify all the dynamic dims of this input')

    def update_params(self) -> None:
        self.get_template_config()
        self.update_model_params()
        self.update_input_name_and_shape()
        self.update_compile_params()

        if self.mode == "fast_perf":
            self.update_params_of_fast_perf()
        elif self.mode == "check":
            self.update_params_of_check()
        else:
            return None

    def update_model_params(self) -> None:
        cwd = os.getcwd()
        model_abs_path = os.path.join(cwd, self.model_path)
        model_params = self.config["model_parameters"]
        if self.model_type == "caffe":
            proto_abs_path = os.path.join(cwd, self.proto_path)
            model_params["caffe_model"] = model_abs_path
            model_params["prototxt"] = proto_abs_path
        if self.model_type == "onnx":
            model_params["onnx_model"] = model_abs_path
        model_params["march"] = self.march
        model_params["output_model_file_prefix"] = self.model_name
        model_params["working_dir"] = os.path.join(cwd, "model_output")
        if os.getenv("MAPPER_LOG_FORMAT_TIMESTAMP"):
            model_params["working_dir"] += "_" + str(
                time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.config["model_parameters"].update(model_params)
        return None

    def update_input_name_and_shape(self) -> None:
        input_params = self.config["input_parameters"]
        input_params["input_name"] = ';'.join(self.input_names)
        input_shapes = []
        for idx in range(len(self.input_names)):
            shape = 'x'.join(map(str, self.input_shapes[idx]))
            input_shapes.append(shape)
            self.model_input_shapes.append(shape)
        input_params["input_shape"] = ';'.join(input_shapes)
        self.config["input_parameters"].update(input_params)
        return None

    def update_compile_params(self) -> None:
        opt_level = self.config["compiler_parameters"]["optimize_level"]
        if self.hbdk_version == 3:
            opt_level = opt_level.replace("O2", "O3")
        self.config["compiler_parameters"]["optimize_level"] = opt_level

    def update_params_of_fast_perf(self) -> None:
        input_type_rt = ['featuremap'] * self.input_num
        input_type_train = ['featuremap'] * self.input_num
        input_layout_train = ['NCHW'] * self.input_num
        for idx in range(len(self.input_names)):
            shape = self.input_shapes[idx]
            # Data is not four-dimensional or channel dimension is not 3
            if len(shape) != 4 or (shape[1] != 3 and shape[3] != 3):
                continue
            # NCHW, NHWC
            h_idx, w_idx, c_idx = (2, 3, 1) if shape[1] == 3 else (1, 2, 3)
            if shape[h_idx] % 2 != 0 or shape[w_idx] % 2 != 0:
                continue
            if self.model.layout[idx]:
                input_type_rt[idx] = "nv12"
                input_type_train[idx] = "bgr"
            input_layout_train[idx] = "NCHW" if c_idx == 1 else "NHWC"

        input_params = self.config["input_parameters"]
        input_params["input_type_rt"] = ';'.join(input_type_rt)
        input_params["input_type_train"] = ';'.join(input_type_train)
        # To be consistent with the original layout, do not do Transpose
        input_params["input_layout_rt"] = ';'.join(input_layout_train)
        input_params["input_layout_train"] = ';'.join(input_layout_train)

        norm_type = ';'.join(['no_preprocess'] * self.input_num)
        input_params["norm_type"] = norm_type
        self.config["input_parameters"].update(input_params)
        return None

    def update_params_of_check(self) -> None:
        input_type_rt = ['featuremap'] * self.input_num
        input_type_train = ['featuremap'] * self.input_num
        input_params = self.config["input_parameters"]
        input_params["input_type_rt"] = ';'.join(input_type_rt)
        input_params["input_type_train"] = ';'.join(input_type_train)
        norm_type = ';'.join(['no_preprocess'] * self.input_num)
        input_params["norm_type"] = norm_type
        self.config["input_parameters"].update(input_params)
        # update save path
        self.config["model_parameters"]["working_dir"] = os.path.join(
            os.getcwd(), ".hb_compile")

    def dump(self) -> str:
        logging.info(f"Updated yaml config info: {self.config}")
        validated_config = Schema(conf_schema_dict).validate(self.config)
        with open(self.yaml_path, 'w', encoding="utf-8") as f:
            yaml.safe_dump(validated_config, f)
        return self.yaml_path

    def build(self) -> str:
        self.prepare_env()
        self.model_parser()
        self.check_input()
        self.update_params()
        return self.dump()
