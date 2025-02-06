#  Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#  #
#  The material in this file is confidential and contains trade secrets
#  of Horizon Robotics Inc. This is proprietary information owned by
#  Horizon Robotics Inc. No part of this work may be disclosed,
#  reproduced, copied, transmitted, or used in any way for any purpose,
#  without the express written permission of Horizon Robotics Inc.

import logging
import os
import sys
from copy import deepcopy
from typing import Iterable

import numpy as np
from horizon_nn.api import build_model
from horizon_nn.custom.op_registration import op_register

from horizon_tc_ui.config.config_info import ConfigInfo
from horizon_tc_ui.config.mapper_consts import march_list
from horizon_tc_ui.data import data_loader_factory as dlf
from horizon_tc_ui.data.transformer import (F32ToS8Transformer,
                                            F32ToU8Transformer)
from horizon_tc_ui.utils.tool_utils import get_all_data
from horizon_tc_ui.utils.wrap_utils import try_except_wrapper


class PTQModelBuilder:
    def __init__(
            self,
            march: str,
            model: str = '',
            proto: str = '',
            conf: ConfigInfo = None,
            name_prefix: str = 'model',
            workspace: str = None,
    ) -> None:
        if march not in march_list:
            raise ValueError(f'Invalid march {march},',
                             f'march parameter only supports {march_list}')
        self.model = model
        self.proto = proto
        self.march = march
        self.conf = conf
        self.name_prefix = name_prefix
        self.quantize_params = None
        self.calibration_model = None
        self.workspace = workspace
        if not self.workspace:
            self.workspace = os.getcwd()
        if not os.path.exists(self.workspace):
            raise ValueError(
                f'workspace does not exist. workspace: {self.workspace}')
        if self.march.startswith('nash'):
            self.march = 'nash'
        return None

    @try_except_wrapper("hmct.custom.op_registration.op_register")
    def _op_register(self):
        if not self.conf.custom_op:
            return None

        cop_method = self.conf.custom_op_method
        if cop_method != "register":
            raise ValueError(f"custom op method: {cop_method} not recognized")

        sys.path.append('..')
        cop_register_files = self.conf.cop_register_files
        cop_dir = self.conf.custom_op_dir
        # if cop dir exist, user folder name as prefix
        if cop_dir:
            if os.path.samefile(cop_dir, os.getcwd()):
                cop_dir_prefix = ''
            else:
                cop_dir_prefix = cop_dir.lstrip("./") + '.'
        else:
            cop_dir_prefix = ''

        for cop_module in cop_register_files:
            if cop_module.endswith(".py"):
                cop_module = os.path.splitext(cop_module)[0]
            op_register(f"{cop_dir_prefix}{cop_module}")
            logging.info(f"Register {cop_dir_prefix}{cop_module}")

    def _get_cali_data_loader(self, idx: int) -> Iterable:
        transformers = []
        input_type_rt = self.conf.input_type_rt[idx]
        input_shape = self.conf.input_shapes[idx]
        if self.conf.cal_data_type:
            cal_data_type = self.conf.cal_data_type[idx]
        else:
            cal_data_type = None

        if input_type_rt.startswith("featuremap"):
            if input_type_rt.endswith("s8"):
                transformers.append(F32ToS8Transformer())
            if input_type_rt.endswith("u8"):
                transformers.append(F32ToU8Transformer())

        dtype = np.uint8 if cal_data_type == "uint8" else np.float32
        data_loader = dlf.get_raw_image_dir_loader(transformers,
                                                   self.conf.cal_data_dir[idx],
                                                   input_shape, dtype)

        return data_loader

    def _get_cali_data(self) -> dict:
        calibration_data = {}
        if self.conf.calibration_type in ['skip']:
            return calibration_data
        # TODO(ruxin.song): custom_op not support
        for idx, name in enumerate(self.conf.input_names):
            logging.info(f"Processing calibration set data for input[{name}]")
            loader = self._get_cali_data_loader(idx)
            calibration_data.update({name: get_all_data(loader)})
            logging.info(f"Finished. data num: {len(calibration_data[name])}")
        return calibration_data

    def _get_cali_params(self) -> dict:
        calibration_type = self.conf.calibration_type
        # skip is configured to be "", hmct configuration defaults
        if calibration_type == 'skip':
            calibration_type = ''

        self._op_register()

        cali_dict = {
            'calibration_type': calibration_type,
            'calibration_data': self._get_cali_data()
        }

        for input_name, cali_data in cali_dict['calibration_data'].items():
            self.conf.first_cali_data.update({input_name: cali_data[0]})

        if self.conf.per_channel:
            cali_dict['per_channel'] = self.conf.per_channel
        if self.conf.max_percentile:
            cali_dict['max_percentile'] = self.conf.max_percentile
        # for temporary parameters
        if self.conf.cali_extra_param:
            for key, value in self.conf.cali_extra_param.items():
                cali_dict[key] = value
        return cali_dict

    def _get_build_input_param(self, input_name: str,
                               input_index: int) -> dict:
        input_shape = self.conf.input_shapes[input_index]
        build_input_dict = {
            'input_shape': input_shape,
        }

        if self.conf.input_batches:
            input_batch = self.conf.input_batches[0]
            build_input_dict.update({'input_batch': int(input_batch)})

        return build_input_dict

    def _get_input_params(self) -> dict:
        """Generate input params dict
        """
        input_dict = {}
        for input_index, input_name in enumerate(self.conf.input_names):
            input_dict[input_name] = self._get_build_input_param(
                input_name=input_name, input_index=input_index)

        return input_dict

    # TODO(wenhao.ma) Remove this func
    def _get_check_input_dict(self, model, user_input_names,
                              user_input_shapes) -> dict:
        model_input_name = model.get_input_names()
        for name in user_input_names:
            if name not in model_input_name:
                message = 'wrong input name: %s, available: %s' % (
                    name, model_input_name)
                raise ValueError(message)
        input_dict = {}
        for input_name, input_shape in user_input_shapes.items():
            input_shape_model = model.get_input_dims(input_name)
            if input_shape != input_shape_model:
                logging.warning('for input "%s", user input_shape:%s '
                                'is not same with model input_shape: %s' %
                                (input_name, input_shape, input_shape_model))
            input_dict[input_name] = {'input_batch': int(input_shape[0])}
            input_shape[0] = 1
            input_dict[input_name].update({'input_shape': input_shape})

        return input_dict

    def _get_node_params(self) -> dict:
        node_dict = {}

        if self.conf.node_dict:
            node_dict = self.conf.node_dict

        if self.conf.get('run_on_cpu'):
            for item in self.conf.run_on_cpu:
                val_dict = node_dict.get(item, {})
                if not val_dict or "ON" not in val_dict:
                    val_dict["ON"] = "CPU"

                node_dict[item] = val_dict

        if self.conf.get('run_on_bpu'):
            for item in self.conf.run_on_bpu:
                val_dict = node_dict.get(item, {})
                if not val_dict or "ON" not in val_dict:
                    val_dict["ON"] = "BPU"

                node_dict[item] = val_dict
        return node_dict

    def _get_debug_mode(self) -> list:
        debug_mode = self.conf.model_debug_mode
        if self.conf.layer_out_dump:
            debug_mode.append("dump_all_layers_output")
        return debug_mode

    def get_quantize_params(self) -> dict:
        if self.quantize_params is not None:
            return self.quantize_params
        cali_params = self._get_cali_params()
        input_params = self._get_input_params()
        node_params = self._get_node_params()
        debug_params = self._get_debug_mode()
        check_mode = False if cali_params.get('calibration_data') else True
        self.quantize_params = {
            'cali_dict': cali_params,
            'input_dict': input_params,
            'node_dict': node_params,
            'debug_methods': debug_params,
            'output_nodes': self.conf.output_nodes,
            'optimization_methods': self.conf.calibration_optimization,
            'quant_config': deepcopy(self.conf.quant_config),
            'check_mode': check_mode
        }
        return self.quantize_params

    @try_except_wrapper("hmct.api.build_model",
                        ["params.cali_dict.calibration_data"])
    def build_model(self, params: dict = {}) -> None:
        name_prefix = os.path.join(self.workspace, self.name_prefix)
        self.calibration_model = build_model(
            onnx_file=self.conf.get('onnx_model'),
            caffemodel_file=self.conf.get('caffemodel_file'),
            prototxt_file=self.conf.get('prototxt_file'),
            march=self.march,
            name_prefix=name_prefix,
            verbose=False,
            **params)
        return None

    def build(self) -> None:
        params = self.get_quantize_params()
        self.build_model(params=params)
