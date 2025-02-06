#  Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#  #
#  The material in this file is confidential and contains trade secrets
#  of Horizon Robotics Inc. This is proprietary information owned by
#  Horizon Robotics Inc. No part of this work may be disclosed,
#  reproduced, copied, transmitted, or used in any way for any purpose,
#  without the express written permission of Horizon Robotics Inc.

import logging
import os
from typing import Literal

from hbdk4.compiler import Module, hbm_perf, save
from onnx.onnx_pb import ModelProto

from horizon_tc_ui.config.mapper_consts import preprocess_mode_map
from horizon_tc_ui.config.params_parser import ConfigInfo
from horizon_tc_ui.hb_model_info import hbm_model_info
from horizon_tc_ui.hbir_handle import HBIRHandle
from horizon_tc_ui.hbm_handle import HBMHandle
from horizon_tc_ui.utils.node_info import NodeInfo
from horizon_tc_ui.utils.wrap_utils import try_except_wrapper


class HBMBuilder:
    def __init__(self,
                 ptq_model: ModelProto,
                 march: str,
                 config_info: ConfigInfo,
                 skip: Literal["export", "convert", "compile"] = None) -> None:
        self.ptq_model = ptq_model
        self.march = march
        self.config_info = config_info
        self.skip = skip
        self.name_prefix = self.config_info.output_model_file_prefix
        self.float_hbir: Module = None
        self.quantize_hbir: Module = None
        self.hbm_path = None
        self.hbir_handler: HBIRHandle = None
        self.workspace = self.config_info.working_dir
        if not self.workspace:
            self.workspace = os.getcwd()
        self.hbm_save_path = os.path.join(self.workspace,
                                          self.name_prefix) + '.hbm'

    def _get_compile_params(self) -> dict:
        compile_params = {}
        compile_params["jobs"] = self.config_info.jobs
        compile_params["balance"] = self.config_info.balance_factor
        if self.config_info.advice:
            compile_params["advice"] = self.config_info.advice

        compile_params[
            "hbdk3_compatible_mode"] = self.config_info.hbdk3_compatible_mode
        compile_params["progress_bar"] = True
        compile_params["opt"] = int(self.config_info.optimize_level[-1:])
        compile_params["max_time_per_fc"] = int(
            self.config_info.max_time_per_fc)
        compile_params["debug"] = self.config_info.compile_debug_mode
        extra_params = self.config_info.compile_extra_params
        if extra_params and extra_params.keys():
            for key, value in extra_params.items():
                compile_params[key] = value
        logging.debug(f'Compile params: {compile_params}')
        return compile_params

    def remove_node(self) -> None:
        handle = HBIRHandle(self.quantize_hbir, func_index=0)
        status = handle.remove_io_op(
            remove_node_types=self.config_info.remove_node_type,
            remove_node_names=self.config_info.remove_node_name)
        if status:
            self.save(handle.model, suffix="_quantized_removed_model.bc")
        return None

    def insert_image_preprocess(self, config_idx: int, input_type_rt: str,
                                input_type_train: str,
                                insert_idx: int) -> None:
        model_input_name = self.hbir_handler.func.flatten_inputs[
            insert_idx].name
        c_dim = 3 if input_type_rt != 'gray' else 1
        if input_type_rt == 'nv12' and \
                self.config_info.input_space_and_range[config_idx]:
            if self.config_info.input_space_and_range[config_idx] == "regular":
                input_type_rt += '_full'
            else:
                input_type_rt += '_video'
        convert_type = input_type_rt + '_' + input_type_train

        if convert_type not in preprocess_mode_map and input_type_rt != input_type_train:  # noqa
            raise ValueError('This type of image conversion is not supported. '
                             f'input_type_train: {input_type_train},'
                             f'input_type_rt: {input_type_rt}')
        elif input_type_rt == input_type_train:
            logging.warning(
                'When input_type_train is %s '
                'and input_type_rt is %s, '
                'the image preprocess op '
                'will not have a color conversion operation', input_type_train,
                input_type_rt)

        mode = preprocess_mode_map.get(convert_type, 'skip')
        logging.warning("Insert image preprocess op for %s at input %s",
                        model_input_name, insert_idx)
        mean = [0.0] * c_dim
        if 'mean' in self.config_info.norm_type[config_idx]:
            mean = self.config_info.mean[config_idx]
            if len(mean) == 1:
                mean = [mean[0]] * c_dim
        std = [1.0] * c_dim
        if 'scale' in self.config_info.norm_type[config_idx]:
            scale = self.config_info.scale[config_idx]
            if len(scale) == 1:
                scale = [scale[0]] * c_dim
            std = [1 / _scale for _scale in scale]
        elif 'std' in self.config_info.norm_type[config_idx]:
            std = self.config_info.std[config_idx]
            if len(std) == 1:
                std = [std[0]] * c_dim
        self.hbir_handler.insert_image_preprocess(index=insert_idx,
                                                  mode=mode,
                                                  divisor=1,
                                                  mean=mean,
                                                  std=std,
                                                  is_signed=True)

    def insert_nodes(self, ori_idx: int, insert_idx: int) -> int:
        insert_count = 0
        model_input_name = self.hbir_handler.func.flatten_inputs[
            insert_idx].name
        ori_input_name = self.config_info.input_names[ori_idx]
        input_source = self.config_info.input_source.get(ori_input_name, {})
        input_type_train = self.config_info.input_type_train[ori_idx]
        input_type_rt = self.config_info.input_type_rt[ori_idx]

        if input_type_rt == 'featuremap':
            return 1

        if self.config_info.input_layout_train[ori_idx] == "NCHW":
            logging.warning('Insert transpose op for %s at input %s',
                            model_input_name, insert_idx)
            self.hbir_handler.insert_transpose(index=insert_idx,
                                               permutes=[0, 3, 1, 2])

        self.insert_image_preprocess(config_idx=ori_idx,
                                     input_type_rt=input_type_rt,
                                     input_type_train=input_type_train,
                                     insert_idx=insert_idx)

        if input_source == 'pyramid':
            logging.warning('Insert image convert op for %s at input %s',
                            model_input_name, insert_idx)
            self.hbir_handler.insert_image_convert(index=insert_idx,
                                                   mode=input_type_rt)
            insert_count = 2 if input_type_rt != "gray" else 1
        elif input_source == 'resizer':
            logging.warning('Insert roi resize op for %s at input %s',
                            model_input_name, insert_idx)
            self.hbir_handler.insert_roi_resize(index=insert_idx,
                                                mode=input_type_rt)
            insert_count = 3 if input_type_rt != "gray" else 2
        elif input_source == 'ddr':
            insert_count = 1
        return insert_count

    def input_insert_nodes_based_name(self) -> None:
        model_input_names = [
            i.name for i in self.hbir_handler.func.flatten_inputs
        ]
        logging.debug("current input_names: %s", model_input_names)
        insert_idx = 0
        for model_input_name in model_input_names:

            input_source = self.config_info.input_source.get(
                model_input_name, {})
            logging.debug("%s input_source %s", model_input_name, input_source)
            ori_idx = self.config_info.input_names.index(model_input_name)

            insert_idx += self.insert_nodes(ori_idx=ori_idx,
                                            insert_idx=insert_idx)
        return None

    def input_insert_nodes_based_separate_batch(self) -> None:
        separate_mapping = {}
        model_input_names = [
            i.name for i in self.hbir_handler.func.flatten_inputs
        ]
        for input_name in model_input_names:
            input_idx = [
                i.name for i in self.hbir_handler.func.flatten_inputs
            ].index(input_name)
            input_batch = int(self.config_info.input_batches[0])

            self.hbir_handler.insert_split(index=input_idx, dim=0)
            logging.info("Split input %s with input_batch %s on dim %s",
                         input_name, input_batch, 0)
            for i in range(input_batch):
                separate_mapping[f"{input_name}_{i}"] = input_name

        insert_idx = 0
        model_input_names = [
            i.name for i in self.hbir_handler.func.flatten_inputs
        ]
        for model_input_name in model_input_names:
            ori_input_name = separate_mapping.get(model_input_name,
                                                  model_input_name)
            ori_idx = self.config_info.input_names.index(ori_input_name)
            insert_idx += self.insert_nodes(ori_idx=ori_idx,
                                            insert_idx=insert_idx)

        return None

    def input_insert_nodes_based_separate_name(self) -> None:
        separate_mapping = {}
        model_input_names = [
            i.name for i in self.hbir_handler.func.flatten_inputs
        ]
        insert_split_index = 0
        for _, input_name in enumerate(model_input_names):
            if input_name not in self.config_info.separate_names:
                separate_mapping[input_name] = input_name
                insert_split_index += 1
                continue
            if self.config_info.input_batches:
                input_batch = int(self.config_info.input_batches[0])
            else:
                input_name_idx = self.config_info.input_names.index(input_name)
                input_batch = int(
                    self.config_info.input_shapes[input_name_idx][0])  # noqa

            self.hbir_handler.insert_split(index=insert_split_index, dim=0)
            insert_split_index += input_batch
            logging.info("Split input %s with input_batch %s on dim %s",
                         input_name, input_batch, 0)
            for i in range(input_batch):
                separate_mapping[f"{input_name}_{i}"] = input_name
        insert_idx = 0
        model_input_names = [
            i.name for i in self.hbir_handler.func.flatten_inputs
        ]
        for model_input_name in model_input_names:
            ori_input_name = separate_mapping.get(model_input_name,
                                                  model_input_name)
            ori_idx = self.config_info.input_names.index(ori_input_name)
            insert_idx += self.insert_nodes(ori_idx=ori_idx,
                                            insert_idx=insert_idx)
        return None

    def input_insert_nodes(self) -> None:
        if self.config_info.separate_batch:
            # input_batch parameter only supports single input models
            # default splitting of the first dimension of the first input
            self.input_insert_nodes_based_separate_batch()
        elif self.config_info.separate_names:
            self.input_insert_nodes_based_separate_name()
        else:
            self.input_insert_nodes_based_name()
        return None

    def node_info(self) -> None:
        node_info_handle = NodeInfo(self.config_info)
        node_info_handle.generate_info()
        node_info_handle.print_node_info()
        node_info_handle.dump_node_info()
        if node_info_handle.mode == "full":
            node_info_handle.print_output_tensor_info()
        return None

    def export_model(self) -> None:
        logging.info("Start to export model.")
        self.float_hbir = HBIRHandle.export_hbir(
            ptq_model=self.ptq_model,
            output_prefix_name=self.config_info.output_model_file_prefix)
        logging.info("Successful export model.")

    def convert_model(self) -> None:
        self.hbir_handler = HBIRHandle(model=self.float_hbir, func_index=0)
        self.hbir_handler.set_model_info(config_info=self.config_info)
        if os.environ.get("HORIZON_TC_UI_DEBUG"):
            self.save(m=self.hbir_handler.model, suffix='_ptq_model.bc')
        self.input_insert_nodes()

        advice_path = self.config_info.working_dir + "/"
        logging.info("Start to convert model.")
        self.quantize_hbir = self.hbir_handler.convert_quantize_model(
            self.march,
            advice=True,
            advice_path=advice_path,
            enable_vpu=self.config_info.enable_vpu)
        self.save(self.quantize_hbir, suffix='_quantized_model.bc')
        logging.info("Successful convert model.")
        return None

    @try_except_wrapper("hbdk.save", ['m'])
    def save(self, m: Module, suffix: str) -> None:
        workspace = os.path.join(self.workspace, self.name_prefix)
        save_path = workspace + suffix
        save(m, save_path)
        return None

    def compile_model(self) -> None:
        compile_params = self._get_compile_params()
        compile_handler = HBIRHandle(model=self.quantize_hbir, func_index=0)
        logging.info("Start to compile model.")
        compile_handler.compile_model(save_path=self.hbm_save_path,
                                      march=self.march,
                                      compile_params=compile_params)
        logging.info("Successful compile hbm %s", self.hbm_save_path)
        return None

    def update_hbm_desc(self) -> None:
        desc = self.hbir_handler.convert_model_info(
            config_info=self.config_info)
        hbm_handle = HBMHandle(self.hbm_save_path)
        hbm_handle.update_desc(model_name=self.name_prefix, desc=desc)

    @try_except_wrapper("hbdk.compiler.hbm_perf")
    def hbm_perf(self) -> None:
        hbm_perf(model=self.hbm_save_path, output_dir=self.workspace)
        logging.info("Successful perf model, result generate at %s",
                     self.workspace)

    def print_model_info(self) -> None:
        hbm_model_info(model_path=self.hbm_save_path)

    def build(self) -> None:
        if self.skip == "export":
            return None
        self.export_model()
        if self.skip == "convert":
            return None
        self.convert_model()
        self.node_info()
        self.remove_node()
        if self.skip == "compile":
            return None
        self.compile_model()
        self.hbm_perf()
        self.print_model_info()
