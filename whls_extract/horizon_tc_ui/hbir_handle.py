#  Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#  #
#  The material in this file is confidential and contains trade secrets
#  of Horizon Robotics Inc. This is proprietary information owned by
#  Horizon Robotics Inc. No part of this work may be disclosed,
#  reproduced, copied, transmitted, or used in any way for any purpose,
#  without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
import time
from typing import List

import onnx
from hbdk4.compiler import compile, convert, statistics
from hbdk4.compiler.extra_apis import internal_compile
from hbdk4.compiler.onnx import export
from hbdk4.compiler.overlay import Module
from hbdk4.compiler.utils.visualize import OnnxConvertor
from hbdk4.compiler.version import VERSION as hbdk_version
from horizon_nn.api import version as horizon_nn_version

from horizon_tc_ui import __version__ as horizon_tc_ui_version
from horizon_tc_ui.config.config_info import ConfigInfo, ModelBuildInfo
from horizon_tc_ui.utils.tool_utils import get_str_from_list, tabulate
from horizon_tc_ui.utils.wrap_utils import try_except_wrapper


class HBIRHandle:
    def __init__(self, model: Module, func_index: int = 0) -> None:
        self.model = model
        self.module = model.module
        self.func = model.functions[func_index]

        self.current_phase = None
        self.desc = {}
        self.check_current_phase()
        self.parse_desc()

    def check_current_phase(self) -> None:
        for op in self.func.operations:
            if op.type == 'qnt.const_fake_quant':
                self.current_phase = 'export'

    def parse_desc(self) -> None:
        try:
            if not self.func.desc:
                return
            self.desc = json.loads(self.func.desc)
        except Exception as e:
            logging.warning("Model desc parse failed, error log: %s", e)

    @staticmethod
    @try_except_wrapper(module_info="hbdk.export", ignore_keys=["ptq_model"])
    def export_hbir(ptq_model: onnx.ModelProto,
                    output_prefix_name: str) -> Module:
        return export(proto=ptq_model, name=output_prefix_name)

    @try_except_wrapper(module_info="hbdk.convert")
    def convert_quantize_model(self,
                               march: str,
                               advice: bool,
                               advice_path: str,
                               enable_vpu: bool = False) -> Module:
        return convert(m=self.model,
                       march=march,
                       advice=advice,
                       advice_path=advice_path,
                       enable_vpu=enable_vpu)

    @try_except_wrapper(module_info="hbdk.compile")
    def compile_model(self, save_path: str, march: str,
                      compile_params: dict) -> None:
        start = time.time()
        compile(m=self.model, path=save_path, march=march, **compile_params)
        if os.environ.get("HORIZON_TC_UI_DEBUG"):
            logging.info(
                f"*** HBM-COMPILE-SPENT-TIME {time.time() - start:.2f} ***")

    @try_except_wrapper(module_info="hbdk.internal_compile")
    def internal_compile_model(self, save_path: str, march: str,
                               compile_params: dict) -> None:
        internal_compile(m=self.model,
                         perf_output_dir=save_path,
                         march=march,
                         opt=compile_params.get('opt', 0),
                         jobs=compile_params.get('jobs', 0),
                         progress_bar=True,
                         advice=compile_params.get('advice', 0.0),
                         balance=compile_params.get('balance', 100))

    @try_except_wrapper(module_info="hbdk.statistics")
    def print_statistics(self) -> None:
        statistics(self.model)

    def removable_node(self) -> None:
        removable_res = self.get_removable_info(self.model)
        logging.info(f"Removable node: {removable_res}")
        return None

    def get_type_for_hbtl_call(self, attached_op) -> str:
        schema = attached_op.schema
        node_type = attached_op.type + "::" + \
            schema.namespace + "::" + schema.signature
        return node_type

    @try_except_wrapper(module_info="hbdk.insert_split")
    def insert_split(self, index: int, dim: int) -> None:
        self.func.flatten_inputs[index].insert_split(dim=dim)

    @try_except_wrapper(module_info="hbdk.insert_image_preprocess")
    def insert_image_preprocess(self, index: int, mode: str, divisor: int,
                                mean: List[float], std: List[float],
                                is_signed: bool) -> None:
        logging.debug('before insert_image_preprocess %s', self.func)
        self.func.flatten_inputs[index].insert_image_preprocess(
            mode=mode,
            divisor=divisor,
            mean=mean,
            std=std,
            is_signed=is_signed)
        logging.debug('after insert_image_preprocess %s', self.func)

    @try_except_wrapper(module_info="hbdk.insert_image_convert")
    def insert_image_convert(self, index: int, mode: str) -> None:
        self.func.flatten_inputs[index].insert_image_convert(mode=mode)

    @try_except_wrapper(module_info="hbdk.insert_transpose")
    def insert_transpose(self, index: int, permutes: List[int]) -> None:
        self.func.flatten_inputs[index].insert_transpose(permutes=permutes)

    @try_except_wrapper(module_info="hbdk.insert_roi_resize")
    def insert_roi_resize(self, index: int, mode: str) -> None:
        self.func.flatten_inputs[index].insert_roi_resize(mode=mode)

    @try_except_wrapper("hbdk.overlay.remove_io_op")
    def remove_io_op(self, remove_node_types: list,
                     remove_node_names: list) -> bool:
        removable_res = self.func.remove_io_op(op_names=remove_node_names,
                                               op_types=remove_node_types)
        if removable_res:
            logging.info("Successfully remove node, the removed node info is:")
            removable_res.insert(0, ("Node Name", "Type"))
            tabulate(removable_res)
            return True
        return False

    def remove(self, remove_node_types: list, remove_node_names: list) -> bool:
        if not remove_node_types and not remove_node_names:
            return False

        # obtain information about the nodes that can be deleted
        removable_res = self.get_removable_function_io(
            op_types=remove_node_types, op_names=remove_node_names)
        if len(removable_res) == 0:
            logging.warning("No nodes can be deleted")
            return False

        logging.info(f"Removable node: {removable_res}")

        self.remove_function_io(op_types=remove_node_types,
                                op_names=remove_node_names)
        self.remove(remove_node_types, remove_node_names)
        return True

    @try_except_wrapper(module_info="hbdk.set_func_desc")
    def set_model_info(self, config_info: ConfigInfo) -> None:
        self.func.desc = self.convert_model_info(config_info=config_info)

    @try_except_wrapper(module_info="hbdk.hbir.visualize")
    def visualize(self, save_path) -> None:
        """Use hbdk api to generate a onnx file

        Args:
            save_path (str): onnx file save path
        """
        cvt = OnnxConvertor(func=self.func)
        cvt.gen_onnx(save_path)
        logging.info(f"Successfule generate onnx to {save_path}")

    @staticmethod
    def convert_model_info(config_info: ConfigInfo) -> str:
        """Convert ConfigInfo to ModelBuildInfo

        Args:
            config_info (ConfigInfo): model config info

        Returns:
            str: json strings, structure is ModelBuildInfo
        """
        build_info = ModelBuildInfo()
        # build info
        build_info.BUILDER_VERSION = horizon_tc_ui_version
        build_info.HORIZON_NN_VERSION = horizon_nn_version
        build_info.HBDK_VERSION = hbdk_version
        # model_parameters info
        build_info.CAFFE_MODEL = config_info.caffemodel_file
        build_info.PROTOTXT = config_info.prototxt_file
        build_info.ONNX_MODEL = config_info.onnx_model
        build_info.MARCH = config_info.march
        build_info.LAYER_OUT_DUMP = str(config_info.layer_out_dump)
        build_info.WORKING_DIR = config_info.working_dir
        build_info.MODEL_PREFIX = config_info.output_model_file_prefix
        build_info.OUTPUT_NODES = get_str_from_list(config_info.output_nodes)
        build_info.REMOVE_NODE_TYPE = get_str_from_list(
            config_info.get('remove_node_type'))
        build_info.REMOVE_NODE_NAME = get_str_from_list(
            config_info.get('remove_node_name'))
        build_info.DEBUG_MODE = get_str_from_list(config_info.model_debug_mode)
        build_info.NODE_INFO = str(config_info.node_dict)
        # input_parameters info
        build_info.INPUT_NAMES = get_str_from_list(config_info.input_names)
        build_info.INPUT_SPACE_AND_RANGE = get_str_from_list(
            config_info.input_space_and_range)
        build_info.INPUT_TYPE_RT = get_str_from_list(config_info.input_type_rt)
        build_info.INPUT_TYPE_TRAIN = get_str_from_list(
            config_info.input_type_train)
        build_info.INPUT_LAYOUT_TRAIN = get_str_from_list(
            config_info.input_layout_train)
        build_info.INPUT_LAYOUT_RT = get_str_from_list(
            config_info.input_layout_rt)
        build_info.NORM_TYPE = get_str_from_list(config_info.norm_type)
        build_info.MEAN_VALUE = get_str_from_list(config_info.mean)
        build_info.SCALE_VALUE = get_str_from_list(config_info.scale)
        build_info.STD_VALUE = get_str_from_list(config_info.std)
        build_info.INPUT_SHAPE = get_str_from_list(
            ["x".join([str(y) for y in x]) for x in config_info.input_shapes])
        build_info.INPUT_BATCH = get_str_from_list(config_info.input_batches)
        build_info.SEPARATE_BATCH = str(config_info.separate_batch)
        build_info.SEPARATE_NAME = get_str_from_list(
            config_info.separate_names)
        # custom op info
        build_info.CUSTOM_OP_METHOD = config_info.custom_op_method
        build_info.CUSTOM_OP_DIR = config_info.custom_op_dir
        build_info.CUSTOM_OP_REGISTER_FILES = get_str_from_list(
            config_info.get('cop_register_files', []))
        # calibration_parameters info
        build_info.OPTIMIZATION = get_str_from_list(
            config_info.calibration_optimization)
        build_info.CALI_TYPE = config_info.calibration_type
        build_info.CAL_DATA_DIR = get_str_from_list(config_info.cal_data_dir)
        build_info.PER_CHANNEL = str(config_info.per_channel)
        build_info.MAX_PERCENTILE = str(config_info.max_percentile)
        build_info.RUN_ON_BPU = get_str_from_list(config_info.run_on_bpu)
        build_info.RUN_ON_CPU = get_str_from_list(config_info.run_on_cpu)
        build_info.CALI_EXTRA_PARAM = config_info.cali_extra_param
        build_info.QUANT_CONFIG = config_info.quant_config
        # compiler_parameters info
        build_info.DEBUG = str(config_info.compile_debug_mode)
        build_info.OPTIMIZE_LEVEL = config_info.optimize_level
        build_info.COMPILE_MODE = config_info.compile_mode
        build_info.CORE_NUM = config_info.core_num
        build_info.MAX_TIME_PER_FC = config_info.max_time_per_fc
        build_info.ADVICE = config_info.advice
        build_info.BALANCE_FACTOR = config_info.balance_factor
        build_info.ABILITY_ENTRY = config_info.ability_entry
        build_info.INPUT_SOURCE = config_info.input_source
        build_info.hbdk3_compatible_mode = str(
            config_info.hbdk3_compatible_mode)
        build_info.EXTRA_PARAMS = config_info.compile_extra_params
        json_string = json.dumps(vars(build_info))
        return json_string

    def update_desc_from_bc(self, config_info: ConfigInfo) -> None:
        try:
            desc = json.loads(self.func.desc)
        except (json.decoder.JSONDecodeError, TypeError):
            return
        desc['BUILDER_VERSION'] = horizon_tc_ui_version
        desc['HBDK_VERSION'] = hbdk_version
        desc['REMOVE_NODE_TYPE'] = get_str_from_list(
            config_info.get('remove_node_type'))
        desc['REMOVE_NODE_NAME'] = get_str_from_list(
            config_info.get('remove_node_name'))
        desc['DEBUG'] = str(config_info.compile_debug_mode)
        if desc.get('OPTIMIZATION_LEVEL'):
            del desc['OPTIMIZATION_LEVEL']
        desc['OPTIMIZE_LEVEL'] = config_info.optimize_level
        desc['COMPILE_MODE'] = config_info.compile_mode
        desc['CORE_NUM'] = config_info.core_num
        desc['MAX_TIME_PER_FC'] = config_info.max_time_per_fc
        desc['ADVICE'] = config_info.advice
        desc['WORKING_DIR'] = config_info.working_dir
        desc['MODEL_PREFIX'] = config_info.output_model_file_prefix
        desc['BALANCE_FACTOR'] = config_info.balance_factor
        desc['ABILITY_ENTRY'] = config_info.ability_entry
        desc['INPUT_SOURCE'] = config_info.input_source
        desc['EXTRA_PARAMS'] = config_info.compile_extra_params
        self.func.desc = json.dumps(desc)
        return None

    def update_deleted_node_info(self, node_type: str) -> None:
        desc = json.loads(self.func.desc)
        deleted_info = desc.get("DELETED_NODE_INFO", [])
        deleted_info.append(node_type)
        desc["DELETED_NODE_INFO"] = deleted_info
        self.func.desc = json.dumps(desc)
        return None

    def convert_model_info_to_dict(self) -> dict:
        """Covnert model info to model build yaml

        Returns:
            dict: model build yaml
        """
        model_info = json.loads(self.func.desc)
        if model_info.get('CALI_DIR'):
            model_info['CAL_DATA_DIR'] = model_info.get('CALI_DIR')
        if model_info.get('OPTIMIZATION_LEVEL'):
            model_info['OPTIMIZE_LEVEL'] = model_info.get('OPTIMIZATION_LEVEL')
        if model_info.get('MAX_PERCENTILE') and model_info.get(
                'MAX_PERCENTILE') != 'None':  # noqa
            max_percentile = float(model_info.get('MAX_PERCENTILE'))
        else:
            max_percentile = None
        build_info = {
            'model_parameters': {
                'onnx_model': model_info.get('ONNX_MODEL'),
                'caffe_model': model_info.get('CAFFE_MODEL'),
                'prototxt': model_info.get('PROTOTXT'),
                'march': model_info.get('MARCH'),
                'layer_out_dump': bool(model_info.get('LAYER_OUT_DUMP')),
                'working_dir': model_info.get('WORKING_DIR'),
                'output_model_file_prefix': model_info.get('MODEL_PREFIX'),
                'output_nodes': model_info.get('OUTPUT_NODES'),
                'remove_node_type': model_info.get('REMOVE_NODE_TYPE'),
                'remove_node_name': model_info.get('REMOVE_NODE_NAME'),
                'debug_mode': model_info.get('DEBUG_MODE'),
            },
            'input_parameters': {
                'input_name':
                    model_info.get('INPUT_NAMES'),
                'input_type_rt':
                    model_info.get('INPUT_TYPE_RT'),
                'input_space_and_range':
                    model_info.get('INPUT_SPACE_AND_RANGE'),
                'input_type_train':
                    model_info.get('INPUT_TYPE_TRAIN'),
                'input_layout_rt':
                    model_info.get('INPUT_LAYOUT_RT'),
                'input_layout_train':
                    model_info.get('INPUT_LAYOUT_TRAIN'),
                'norm_type':
                    model_info.get('NORM_TYPE'),
                'input_shape':
                    model_info.get('INPUT_SHAPE'),
                'input_batch':
                    model_info.get('INPUT_BATCH'),
                'separate_batch':
                    bool(model_info.get('SEPARATE_BATCH')),
                'separate_name':
                    model_info.get('SEPARATE_NAME', []),
                'mean_value':
                    model_info.get('MEAN_VALUE'),
                'scale_value':
                    model_info.get('SCALE_VALUE'),
            },
            'calibration_parameters': {
                'cal_data_dir': model_info.get('CAL_DATA_DIR'),
                'calibration_type': model_info.get('CALI_TYPE'),
                'per_channel': bool(model_info.get('PER_CHANNEL')),
                'max_percentile': max_percentile,
                'run_on_cpu': model_info.get('RUN_ON_CPU'),
                'run_on_bpu': model_info.get('RUN_ON_BPU'),
                'optimization': model_info.get('OPTIMIZATION'),
                'cali_extra_param': model_info.get('CALI_EXTRA_PARAM'),
                'quant_config': model_info.get('QUANT_CONFIG')
            },
            'compiler_parameters': {
                'compile_mode':
                    model_info.get('COMPILE_MODE'),
                'debug':
                    bool(model_info.get('DEBUG')),
                'optimize_level':
                    model_info.get('OPTIMIZE_LEVEL'),
                'ability_entry':
                    model_info.get('ABILITY_ENTRY'),
                'core_num':
                    int(model_info.get('CORE_NUM', 1)),
                'max_time_per_fc':
                    int(model_info.get('MAX_TIME_PER_FC', 0)),
                'advice':
                    int(model_info.get('ADVICE', 0)),
                'hbdk3_compatible_mode':
                    bool(model_info.get('hbdk3_compatible_mode',
                                        False)),  # noqa
                'extra_params':
                    model_info.get('EXTRA_PARAMS', {}),
            },
            'custom_op': {
                'op_register_files':
                    model_info.get('CUSTOM_OP_REGISTER_FILES'),
                'custom_op_method':
                    model_info.get('CUSTOM_OP_METHOD'),
                'custom_op_dir':
                    model_info.get('CUSTOM_OP_DIR'),
            }
        }

        return build_info
