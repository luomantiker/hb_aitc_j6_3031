# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from schema import And, Optional, Or, Use

schema_yaml = {
    'model_parameters': {
        Optional('onnx_model', default=None): Use(str),
        Optional('caffe_model', default=None): Use(str),
        Optional('prototxt', default=None): Use(str),
        Optional('march', default=""): Use(str),  # required
        Optional('log_level'): Use(int),
        Optional('layer_out_dump', default=False): Or(bool, str, int),
        Optional('working_dir', default='./model_output'): Use(str),
        Optional('output_model_file_prefix', default='model'): Use(str),
        Optional('output_nodes', default=None): Use(str),
        Optional('remove_node_type', default=None): Use(str),
        Optional('remove_node_name', default=None): Use(str),
        Optional('node_info', default=None): Or(str, dict),
        Optional('debug_mode', default=None): Or(str),
        Optional('enable_vpu', default=False): Or(bool),
    },
    # Optional('input_parameters', default=""): {
    'input_parameters': {
        Optional('input_name', default=""): Use(str),
        Optional('input_type_rt', default=""): Use(str),
        Optional('input_space_and_range', default=""): Use(str),
        Optional('input_type_train', default=""): Use(str),
        Optional('input_layout_rt', default=""): Use(str),  # deprecated
        Optional('input_layout_train', default=""): Use(str),
        Optional('norm_type', default=None): Use(str),
        Optional('input_shape', default=None): Use(str),
        Optional('input_batch', default=None): Use(str),
        Optional('separate_batch', default=False): Or(bool, str, int),
        Optional('separate_name', default=None): Use(str),
        Optional('mean_value', default=None): Use(str),
        Optional('scale_value', default=None): Use(str),
        Optional('std_value', default=None): Use(str),
    },
    # Optional('calibration_parameters', default=object): {
    'calibration_parameters': {
        Optional('cal_data_dir', default=None): And(str, len),
        Optional('calibration_type', default="default"): Use(str),
        Optional('preprocess_on'): Or(bool, str, int),
        Optional('per_channel', default=False): Or(bool, str, int),
        Optional('max_percentile', default=None): Or(None, float),
        Optional('run_on_cpu', default=None): Use(str),
        Optional('run_on_bpu', default=None): Use(str),
        Optional('optimization', default=None): Use(str),
        Optional('cal_data_type', default=None): Use(str),
        Optional('quant_config', default=None): Or(str, dict),
        Optional(str): object,
    },
    # Optional('compiler_parameters', default=object): {
    'compiler_parameters': {
        Optional('compile_mode', default="latency"): Use(str),
        Optional('debug', default=True): Or(bool, str, int),
        Optional('optimize_level', default="O2"): Use(str),
        Optional("ability_entry", default=None): Use(str),
        Optional("core_num", default=1): Use(int),
        Optional("max_time_per_fc", default=0): Use(int),
        Optional("jobs", default=16): Use(int),
        Optional('input_source', default={}): Or(str, dict),
        Optional("advice", default=0): Use(int),
        Optional("balance_factor", default=0): Use(int),
        Optional('hbdk3_compatible_mode', default=False): Or(bool, str, int),
        Optional('extra_params', default={}): Use(dict),
    },
    # Optional('custom_op', default=object): {
    'custom_op': {
        Optional("op_register_files", default=None): Use(str),
        Optional("custom_op_method", default=None): Use(str),
        Optional("custom_op_dir", default=None): Use(str),
    },
}
