# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from schema import And, Optional, Or, Use

conf_schema_dict = {
    'model_parameters': {
        Optional('onnx_model'): str,
        Optional('caffe_model'): str,
        Optional('prototxt'): str,
        Optional('march'): str,  # required
        Optional('log_level'): str,
        Optional('layer_out_dump', default=False): bool,
        Optional('working_dir', default='./model_output'): str,
        Optional('output_model_file_prefix', default='model'): str,
        Optional('output_nodes'): str,
        Optional('remove_node_type'): str,
        Optional('remove_node_name'): str,
        Optional('set_node_data_type'): Or(str, dict),
        Optional('debug_mode'): Or(str, dict),
        Optional('node_info'): Or(str, dict),
    },
    'input_parameters': {
        Optional('input_name'): Use(str),
        Optional('input_type_rt'): str,  # required
        Optional('input_space_and_range'): Use(str),
        Optional('input_layout_rt'): Use(str),
        Optional('input_type_train'): str,  # required
        Optional('input_layout_train'): str,  # required
        Optional('norm_type'): Use(str),
        Optional('input_shape'): Use(str),
        Optional('input_batch'): Use(str),
        Optional('mean_value'): Use(str),
        Optional('scale_value'): Use(str),
    },
    Optional('custom_op'): {
        Optional("op_register_files"): str,
        Optional("custom_op_method"): str,
        Optional("custom_op_dir"): str,
    },
    Optional('calibration_parameters', default={}): {
        Optional('cal_data_dir'): And(str, len),
        Optional('calibration_type'): str,
        Optional('preprocess_on'): bool,
        Optional('per_channel'): bool,
        Optional('max_percentile'): Use(float),
        Optional('run_on_cpu'): Use(str),
        Optional('run_on_bpu'): Use(str),
        Optional('enable_int16'): bool,
        Optional('optimization'): str,
        Optional('cal_data_type'): str,
    },
    Optional('compiler_parameters', default={}): {
        Optional('compile_mode'): str,
        Optional('balance_factor'): int,
        Optional('debug'): bool,
        Optional('optimize_level'): str,
        Optional("ability_entry"): str,
        Optional("core_num"): int,
        Optional("max_time_per_fc"): int,
        Optional("jobs"): int,
        Optional('input_source'): Or(str, dict),
        Optional('advice'): int,
    },
}
