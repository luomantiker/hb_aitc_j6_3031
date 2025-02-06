# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from dataclasses import dataclass, field
from typing import Any, Literal

from onnx.onnx_pb import ModelProto


@dataclass
class ConfigBase:
    def get(self, name: str, value: Any = None) -> Any:
        res = getattr(self, name, value)
        return res

    def __getitem__(self, item: str) -> Any:
        res = getattr(self, item)
        if res is None:
            raise KeyError(item)
        return res

    def __setitem__(self, item_name: str, item_value: Any) -> None:
        setattr(self, item_name, item_value)


@dataclass
class ConfigInfo(ConfigBase):
    """Compile info after params_parser
    """
    # model_parameters after parsing
    model: ModelProto = None
    onnx_model: str = None
    caffemodel_file: str = None
    load_model: ModelProto = None
    prototxt_file: str = None
    model_type: str = None
    march: str = None
    input_num: int = 0
    working_dir: str = None
    output_model_file_prefix_full: str = None
    output_model_file_prefix: str = None
    output_nodes: list = field(default_factory=list)
    model_debug_mode: list = None
    node_dict: dict = field(default_factory=dict)
    layer_out_dump: bool = False
    # input_parameters after parsing
    input_names: list = field(default_factory=list)
    input_shapes: list = field(default_factory=list)
    input_batches: list = field(default_factory=list)
    separate_batch: bool = False
    separate_names: list = field(default_factory=list)
    input_type_rt: list = field(default_factory=list)
    input_type_train: list = field(default_factory=list)
    input_space_and_range: list = field(default_factory=list)
    input_layout_train: list = field(default_factory=list)
    input_layout_rt: list = field(default_factory=list)
    from_color: list = field(default_factory=list)
    to_color: list = field(default_factory=list)
    norm_type: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    scale: list = field(default_factory=list)
    std: list = field(default_factory=list)
    # calibration_parameters after parsing
    calibration_type: str = 'skip'
    per_channel: bool = False
    max_percentile: float = 0
    run_on_cpu: list = field(default_factory=list)
    run_on_bpu: list = field(default_factory=list)
    calibration_optimization: list = field(default_factory=list)
    cal_data_type: list = field(default_factory=list)
    cal_data_dir: list = field(default_factory=list)
    quant_config: str = None
    # compiler_parameters after parsing
    compile_mode: Literal['bandwidth', 'latency', 'balance'] = 'latency'
    balance_factor: int = None
    optimize_level: str = 'O2'
    compile_debug_mode: bool = True
    hbdk3_compatible_mode: bool = False
    ability_entry: str = None
    core_num: int = 1
    max_time_per_fc: int = 0
    jobs: int = 16
    advice: int = 0
    custom_op: bool = False
    custom_op_method: str = None
    cop_register_files: list = field(default_factory=list)
    custom_op_dir: str = None
    remove_node_name: list = field(default_factory=list)
    remove_node_type: list = field(default_factory=list)
    input_source: dict = field(default_factory=dict)
    first_cali_data: dict = field(default_factory=dict)
    # temp! it is used to add undefined calibration params
    cali_extra_param: dict = field(default_factory=dict)
    compile_extra_params: dict = field(default_factory=dict)
    enable_vpu: bool = False


@dataclass
class ModelBuildInfo(ConfigBase):
    """Model build info
    """
    # build info
    BUILDER_VERSION: str = None
    HBDK_VERSION: str = None
    HBDK_RUNTIME_VERSION: str = None
    HORIZON_NN_VERSION: str = None
    # model_parameters info
    CAFFE_MODEL: str = None
    PROTOTXT: str = None
    ONNX_MODEL: str = None
    MARCH: str = None
    LAYER_OUT_DUMP: str = None
    LOG_LEVEL: str = None
    WORKING_DIR: str = None
    MODEL_PREFIX: str = None
    OUTPUT_NODES: str = None
    REMOVE_NODE_TYPE: str = None
    REMOVE_NODE_NAME: str = None
    DEBUG_MODE: str = None
    NODE_INFO: str = None
    # input_parameters info
    INPUT_NAMES: str = None
    INPUT_SPACE_AND_RANGE: str = None
    INPUT_TYPE_RT: str = None
    INPUT_TYPE_TRAIN: str = None
    INPUT_LAYOUT_TRAIN: str = None
    INPUT_LAYOUT_RT: str = None
    NORM_TYPE: str = None
    MEAN_VALUE: str = None
    SCALE_VALUE: str = None
    STD_VALUE: str = None
    INPUT_SHAPE: str = None
    INPUT_BATCH: str = None
    SEPARATE_BATCH: str = None
    SEPARATE_NAME: str = None
    # custom op info
    CUSTOM_OP_METHOD: str = None
    CUSTOM_OP_DIR: str = None
    CUSTOM_OP_REGISTER_FILES: str = None
    # calibration_parameters info
    OPTIMIZATION: str = None
    CALI_TYPE: str = None
    CAL_DATA_DIR: str = None
    PER_CHANNEL: str = None
    MAX_PERCENTILE: str = None
    RUN_ON_CPU: str = None
    RUN_ON_BPU: str = None
    QUANT_CONFIG: str = None
    # compiler_parameters info
    ADVICE: int = 0
    DEBUG: str = None
    OPTIMIZE_LEVEL: str = None
    COMPILE_MODE: str = None
    CORE_NUM: int = 1
    MAX_TIME_PER_FC: int = 0
    BALANCE_FACTOR: int = 0
    ABILITY_ENTRY: str = None
    INPUT_SOURCE: dict = None
    hbdk3_compatible_mode: str = None
    # temp! it is used to add undefined calibration params
    CALI_EXTRA_PARAM: dict = None
    EXTRA_PARAMS: dict = None
