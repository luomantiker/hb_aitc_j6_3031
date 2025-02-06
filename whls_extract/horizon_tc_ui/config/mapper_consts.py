# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

# input parameters
# some unused types are maintained here for forward compatibility
# the list of currently supported input_type_rt is defined
# in the input_type_rt_open_list
input_type_rt_list = [
    'nv12',
    'yuv444',
    'yuv444_128',
    'featuremap',
    # 'rgbp',
    # 'bgrp',
    'rgb',
    'bgr',
    'gray',
    'yuv_bt601_full',
    'yuv_bt601_video'
]
input_type_rt_open_list = [
    'nv12',
    'yuv444',
    'rgb',
    'bgr',
    'gray',
    'featuremap',
]
input_type_train_list = [
    # 'rgbp',
    # 'bgrp',
    'rgb',
    'bgr',
    'featuremap',
    'gray',
    'yuv444',
    # 'yuv444p',
    'yuv444_128',
    'yuv_bt601_full',
    'yuv_bt601_video'
]

input_type_train_open_list = [
    'rgb',
    'bgr',
    'featuremap',
    'gray',
    'yuv444',
]

# key: train  value: rt
legal_trans_dict = {
    'rgbp': ['bgrp', 'nv12', 'yuv444', 'yuv444_128', 'yuv420sp_bt601_video'],
    'bgrp': ['rgbp', 'nv12', 'yuv444', 'yuv444_128', 'yuv420sp_bt601_video'],
    'rgb': ['bgr', 'nv12', 'yuv444', 'yuv444_128', 'yuv420sp_bt601_video'],
    'bgr': ['rgb', 'nv12', 'yuv444', 'yuv444_128', 'yuv420sp_bt601_video'],
    'yuv444': [
        'yuv444_128',
        'nv12',
    ],
    'yuv444p': [
        'yuv444_128',
        'nv12',
    ],
}

input_type_rt_parse_dict = {
    'yuv444': 'YUV444_128',
    'nv12': 'YUV444_128',
    'yuv444_128': 'YUV444_128',
    'rgbp': 'RGB_128',
    'bgrp': 'BGR_128',
    'rgb': 'RGB_128',
    'bgr': 'BGR_128',
    'gray': 'GRAY_128',
    'yuv420sp_bt601_video': 'YUV_BT601_Video_Range',
}

input_type_train_parse_dict = {
    'rgbp': 'RGB',
    'bgrp': 'BGR',
    'rgb': 'RGB',
    'bgr': 'BGR',
    'gray': 'GRAY',
    'yuv444': 'YUV444',
    'yuv444p': 'YUV444',
}

NHWC_types = [
    'rgb',
    'bgr',
    'yuv444',
    'yuv444_128',
]

NCHW_types = ['rgbp', 'bgrp', 'yuv444p', 'gray', 'featuremap']

NONE_types = ['nv12', 'yuv420sp_bt601_video']

norm_type_list = [
    'no_preprocess',
    'mean_file',  # deprecated
    'data_mean',
    'data_scale',
    'data_std',
    'mean_file_and_scale',  # deprecated
    'data_mean_and_scale',
    'data_mean_and_std',
]

layout_list = ['NHWC', 'NCHW']
autoq_caltype_list = ['kl', 'max', 'mix', 'default']
preq_caltype_list = ['skip']
march_list = ['nash-e', 'nash-m', 'nash-p']
run_on_list = ['cpu', 'bpu']
custom_op_method_list = ['register']
params_without_args = ['optimize_level', 'no_arg_option']
# defines the input_type_rt supported by each input_source
input_source_support_dict = {
    "pyramid": ['nv12', 'gray', 'yuv420sp_bt601_video', 'yuv_bt601_full'],
    "ddr": ['rgb', 'bgr', 'yuv444', 'yuv444_128', 'gray', 'featuremap'],
    'resizer': ['nv12', 'gray', 'yuv420sp_bt601_video', 'yuv_bt601_full']
}

BPU_OP_TYPES = [
    "HzQuantizedConv", "HzSQuantizedConv", "HzQuantizedConvTranspose",
    "HzSQuantizedConvTranspose", "HzQuantizedAveragePool",
    "HzQuantizedMaxPool", "HzQuantizedGlobalAveragePool",
    "HzQuantizedGlobalMaxPool", "HzLeakyRelu", "HzPRelu", "HzQuantize",
    "HzQuantizedResizeUpsample", "HzQuantizedPreprocess",
    "HzSQuantizedPreprocess", "HzQuantizedRoiResize", "HzLut",
    "HzDepthToSpace", "HzSQuantizedMul", "HzArgMax", "HzSegmentedLut",
    "HzQSoftmax", "HzQuantizedComparison"
]
BPU_OP_TYPES_UNIQUE = [
    "HzPad", "HzSpaceToDepth", "Split", "Slice", "HzChannelShuffle"
]

optimize_level = ['O0', 'O1', 'O2', 'O3']
optimize_level_hbdk4 = ['O0', 'O1', 'O2']

input_space_and_range = ['regular', 'bt601_video']

removal_list = [
    "Quantize", "Transpose", "Dequantize", "Cast", "Reshape", "Softmax"
]

# calibration_optimization parameter list
# not used now because horizon_NN will check optimization
cali_optimization_list = ['set_model_output_int8', 'set_model_output_int16']
cali_optimization_list_xj3 = ['set_model_output_int8']

input_source_range = ['pyramid', 'ddr', 'resizer']

core_num_range = [1, 2]

# calibration data type parameter list
cal_data_type_list = ['uint8', 'float32']

compile_mode_list = ['bandwidth', 'latency', 'balance']

model_debug_mode_list = [
    "dump_calibration_data",
]

run_on_list = ["BPU", "CPU"]
color_map = {
    'nv12': 'YUV_BT601_FULL_RANGE',
    'yuv420sp_bt601_video': 'YUV_BT601_VIDEO_RANGE',
}

node_type_mapping = {
    "qnt.quantize": "Quantize",
    "qnt.dequantize": "Dequantize",
    "hbir.transpose": "Transpose",
    "hbir.cast_type": "Cast",
    "hbir.reshape": "Reshape",
    "hbir.softmax": "Softmax",
    "hbtl.call::native::Cast": "Cast",
    "hbtl.call::native::Transpose": "Transpose",
    "hbtl.call::native::Reshape": "Reshape",
    "hbtl.call::native::Softmax": "Softmax",
    "hbtl.call::quant::qcast": "Quantize",
    "hbtl.call::quant::dcast": "Dequantize",
}

image_read_mode_list = ["skimage", "opencv", "PIL"]

balance_factor_mapping = {"bandwidth": 0, "latency": 100}
# temp! it is used to add undefined calibration params
calibration_parameters = [
    "cal_data_dir", "cal_data_type", "preprocess_on", "calibration_type",
    "max_percentile", "per_channel", "run_on_cpu", "run_on_bpu",
    "optimization", "quant_config"
]

preprocess_mode_map = {
    'nv12_full_rgb': 'yuvbt601full2rgb',
    'nv12_full_bgr': 'yuvbt601full2bgr',
    'nv12_video_rgb': 'yuvbt601video2rgb',
    'nv12_video_bgr': 'yuvbt601video2bgr',
    'yuv444_rgb': 'yuvbt601full2rgb',
    'yuv444_bgr': 'yuvbt601full2bgr',
    'rgb_bgr': 'rgb2bgr',
    'bgr_rgb': 'bgr2rgb',
    'nv12_full_yuv444': 'skip',
    'nv12_video_yuv444': 'skip',
}
