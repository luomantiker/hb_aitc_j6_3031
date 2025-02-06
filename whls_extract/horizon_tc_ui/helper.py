# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

# yapf: disable
from horizon_tc_ui.data.transformer import (AddTransformer,
                                            BGR2NV12Transformer,
                                            BGR2RGBTransformer,
                                            BGR2YUV444Transformer,
                                            BGR2YUVBT601VIDEOTransformer,
                                            HWC2CHWTransformer,
                                            NV12ToYUV444Transformer,
                                            ResizeTransformer,
                                            RGB2BGRTransformer,
                                            RGB2GRAY_128Transformer,
                                            RGB2NV12Transformer,
                                            RGB2YUV444Transformer,
                                            RGB2YUVBT601VIDEOTransformer,
                                            ScaleTransformer)

# yapf: enable


class ModelProtoBase:
    def __init__(self):
        pass

    def get_input_names(self):
        raise NotImplementedError

    def get_input_dims(self, input_name):
        return []


def get_raw_transformer(input_type_rt, input_type_train, input_layout_train,
                        image_height, image_width):
    data_format = input_layout_train[1:]
    # yuv444->yuv444/nv12, gray->gray
    if input_type_train.startswith(("yuv444", "gray")):
        return [AddTransformer(-128)]
    # bgr->bgr, rgb->rgb
    if input_type_train == input_type_rt:
        return [AddTransformer(-128)]

    # key mean of mapping: (input_type_train, input_type_rt)
    transformer_mapping_table = {
        ('rgb', "bgr"): [RGB2BGRTransformer(data_format=data_format)],
        ('rgb', 'nv12'): [
            RGB2NV12Transformer(data_format=data_format),
            NV12ToYUV444Transformer((image_height, image_width),
                                    yuv444_output_layout=data_format),
        ],
        ('rgb', 'yuv444'): [RGB2YUV444Transformer(data_format=data_format)],
        ('rgb', 'yuv420sp_bt601_video'): [
            RGB2YUVBT601VIDEOTransformer(data_format=data_format)
        ],
        ('bgr', 'rgb'): [BGR2RGBTransformer(data_format=data_format)],
        ('bgr', 'nv12'): [
            BGR2NV12Transformer(data_format=data_format),
            NV12ToYUV444Transformer((image_height, image_width),
                                    yuv444_output_layout=data_format)
        ],
        ('bgr', 'yuv444'): [BGR2YUV444Transformer(data_format=data_format)],
        ('bgr', 'yuv420sp_bt601_video'): [
            BGR2YUVBT601VIDEOTransformer(data_format=data_format)
        ]
    }

    transformers = transformer_mapping_table[(input_type_train, input_type_rt)]
    if input_type_rt != "yuv420sp_bt601_video":
        transformers.append(AddTransformer(-128))
    return transformers


def get_default_transformer(input_type_rt, input_type_train,
                            input_layout_train, image_height, image_width):
    transformers = [
        ResizeTransformer((image_height, image_width)),
        ScaleTransformer(255),
    ]

    # 如果模型输入是NCHW, 则需要把读进来的图片(NHWC)转换一下layout
    if input_layout_train != "NHWC":
        transformers += HWC2CHWTransformer(),

    layout = input_layout_train[1:]
    trans_dict = {
        'rgb': [AddTransformer(-128)],
        'rgbp': [AddTransformer(-128)],
        'bgr': [RGB2BGRTransformer(data_format=layout),
                AddTransformer(-128)],
        'bgrp': [RGB2BGRTransformer(data_format=layout),
                 AddTransformer(-128)],
        'nv12': [
            RGB2NV12Transformer(data_format=layout),
            NV12ToYUV444Transformer((image_height, image_width),
                                    yuv444_output_layout=layout),
            AddTransformer(-128)
        ],
        'yuv444': [
            RGB2YUV444Transformer(data_format=layout),
            AddTransformer(-128)
        ],
        # 'yuv444': [RGB2YUV444_128Transformer(data_format=layout)],
        'yuv444_128': [
            RGB2YUV444Transformer(data_format=layout),
            AddTransformer(-128)
        ],
        'gray': [RGB2GRAY_128Transformer(data_format=layout)],
        'yuv420sp_bt601_video': [
            RGB2YUVBT601VIDEOTransformer(data_format=layout)
        ],
    }
    transformers += trans_dict[input_type_rt]
    return transformers
