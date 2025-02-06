# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

# yapf: disable
from horizon_tc_ui.data.transformer import (BGR2NV12Transformer,
                                            BGR2RGBTransformer,
                                            CenterCropTransformer,
                                            PaddedCenterCropTransformer,
                                            PadResizeTransformer,
                                            PadTransformer,
                                            PILCenterCropTransformer,
                                            PILResizeTransformer,
                                            ResizeTransformer,
                                            RGB2BGRTransformer,
                                            RGB2NV12Transformer,
                                            ScaleTransformer,
                                            ShortSideResizeTransformer,
                                            WarpAffineTransformer)

# yapf: enable


def mobilenetv1_data_transformer():
    transformers = [
        ShortSideResizeTransformer(short_size=256),
        CenterCropTransformer(crop_size=224),
        RGB2BGRTransformer(data_format="HWC"),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        BGR2NV12Transformer(data_format="HWC"),
    ]
    return transformers, (224, 224)


def mobilenetv2_data_transformer():
    transformers = [
        ShortSideResizeTransformer(short_size=256),
        CenterCropTransformer(crop_size=224),
        RGB2BGRTransformer(data_format="HWC"),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        BGR2NV12Transformer(data_format="HWC"),
    ]
    return transformers, (224, 224)


def googlenet_data_transformer():
    transformers = [
        ShortSideResizeTransformer(short_size=256),
        CenterCropTransformer(crop_size=224),
        BGR2NV12Transformer(data_format="HWC"),
    ]
    return transformers, (224, 224)


def resnet50_data_transformer():
    transformers = [
        PILResizeTransformer(size=256),
        PILCenterCropTransformer(size=224),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (224, 224)


def resnet18_data_transformer():
    transformers = [
        PILResizeTransformer(size=256),
        PILCenterCropTransformer(size=224),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (224, 224)


def efficientnet_lite0_data_transformer():

    transformers = [
        PaddedCenterCropTransformer(image_size=224, crop_pad=32),
        ResizeTransformer(target_size=(224, 224), mode='skimage', method=3),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC"),
    ]
    return transformers, (224, 224)


def efficientnet_lite1_data_transformer():
    image_size = 240
    transformers = [
        PaddedCenterCropTransformer(image_size=image_size, crop_pad=32),
        ResizeTransformer(target_size=(image_size, image_size),
                          mode='skimage',
                          method=3),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (240, 240)


def efficientnet_lite2_data_transformer():
    image_size = 260
    transformers = [
        PaddedCenterCropTransformer(image_size=image_size, crop_pad=32),
        ResizeTransformer(target_size=(image_size, image_size),
                          mode='skimage',
                          method=3),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (260, 260)


def efficientnet_lite3_data_transformer():
    image_size = 280
    transformers = [
        PaddedCenterCropTransformer(image_size=image_size, crop_pad=32),
        ResizeTransformer(target_size=(image_size, image_size),
                          mode='skimage',
                          method=3),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (280, 280)


def efficientnet_lite4_data_transformer():
    image_size = 300
    transformers = [
        PaddedCenterCropTransformer(image_size=image_size, crop_pad=32),
        ResizeTransformer(target_size=(image_size, image_size),
                          mode='skimage',
                          method=3),
        ScaleTransformer(scale_value=255, data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, (300, 300)


def vargconvnet_data_transformer():
    image_size = 224
    transformers = [
        ResizeTransformer(target_size=(256, 256)),
        CenterCropTransformer(crop_size=image_size),
        RGB2NV12Transformer(data_format="HWC"),
    ]
    return transformers, (224, 224)


def efficientnasnet_m_data_transformer():
    input_shape = (300, 300)
    transformers = [
        BGR2RGBTransformer(data_format="HWC"),
        ShortSideResizeTransformer(short_size=256,
                                   data_type="uint8",
                                   interpolation="INTER_CUBIC"),
        CenterCropTransformer(crop_size=224, data_type="uint8"),
        ResizeTransformer(target_size=(300, 300),
                          data_type="uint8",
                          mode='opencv',
                          method=1,
                          interpolation="INTER_CUBIC"),
        RGB2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def efficientnasnet_s_data_transformer():
    input_shape = (280, 280)
    transformers = [
        BGR2RGBTransformer(data_format="HWC"),
        ShortSideResizeTransformer(short_size=256,
                                   data_type="uint8",
                                   interpolation="INTER_CUBIC"),
        CenterCropTransformer(crop_size=224, data_type="uint8"),
        ResizeTransformer(target_size=(280, 280),
                          data_type="uint8",
                          mode='opencv',
                          method=1,
                          interpolation="INTER_CUBIC"),
        RGB2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def yolov2_darknet19_data_transformer():
    input_shape = (608, 608)
    transformers = [
        PadResizeTransformer(target_size=input_shape),
        BGR2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def yolov3_darknet53_data_transformer():
    input_shape = (416, 416)
    transformers = [
        PadResizeTransformer(target_size=input_shape),
        BGR2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def yolov5x_data_transformer():
    input_shape = (672, 672)
    transformers = [
        PadResizeTransformer(target_size=input_shape),
        BGR2RGBTransformer(data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def ssd_mobilenetv1_data_transformer():
    input_shape = (300, 300)
    transformers = [
        ResizeTransformer(target_size=input_shape, mode='opencv', method=1),
        BGR2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def efficientdetd0_data_transformer():
    input_shape = (512, 512)
    transformers = [
        PadResizeTransformer(target_size=input_shape,
                             pad_value=0.,
                             pad_position='bottom_right'),
        BGR2RGBTransformer(data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def centernet_resnet101_data_transformer():
    input_shape = (512, 512)
    transformers = [
        BGR2RGBTransformer(data_format="HWC"),
        WarpAffineTransformer(input_shape, 1.0),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers, input_shape


def fcos_efficientnetb0_data_transformer():
    input_shape = (512, 512)
    transformers = [
        PadResizeTransformer((512, 512),
                             pad_position='bottom_right',
                             pad_value=0),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv")
    ]
    return transformers, input_shape


def yolov4_data_transformer():
    input_shape = (512, 512)
    transformers = [
        PadResizeTransformer(target_size=input_shape,
                             pad_position='bottom_right',
                             pad_value=0),
        BGR2NV12Transformer(data_format="HWC"),
    ]
    return transformers, input_shape


def yolov3_vargdarknet_data_transformer():
    input_shape = (416, 416)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def fcos_resnet50_data_transformer():
    input_shape = (1024, 1024)
    transformers = [
        PadTransformer(size_divisor=1024, target_size=1024),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv")
    ]
    return transformers, input_shape


def fcos_resnext101_data_transformer():
    input_shape = (1024, 1024)
    transformers = [
        PadResizeTransformer(input_shape, pad_position='bottom_right'),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv")
    ]
    return transformers, input_shape


def unet_mobilenet_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2RGBTransformer(data_format="HWC"),
        RGB2NV12Transformer(data_format="HWC", cvt_mode="opencv")
    ]
    return transformers, input_shape


def deeplabv3plus_efficientnetb0_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def fastscnn_efficientnetb0_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def deeplabv3plus_dilation1248_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer((1024, 2048)),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def deeplabv3plus_efficientnetm1_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape


def deeplabv3plus_efficientnetm2_data_transformer():
    input_shape = (1024, 2048)
    transformers = [
        ResizeTransformer(input_shape),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
    ]
    return transformers, input_shape
