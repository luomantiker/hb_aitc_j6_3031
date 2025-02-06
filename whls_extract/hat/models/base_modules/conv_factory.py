from typing import Dict, Optional, Tuple, Type, Union

import torch.nn as nn

from .basic_vargnet_module import BasicVarGBlock
from .conv_module import ConvModule2d
from .separable_conv_module import SeparableGroupConvModule2d


def get_conv_module(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    bn_kwargs: Optional[Dict] = None,
    group_base: int = 8,
    conv_method: str = "conv",
    dw_activation: Optional[Type[nn.Module]] = None,
    pw_activation: Optional[Type[nn.Module]] = None,
    bias: bool = True,
    act_kwargs: Optional[Dict] = None,
    dw_norm_method: Optional[Type[nn.Module]] = None,
    pw_norm_method: Optional[Type[nn.Module]] = None,
):
    assert conv_method in [
        "varg_conv",
        "sep_conv",
        "conv",
    ], "Unsupported 'conv_method' {}".format(conv_method)
    if bn_kwargs is None:
        bn_kwargs = {}
    if act_kwargs is None:
        act_kwargs = {"inplace": True}
    if conv_method == "varg_conv":
        return BasicVarGBlock(
            in_channels=in_channels,
            mid_channels=out_channels,
            out_channels=out_channels,
            stride=stride,
            bn_kwargs=bn_kwargs,
            kernel_size=kernel_size,
            padding=padding,
            factor=1,
            group_base=group_base,
            dw_with_relu=True if dw_activation else False,
            pw_with_relu=True if pw_activation else False,
        )
    elif conv_method == "sep_conv":
        if "num_features" in bn_kwargs:
            bn_kwargs.pop("num_features")
        return SeparableGroupConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            bias=bias,
            dw_act_layer=dw_activation(**act_kwargs)
            if dw_activation
            else None,
            pw_act_layer=pw_activation(**act_kwargs)
            if pw_activation
            else None,
            dw_norm_layer=dw_norm_method(in_channels, **bn_kwargs)
            if dw_norm_method
            else None,
            pw_norm_layer=pw_norm_method(out_channels, **bn_kwargs)
            if pw_norm_method
            else None,
        )
    else:
        return ConvModule2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            norm_layer=dw_norm_method(out_channels, **bn_kwargs)
            if dw_norm_method
            else None,
            act_layer=dw_activation(**act_kwargs) if dw_activation else None,
        )
