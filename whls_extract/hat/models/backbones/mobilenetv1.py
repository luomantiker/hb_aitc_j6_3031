# Copyright (c) Horizon Robotics. All rights reserved.

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["MobileNetV1"]


@OBJECT_REGISTRY.register
class MobileNetV1(nn.Module):
    """
    A module of mobilenetv1.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        alpha (float): Alpha for mobilenetv1.
        bias (bool): Whether to use bias in module.
        dw_with_relu (bool): Whether to use relu in dw conv.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        alpha: float = 1.0,
        bias: bool = True,
        dw_with_relu: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output

        in_chls = [[32], [64, 128], [128, 256], [256] + [512] * 5, [512, 1024]]
        out_chls = [[64], [128, 128], [256, 256], [512] * 6, [1024, 1024]]

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

        self.mod1 = self._make_stage(in_chls[0], out_chls[0], True)
        self.mod2 = self._make_stage(in_chls[1], out_chls[1])
        self.mod3 = self._make_stage(in_chls[2], out_chls[2])
        self.mod4 = self._make_stage(in_chls[3], out_chls[3])
        self.mod5 = self._make_stage(in_chls[4], out_chls[4])

        if self.include_top:
            self.output = nn.Sequential(
                nn.AvgPool2d(7),
                ConvModule2d(
                    int(out_chls[4][-1] * alpha),
                    num_classes,
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, in_chls, out_chls, first_layer=False):
        layers = []
        in_chls = [int(chl * self.alpha) for chl in in_chls]
        out_chls = [int(chl * self.alpha) for chl in out_chls]
        for i, in_chl, out_chl in zip(range(len(in_chls)), in_chls, out_chls):
            stride = 2 if i == 0 else 1
            if first_layer:
                layers.append(
                    ConvModule2d(
                        3,
                        in_chls[0],
                        3,
                        stride,
                        1,
                        bias=self.bias,
                        norm_layer=nn.BatchNorm2d(
                            in_chls[0], **self.bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                stride = 1
            layers.append(
                SeparableConvModule2d(
                    in_chl,
                    out_chl,
                    3,
                    stride,
                    1,
                    bias=self.bias,
                    dw_norm_layer=nn.BatchNorm2d(
                        in_chl,
                        **self.bn_kwargs,
                    ),
                    dw_act_layer=nn.ReLU(inplace=True),
                    pw_norm_layer=nn.BatchNorm2d(
                        out_chl,
                        **self.bn_kwargs,
                    ),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
