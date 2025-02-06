# Copyright (c) Horizon Robotics. All rights reserved.
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargconvnet_module import VargConvNetBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class VargConvNet(nn.Module):
    """
    A module of vargconvnet.

    Args:
        num_classes: Num classes of output layer.
        bn_kwargs: Dict for BN layer.
        channels_list: List for output channels
        repeats: Depth of each stage.
        group_list: Group of each stage.
        factor_list: Factor for each stage.
        out_channels: Output channels.
        bias: Whether to use bias in module.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        input_channels: Input channels of first conv.
        deep_stem: Whether use deep stem.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        channels_list: list,
        repeats: list,
        group_list: int,
        factor_list: int,
        out_channels: int = 1024,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        deep_stem: bool = True,
    ):
        super(VargConvNet, self).__init__()
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.deep_stem = deep_stem

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

        self.in_channels = channels_list[0]
        if deep_stem is True:
            self.mod1 = nn.Sequential(
                ConvModule2d(
                    in_channels=3,
                    out_channels=channels_list[0] // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(
                        channels_list[0] // 2, **self.bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                ConvModule2d(
                    in_channels=channels_list[0] // 2,
                    out_channels=channels_list[0] // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(
                        channels_list[0] // 2, **self.bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                ConvModule2d(
                    in_channels=channels_list[0] // 2,
                    out_channels=channels_list[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(
                        channels_list[0], **self.bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
            )

        else:
            self.mod1 = ConvModule2d(
                input_channels,
                channels_list[0],
                3,
                stride=2,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(channels_list[0], **bn_kwargs),
            )

        self.mod2 = self._make_stage(
            channels_list[0],
            channels_list[1],
            group_list[0],
            factor_list[0],
            repeats[0],
            False,
        )
        self.mod3 = self._make_stage(
            channels_list[1],
            channels_list[2],
            group_list[1],
            factor_list[1],
            repeats[1],
            False,
        )
        self.mod4 = self._make_stage(
            channels_list[2],
            channels_list[3],
            group_list[2],
            factor_list[2],
            repeats[2],
            False,
        )
        self.mod5 = self._make_stage(
            channels_list[3],
            channels_list[4],
            group_list[3],
            factor_list[3],
            repeats[3],
        )

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    channels_list[-1],
                    out_channels,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
                nn.AvgPool2d(7),
                ConvModule2d(
                    out_channels,
                    num_classes,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(
        self, in_channels, channels, groups, factor, repeats, invse=False
    ):
        layers = []
        layers.append(
            VargConvNetBlock(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=5,
                padding=2,
                stride=2,
                bias=self.bias,
                factor=factor,
                groups=groups,
                bn_kwargs=self.bn_kwargs,
            )
        )

        for _ in range(1, repeats):
            layers.append(
                VargConvNetBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    bias=self.bias,
                    factor=1 / factor if invse is True else factor,
                    groups=groups,
                    bn_kwargs=self.bn_kwargs,
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
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
