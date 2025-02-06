# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional

from .conv_module import ConvModule2d

__all__ = [
    "RepVGGDW",
    "RepViTBlock",
    "ConvBN",
]


def make_divisible(
    v: int,
    divisor: int = 8,
    min_value: Optional[int] = None,
    round_limit: float = 0.9,
) -> int:
    """Make a number divisible by a given divisor.

    Args:
        v (int): The number to be made divisible.
        divisor (int, optional): The divisor. Defaults to 8.
        min_value (int, optional): The minimum value for the result.
            Defaults to None.
        round_limit (float, optional): The limit for rounding down.
            Defaults to 0.9.

    Returns:
        int: The number made divisible by the divisor.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation module.

    Args:
        in_channels (int): The number of input channels.
        num_squeezed_channels (int): The number of squeezed channels.
        out_channels (int): The number of output channels.
        groups (int, optional): The number of groups for grouped convolution.
            Defaults to 1.
        act_layer (nn.Module, optional): The activation function.
            Defaults to nn.ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        num_squeezed_channels: int,
        out_channels: int,
        groups: int = 1,
        act_layer: nn.Module = nn.ReLU,
    ):
        super(SqueezeExcite, self).__init__()

        assert (
            in_channels % groups == 0 and num_squeezed_channels % groups == 0
        )
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                groups=groups,
                bias=True,
                norm_layer=None,
                act_layer=act_layer,
            ),
            ConvModule2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                bias=True,
                norm_layer=None,
                act_layer=nn.Sigmoid(),
            ),
        )
        self.float_func = FloatFunctional()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.float_func.mul(x, inputs)

        return x

    def fuse_model(self):
        """Fuse the model."""
        getattr(self.conv, "1").fuse_model()


class ConvBN(nn.Sequential):
    """ConvBN represents a combination of a conv layer and a bn layer.

    Args:
        in_chn (int): The number of input channels.
        out_chn (int): The number of output channels.
        ks (int, optional): The kernel size of the conv layer. Defaults to 1.
        stride (int, optional): The stride of the conv layer. Defaults to 1.
        pad (int, optional): The padding of the conv layer. Defaults to 0.
        dilation (int, optional): The dilation of the conv layer.
            Defaults to 1.
        groups (int, optional): The number of groups for the conv layer.
            Defaults to 1.
        bn_weight_init (int, optional): The initial value for the bn
            layer's weight. Defaults to 1.
    """

    def __init__(
        self,
        in_chn: int,
        out_chn: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: int = 1,
    ):
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_chn, out_chn, ks, stride, pad, dilation, groups, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_chn))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """Fuse the conv layer and bn layer into a single conv layer.

        Returns:
            torch.nn.Conv2d: The fused conv layer.
        """
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = (
            bn.bias
            - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        )
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """A residual module that can be fused.

    Args:
        m (nn.Module): The block inside the residual module.
        drop (float, optional): The dropout rate. Defaults to 0.
    """

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.add_func = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.add_func.add(x, self.m(x))

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fuse the conv layers into a single conv layer.

        Returns:
            nn.Module: The fused conv layer.
        """
        if isinstance(self.m, ConvBN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(nn.Module):
    """A RepVGG module with DW conv.

    For more details see https://arxiv.org/abs/2101.03697.

    Args:
        chn (int): The number of input channels.
    """

    def __init__(self, chn: int) -> None:
        super().__init__()
        self.conv = ConvBN(chn, chn, 3, 1, 1, groups=chn)
        self.conv1 = nn.Conv2d(chn, chn, 1, 1, 0, groups=chn)
        self.dim = chn
        self.bn = nn.BatchNorm2d(chn)
        self.add_func = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(
            self.add_func.add(
                self.add_func.add(self.conv(x), self.conv1(x)), x
            )
        )

    @torch.no_grad()
    def fuse(self) -> nn.Module:
        """Fuse the conv layers and identity layer into a single conv layer.

        Returns:
            nn.Module: The fused conv layer.
        """
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])

        identity = F.pad(
            torch.ones(
                conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device
            ),
            [1, 1, 1, 1],
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = (
            bn.bias
            + (conv.bias - bn.running_mean)
            * bn.weight
            / (bn.running_var + bn.eps) ** 0.5
        )
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    """RepViTBlock represents a block in the RepViT backbone.

    Args:
        in_chn (int): The number of input channels.
        hidden_dim (int): The number of hidden channels.
        out_chn (int): The number of output channels.
        kernel_size (int): The size of the conv kernel.
        stride (int): The stride of the conv operation.
        use_se (bool): Whether to use Squeeze-and-Excitation.
        use_hs (bool): Whether to use Hard-Swish activation.
    """

    def __init__(
        self,
        in_chn: int,
        hidden_dim: int,
        out_chn: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and in_chn == out_chn
        assert hidden_dim == 2 * in_chn

        if stride == 2:
            self.token_mixer = nn.Sequential(
                ConvBN(
                    in_chn,
                    in_chn,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=in_chn,
                ),
                SqueezeExcite(
                    in_chn,
                    in_chn // 4,
                    in_chn,
                    groups=1,
                    act_layer=nn.GELU() if use_hs else nn.ReLU(inplace=True),
                )
                if use_se
                else nn.Identity(),
                ConvBN(in_chn, out_chn, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    ConvBN(out_chn, 2 * out_chn, 1, 1, 0),
                    nn.GELU() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    ConvBN(2 * out_chn, out_chn, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(in_chn),
                SqueezeExcite(
                    in_chn,
                    in_chn // 4,
                    in_chn,
                    groups=1,
                    act_layer=nn.GELU() if use_hs else nn.ReLU(inplace=True),
                )
                if use_se
                else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    ConvBN(in_chn, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.ReLU(inplace=True),
                    # pw-linear
                    ConvBN(hidden_dim, out_chn, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class BnLinear(nn.Sequential):
    """BnLinear is a class that represents a batch-normalized linear layer.

    Args:
        in_chn (int): The number of input channels.
        out_chn (int): The number of output channels.
        bias (bool, optional): Whether to include a bias term.
            Defaults to True.
        std (float, optional): The std for weight initialization.
            Defaults to 0.02.
    """

    def __init__(
        self, in_chn: int, out_chn: int, bias: bool = True, std: float = 0.02
    ):

        super().__init__()
        self.add_module("bn", nn.BatchNorm1d(in_chn))
        self.add_module("ll", nn.Linear(in_chn, out_chn, bias=bias))
        nn.init.trunc_normal_(self.ll.weight, std=std)
        if bias:
            nn.init.constant_(self.ll.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        """Fuse the bn and linear layers into a single linear layer.

        Returns:
            nn.Linear: The fused linear layer.
        """
        bn, ll = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = (
            bn.bias
            - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        )
        w = ll.weight * w[None, :]
        if ll.bias is None:
            b = b @ ll.weight.T
        else:
            b = (ll.weight @ b[:, None]).view(-1) + ll.bias
        m = nn.Linear(w.size(1), w.size(0), device=ll.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Classifier(nn.Module):
    """Classifier is a class that represents a classifier module.

    Args:
        dim (int): The dimension of the input features.
        num_classes (int): The number of output classes.
        distillation (bool, optional): Whether to use distillation.
            Defaults to True.
    """

    def __init__(self, dim: int, num_classes: int, distillation: bool = True):
        super().__init__()
        self.classifier = (
            BnLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.distillation = distillation
        if distillation:
            self.classifier_dist = (
                BnLinear(dim, num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self) -> nn.Linear:
        """Fuse the bn and linear layers into a single linear layer.

        Returns:
            nn.Linear: The fused linear layer.
        """
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class HeavyClassifier(nn.Module):
    """HeavyClassfier is a class that represents a heavy classifier module.

    Using this classifier will result in a larger model and helps to improve
    the ImageNet classification accuracy.

    Args:
        dim (int): The dimension of the input features.
        num_classes (int): The number of output classes.
        hidden_dim (int, optional): The dimension of the hidden layer.
            Defaults to 1024.
        distillation (bool, optional): Whether to use distillation.
            Defaults to False.
        use_hs (bool, optional): Whether to use HS activation.
            Defaults to True.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_dim: int = 1024,
        distillation: bool = False,
        use_hs: bool = True,
    ):
        super().__init__()
        self.classifier = (
            nn.Sequential(
                ConvModule2d(
                    dim,
                    hidden_dim,
                    1,
                    bias=True,
                    norm_layer=nn.BatchNorm2d(hidden_dim),
                    act_layer=nn.GELU() if use_hs else nn.ReLU(inplace=True),
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(
                    hidden_dim,
                    num_classes,
                ),
            )
            if num_classes > 0
            else torch.nn.Identity()
        )

        self.distillation = distillation
        if distillation:
            self.classifier_dist = (
                nn.Sequential(
                    ConvModule2d(
                        dim,
                        hidden_dim,
                        1,
                        bias=True,
                        norm_layer=nn.BatchNorm2d(hidden_dim),
                        act_layer=nn.GELU()
                        if use_hs
                        else nn.ReLU(inplace=True),
                    ),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1),
                    nn.Linear(
                        hidden_dim,
                        num_classes,
                    ),
                )
                if num_classes > 0
                else torch.nn.Identity()
            )

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x
