# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import horizon_plugin_pytorch as hpp
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["StereoNetHeadPlus"]


class DeConvResModule(nn.Module):
    """
    A basic module for deconv shortcut.

    Args:
        in_channels: The channels of inputs.
        out_channels:  The channels of outputs.
        bn_kwargs: Dict for BN layer.
        kernel: The kernel_size of deconv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: Dict = None,
        kernel: int = 4,
    ):
        super(DeConvResModule, self).__init__()

        self.conv1 = ConvTransposeModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=2,
            padding=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        )
        self.func_op = hpp.nn.quantized.FloatFunctional()

    def forward(self, x: Tensor, rem: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.conv1(x)
        x = self.func_op.add(x, rem)
        x = self.conv2(x)
        return x


class UnfoldConv(nn.Module):
    """
    A unfold module using conv.

    Args:
        in_channels: The channels of inputs.
        kernel_size: The kernel_size of unfold.
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 2):
        super(UnfoldConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.kernel_size ** 2,
            kernel_size=self.kernel_size,
            stride=1,
            bias=False,
        )
        self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of head module."""

        weight_new = torch.zeros(
            self.conv.weight.size(), dtype=self.conv.weight.dtype
        )
        for i in range(self.kernel_size ** 2):
            wx = i % self.kernel_size
            wy = i // self.kernel_size

            if wx < self.kernel_size / 2:
                if wy < self.kernel_size / 2:
                    weight_new[i, :, 0, 0] = 1
                else:
                    weight_new[i, :, 1, 0] = 1
            else:
                if wy < self.kernel_size / 2:
                    weight_new[i, :, 0, 1] = 1
                else:
                    weight_new[i, :, 1, 1] = 1

        self.conv.weight = torch.nn.Parameter(weight_new, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.pad(x)
        x = self.conv(x)
        return x

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.pad.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            activation_calibration_observer="mix",
        )
        self.conv.qconfig = qconfig_manager.get_qconfig(
            activation_fake_quant=None,
            activation_qat_observer=None,
            activation_qat_qkwargs=None,
            activation_calibration_observer=None,
            activation_calibration_qkwargs=None,
            weight_qat_qkwargs={
                "qscheme": torch.per_channel_symmetric,
                "ch_axis": 0,
                "averaging_constant": 1,
            },
            weight_calibration_qkwargs={
                "qscheme": torch.per_channel_symmetric,
                "ch_axis": 0,
                "averaging_constant": 1,
            },
        )

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.conv.weight_fake_quant.disable_observer()
        self.conv.weight_fake_quant.set_qparams(
            torch.ones(
                self.conv.weight.shape[0], device=self.conv.weight.device
            )
        )


class AdaptiveAggregationModule(nn.Module):
    """
    Adaptive aggregation module for optimizing disparity.

    Args:
        num_scales: The num of cost volume.
        num_output_branches:  The num branch for outputs.
        max_disp: The max value of disparity.
        num_blocks: The num of block.
    """

    def __init__(
        self,
        num_scales: int,
        num_output_branches: int,
        max_disp: int,
        num_blocks: int = 1,
    ):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for _ in range(num_blocks):
                # if simple_bottleneck:
                branch.append(
                    BasicResBlock(num_candidates, num_candidates, bn_kwargs={})
                )
            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        ),
                    )
                elif i > j:
                    layers = nn.ModuleList()
                    for _ in range(i - j - 1):
                        layers.append(
                            nn.Sequential(
                                ConvModule2d(
                                    in_channels=max_disp // (2 ** j),
                                    out_channels=max_disp // (2 ** j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    norm_layer=nn.BatchNorm2d(
                                        max_disp // (2 ** j)
                                    ),
                                    act_layer=nn.ReLU(inplace=True),
                                )
                            )
                        )

                    layers.append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fuse_add = nn.ModuleList()
        for _ in range(len(self.fuse_layers) * len(self.branches)):
            self.fuse_add.append(hpp.nn.quantized.FloatFunctional())

    @fx_wrap()
    def update_idx(self, idx: int) -> int:
        """Update the idx."""

        return idx + 1

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            x: The inputs pyramid costvolume.

        Returns:
            x_fused: The fused pyramid costvolume.
        """

        assert len(self.branches) == len(x)

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])

        x_fused = []
        idx = 0
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    x_fused[i] = self.interpolate_exchange(
                        x_fused, exchange, i, idx
                    )
                    idx = self.update_idx(idx)
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

    @fx_wrap()
    def interpolate_exchange(
        self, x_fused: Tensor, exchange: Tensor, i: int, idx: int
    ) -> Tensor:
        """Unsample costvolume and fuse."""

        if exchange.size()[2:] != x_fused[i].size()[2:]:
            exchange = F.interpolate(
                exchange,
                size=x_fused[i].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.fuse_add[idx].add(exchange, x_fused[i])


@OBJECT_REGISTRY.register
class StereoNetHeadPlus(nn.Module):
    """
    An advanced head for StereoNet.

    Args:
        maxdisp: The max value of disparity.
        refine_levels:  Number of refinement layers.
        bn_kwargs: Dict for BN layer.
        max_stride: The max stride for model input.
        num_costvolume: The number of pyramid costvolume.
        num_fusion: The number of fusion module.
        hidden_dim: The hidden dim.
        in_channels: The channels of input features.
    """

    def __init__(
        self,
        maxdisp: int = 192,
        refine_levels: int = 4,
        bn_kwargs: Dict = None,
        max_stride: int = 32,
        num_costvolume: int = 3,
        num_fusion: int = 6,
        hidden_dim: int = 16,
        in_channels: List[int] = (32, 32, 16, 16, 16),
    ):
        super(StereoNetHeadPlus, self).__init__()
        self.maxdisp = maxdisp
        self.refine_levels = refine_levels
        self.num_costvolume = num_costvolume
        self.D = self.maxdisp // max_stride
        self.num_fusion = num_fusion
        self.gc_pad = nn.ModuleList()
        self.gc_mean = nn.ModuleList()
        self.gc_mul = nn.ModuleList()
        self.hidden_dim = hidden_dim
        for k in range(num_costvolume):
            scale_tmp = pow(2, k)
            for i in range(self.D * scale_tmp):
                self.gc_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
                self.gc_mean.append(hpp.nn.quantized.FloatFunctional())
                self.gc_mul.append(hpp.nn.quantized.FloatFunctional())

        self.gc_cat_final = nn.ModuleList()

        for _ in range(self.refine_levels):
            self.gc_cat_final.append(hpp.nn.quantized.FloatFunctional())

        self.softmax2 = nn.Softmax(dim=1)
        self.softmax2.min_sub_out = -12.0

        low_disp_max = self.D * pow(2, self.num_costvolume - 1)

        self.fusions = nn.ModuleList()
        for i in range(self.num_fusion):
            num_out_branches = 1 if i == self.num_fusion - 1 else 3
            self.fusions.append(
                AdaptiveAggregationModule(
                    self.num_costvolume, num_out_branches, low_disp_max
                )
            )

        self.final_conv = ConvModule2d(
            low_disp_max,
            low_disp_max,
            kernel_size=1,
        )
        self.disp_mul_op = hpp.nn.quantized.FloatFunctional()
        self.disp_sum_op = hpp.nn.quantized.FloatFunctional()
        self.quant_dispvalue = QuantStub()
        self.softmax = nn.Softmax(dim=1)
        self.softmax.min_sub_out = -12.0
        self.disp_values = nn.Parameter(
            torch.range(0, low_disp_max - 1).view(1, low_disp_max, 1, 1)
            / low_disp_max,
            requires_grad=False,
        )
        self.spx_8 = nn.Sequential(
            ConvModule2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
            ),
        )

        self.spx_4 = DeConvResModule(
            self.hidden_dim, self.hidden_dim, bn_kwargs
        )
        self.spx_2 = DeConvResModule(
            self.hidden_dim, self.hidden_dim, bn_kwargs
        )
        self.spx = ConvTransposeModule2d(
            self.hidden_dim,
            4,
            kernel_size=4,
            stride=2,
            padding=1,
            norm_layer=nn.BatchNorm2d(4, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.spx_conv3x3 = ConvModule2d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.mod1 = ConvModule2d(
            in_channels=in_channels[0],
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        )
        self.mod2 = ConvModule2d(
            in_channels=in_channels[1],
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(self.hidden_dim, **bn_kwargs),
        )
        self.unfold = UnfoldConv()
        self.dequant = DeQuantStub()

    @fx_wrap()
    def get_l_img(self, img: Tensor, B: int) -> Tensor:
        """Get left featuremaps.

        Args:
            img: The inputs featuremaps.
            B: Batchsize.

        """

        return img[: B // 2]

    @fx_wrap()
    def dis_mul(self, x: Tensor) -> Tensor:
        """Mul weight to the disparity."""

        disp_values = self.quant_dispvalue(self.disp_values)
        return self.disp_mul_op.mul(x, disp_values)

    @fx_wrap()
    def dis_sum(self, x: Tensor) -> Tensor:
        """Get the low disparity."""
        return self.disp_sum_op.sum(x, dim=1, keepdim=True)

    @fx_wrap()
    def build_aanet_volume(self, refimg_fea, maxdisp, offset, idx):
        """
        Build the cost volume using the same approach as AANet.

        Args:
            refimg_fea: Featuremaps.
            maxdisp: Maximum disparity value.
            offset: The offset of gc_mul and gc_mean floatFunctional.
            idx: The idx of cat floatFunctional.

        Returns:
            volume: Costvolume.
        """

        B, C, H, W = refimg_fea.shape
        num_sample = B // 2
        tmp_volume = []
        for i in range(maxdisp):
            if i > 0:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, i:],
                    refimg_fea[num_sample:, :, :, :-i],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1, keepdim=True)
                tmp_volume.append(self.gc_pad[i + offset](tmp))
            else:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, :],
                    refimg_fea[num_sample:, :, :, :],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1, keepdim=True)
                tmp_volume.append(tmp)

        volume = (
            self.gc_cat_final[idx]
            .cat(tmp_volume, dim=1)
            .view(num_sample, maxdisp, H, W)
        )
        return volume

    @fx_wrap()
    def get_offset(self, offset: int, idx: int) -> int:
        """Get offset of floatFunctional."""
        return offset + self.D * (2 ** idx)

    def forward(self, features_inputs: List[Tensor]) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            features: The inputs featuremaps.

        Returns:
            pred0: The low disparity.
            pred0_unfold: The low disparity after unfold.
            spx_pred: The weight of each point.
        """

        features_inputs[0] = self.mod1(features_inputs[0])
        features_inputs[1] = self.mod2(features_inputs[1])

        # Build cost volume as AANet
        features = features_inputs[-3:][::-1]
        aanet_volumes = []
        offset = 0
        for i in range(len(features)):
            aanet_volume = self.build_aanet_volume(
                features[i], self.D * (2 ** i), offset, i
            )
            offset = self.get_offset(offset, i)
            aanet_volumes.append(aanet_volume)

        # Fusion costvolume as AANet
        aanet_volumes = aanet_volumes[::-1]
        for i in range(len(self.fusions)):
            fusion = self.fusions[i]
            aanet_volumes = fusion(aanet_volumes)

        cost0 = self.final_conv(aanet_volumes[0])

        # Obtain low disparity and unfold it.
        pred0 = self.softmax(cost0)
        pred0 = self.dis_mul(pred0)
        pred0 = self.dis_sum(pred0)
        pred0_unfold = self.unfold(pred0)

        # Obtain weight of each point as Coex.

        B, _, _, _ = features_inputs[0].shape
        xspx = self.spx_8(self.get_l_img(features_inputs[2], B))

        feature1_l = self.get_l_img(features_inputs[1], B)

        xspx = self.spx_4(xspx, feature1_l)

        feature0_l = self.get_l_img(features_inputs[0], B)

        xspx = self.spx_2(xspx, feature0_l)
        spx_pred = self.spx(xspx)
        spx_pred = self.spx_conv3x3(spx_pred)
        spx_pred = self.softmax2(spx_pred)

        return (
            self.dequant(pred0),
            self.dequant(pred0_unfold),
            self.dequant(spx_pred),
        )

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        module_list = [
            self.spx_conv3x3,
            self.softmax2,
            self.disp_mul_op,
            self.disp_sum_op,
            self.final_conv,
            self.quant_dispvalue,
            self.softmax,
        ]
        for m in module_list:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        self.unfold.set_qconfig()

        for m in self.gc_mul:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

        for m in self.gc_mean:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        for m in self.gc_pad:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
