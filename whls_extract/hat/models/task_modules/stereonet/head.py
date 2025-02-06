# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import Dict, List

import horizon_plugin_pytorch as hpp
import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch import quantization
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["StereoNetHead"]


class convbn_3d(nn.Module):
    """The basic structure of conv3d, bn3d and activation.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        pad: Padding added to both sides of the input.
        act: Activation module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pad: int,
        act: nn.Module,
    ):
        super(convbn_3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.conv(x)
        x_bn = self.bn(x)
        x_act = self.act(x_bn)
        return x_act

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        torch.quantization.fuse_modules(
            self,
            ["conv", "bn", "act"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


class EdgeAwareRefinement(nn.Module):
    """
    A Refinement module of Stereonet.

    Args:
        in_channel: Channels of featmap.
        bn_kwargs: Dict for BN layer.
        num_res: Number of res block.
        is_last: Whether is the last refinement layer.
    """

    def __init__(
        self,
        in_channel: int,
        bn_kwargs: Dict = None,
        num_res: int = 6,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.conv2d_feature = nn.Sequential(
            ConvModule2d(
                in_channels=in_channel,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(32, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        )
        self.residual_astrous_blocks = nn.ModuleList()
        self.num_res = num_res
        for _ in range(self.num_res):
            self.residual_astrous_blocks.append(
                BasicResBlock(
                    32,
                    32,
                    stride=1,
                    bias=False,
                    bn_kwargs=bn_kwargs,
                )
            )

        self.conv2d_out = ConvModule2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(1, **bn_kwargs),
            act_layer=None,
        )
        self.act = nn.ReLU(inplace=True)
        self.cat_img = hpp.nn.quantized.FloatFunctional()
        self.res_add = hpp.nn.quantized.FloatFunctional()
        self.mul_s = hpp.nn.quantized.FloatFunctional()

    def forward(
        self,
        low_disparity: Tensor,
        corresponding_rgb: Tensor,
    ) -> Tensor:
        """
        Forward pass of the module to get disparity or offsets.

        Args:
            low_disparity: Input low-resolution disparity map.
            corresponding_rgb: Corresponding left image.

        """
        low_disparity = self.mul_s.mul_scalar(low_disparity, 1.0)

        twice_disparity = F.interpolate(
            low_disparity,
            size=corresponding_rgb.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cat_disp = self.cat_img.cat(
            [twice_disparity, corresponding_rgb], dim=1
        )
        output = self.conv2d_feature(cat_disp)
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        offset = self.conv2d_out(output)

        if self.is_last:
            return offset
        else:
            new_disp = self.res_add.add(offset, twice_disparity)
            return self.act(new_disp)

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        modules = [self.conv2d_feature, self.residual_astrous_blocks]
        for m in modules:
            for m_ in m:
                if hasattr(m_, "fuse_model"):
                    m_.fuse_model()
        if self.is_last:
            torch.quantization.fuse_modules(
                self,
                ["conv2d_out.0", "conv2d_out.1"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            torch.quantization.fuse_modules(
                self,
                ["conv2d_out.0", "conv2d_out.1", "res_add", "act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        m_list = [
            self.cat_img,
            self.conv2d_out,
            self.res_add,
            self.act,
        ]

        for m in m_list:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        self.conv2d_feature.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint8},
            activation_calibration_qkwargs={
                "dtype": qint8,
            },
            activation_calibration_observer="mix",
        )
        for m in self.residual_astrous_blocks:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint8},
                activation_calibration_qkwargs={
                    "dtype": qint8,
                },
                activation_calibration_observer="mix",
            )
        if self.is_last:
            self.conv2d_out.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )


@OBJECT_REGISTRY.register
class StereoNetHead(nn.Module):
    """
    A basic head of StereoNet.

    Args:
        maxdisp: The max value of disparity.
        refine_levels:  Number of refinement layers.
        bn_kwargs: Dict for BN layer.
        num_groups: Number of group for cost volume.
    """

    def __init__(
        self,
        maxdisp: int = 192,
        refine_levels: int = 4,
        bn_kwargs: Dict = None,
        num_groups: int = 32,
    ):
        super(StereoNetHead, self).__init__()
        self.maxdisp = maxdisp
        self.refine_levels = refine_levels
        self.num_groups = num_groups

        self.filter = nn.ModuleList()
        self.filter.append(
            nn.Sequential(
                convbn_3d(
                    64,
                    32,
                    kernel_size=3,
                    stride=1,
                    pad=1,
                    act=nn.ReLU(inplace=True),
                )
            )
        )
        for _ in range(self.refine_levels):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(
                        32,
                        32,
                        kernel_size=3,
                        stride=1,
                        pad=1,
                        act=nn.ReLU(inplace=True),
                    )
                )
            )
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1
        )

        self.edge_aware_refinements = nn.ModuleList()
        for i in range(self.refine_levels):
            if i == self.refine_levels - 1:
                self.edge_aware_refinements.append(
                    EdgeAwareRefinement(4, bn_kwargs, is_last=True)
                )
            else:
                self.edge_aware_refinements.append(
                    EdgeAwareRefinement(4, bn_kwargs)
                )

        self.D = self.maxdisp // pow(2, self.refine_levels)
        self.gc_pad = nn.ModuleList()
        self.c_pad = nn.ModuleList()
        self.gc_mean = nn.ModuleList()
        self.gc_mul = nn.ModuleList()
        self.c_cat = nn.ModuleList()
        for i in range(self.D):
            self.gc_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
            self.c_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
            self.gc_mean.append(hpp.nn.quantized.FloatFunctional())
            self.gc_mul.append(hpp.nn.quantized.FloatFunctional())
            self.c_cat.append(hpp.nn.quantized.FloatFunctional())

        self.c_cat_final = hpp.nn.quantized.FloatFunctional()
        self.gc_cat_final = hpp.nn.quantized.FloatFunctional()
        self.cat_volume = hpp.nn.quantized.FloatFunctional()
        self.disp_mul_op = hpp.nn.quantized.FloatFunctional()
        self.disp_sum_op = hpp.nn.quantized.FloatFunctional()

        self.softmax = nn.Softmax(dim=1)
        self.quant_dispvalue = QuantStub()
        self.downsample = nn.ModuleList()
        for i in range(self.refine_levels - 1, -1, -1):
            self.downsample.append(
                hnn.Interpolate(
                    scale_factor=1 / pow(2, i),
                    mode="bilinear",
                    recompute_scale_factor=True,
                )
            )
        self.disp_values = nn.Parameter(
            torch.range(0, self.D - 1).view(1, self.D, 1, 1) / self.D,
            requires_grad=False,
        )
        self.dequant = DeQuantStub()
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of head module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @fx_wrap()
    def build_concat_volume(
        self,
        refimg_fea: Tensor,
        targetimg_fea: Tensor,
        maxdisp: int,
    ) -> Tensor:
        """
        Build the concat cost volume.

        Args:
            refimg_fea: Left image feature.
            targetimg_fea: Right image feature.
            maxdisp: Maximum disparity value.

        Returns:
            volume: Concatenated cost volume.
        """

        B, C, H, W = refimg_fea.shape
        C = 2 * C

        tmp_volume = []
        for i in range(maxdisp):
            if i > 0:
                tmp = self.c_cat[i].cat(
                    (refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]),
                    dim=1,
                )
                tmp_volume.append(self.c_pad[i](tmp).view(-1, 1, H, W))
            else:
                tmp_volume.append(
                    self.c_cat[i]
                    .cat((refimg_fea, targetimg_fea), dim=1)
                    .view(-1, 1, H, W)
                )
        volume = self.c_cat_final.cat(tmp_volume, dim=1).view(
            B, C, maxdisp, H, W
        )

        return volume

    @fx_wrap()
    def groupwise_correlation(
        self,
        d: int,
        fea1: Tensor,
        fea2: Tensor,
        num_groups: int,
    ) -> Tensor:
        """
        Compute groupwise correlation using the same approach as GWC-Net.

        Args:
            d: Index of the FloatFunctional.
            fea1: Left image featuremap.
            fea2: Right image featuremap.
            num_groups: Number of groups for groupwise correlation.

        Returns:
            cost_new: Groupwise correlation result.
        """

        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups

        cost = (
            self.gc_mul[d].mul(fea1, fea2).view([-1, channels_per_group, H, W])
        )
        cost_new = (
            self.gc_mean[d].mean(cost, dim=1).view([B, num_groups, H, W])
        )
        assert cost_new.shape == (B, num_groups, H, W)
        return cost_new

    @fx_wrap()
    def build_gwc_volume(
        self,
        refimg_fea: Tensor,
        targetimg_fea: Tensor,
        maxdisp: int,
        num_groups: int,
    ):
        """
        Build the cost volume using the same approach as GWC-Net.

        Args:
            refimg_fea: Left image feature.
            targetimg_fea: Right image feature.
            maxdisp: Maximum disparity value.
            num_groups: Number of groups for groupwise correlation.

        Returns:
            volume: Groupwise correlation cost volume.
        """

        B, C, H, W = refimg_fea.shape
        tmp_volume = []
        for i in range(maxdisp):
            if i > 0:
                tmp = self.groupwise_correlation(
                    i,
                    refimg_fea[:, :, :, i:],
                    targetimg_fea[:, :, :, :-i],
                    num_groups,
                )
                tmp_volume.append(self.gc_pad[i](tmp).view(-1, 1, H, W))
            else:
                tmp_volume.append(
                    self.groupwise_correlation(
                        i, refimg_fea, targetimg_fea, num_groups
                    ).view(-1, 1, H, W)
                )

        volume = self.gc_cat_final.cat(tmp_volume, dim=1).view(
            B, num_groups, maxdisp, H, W
        )
        return volume

    @fx_wrap()
    def dis_mul(self, x: Tensor) -> Tensor:
        """Mul weight to the disparity."""

        disp_values = self.quant_dispvalue(self.disp_values)
        return self.disp_mul_op.mul(x, disp_values)

    @fx_wrap()
    def dis_sum(self, x: Tensor) -> Tensor:
        """Get the low disparity."""

        return self.disp_sum_op.sum(x, dim=1, keepdim=True)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform the forward pass of the model.

        Args:
            features: The inputs featuremaps.

        Returns:
            pred_pyramid_list: The normalized predictions of the model.
        """

        (
            gwc_feature_left,
            gwc_feature_right,
            concat_feature_left,
            concat_feature_right,
            imgl,
        ) = features
        B, C, H, W = gwc_feature_left.shape
        # Build GWC volume
        gwc_volume = self.build_gwc_volume(
            gwc_feature_left, gwc_feature_right, self.D, self.num_groups
        )
        # Build concatenated volume
        concat_volume = self.build_concat_volume(
            concat_feature_left,
            concat_feature_right,
            self.D,
        )

        # Concatenate gwc_volume_4D and concat_volume_4D
        gwc_volume_4D = gwc_volume.view(B, self.num_groups, -1, W)
        concat_volume_4D = concat_volume.view(B, 32, -1, W)
        cost0 = self.cat_volume.cat((gwc_volume_4D, concat_volume_4D), 1).view(
            B, 64, self.D, H, W
        )
        # Optimize cost volume to obtain low disparity
        for f in self.filter:
            cost0 = f(cost0)

        cost0 = self.conv3d_alone(cost0)
        cost0 = cost0.squeeze(1)
        pred0 = self.softmax(cost0)
        pred0 = self.dis_mul(pred0)
        pred0 = self.dis_sum(pred0)

        # Downsampling left image
        img_pyramid_list = []
        for i in range(self.refine_levels):
            img_pyramid_list.append(self.downsample[i](imgl))

        pred_pyramid_list = [pred0]

        # Refinement disparity
        for i in range(self.refine_levels):
            pred_new = self.edge_aware_refinements[i](
                pred_pyramid_list[i], img_pyramid_list[i]
            )
            pred_pyramid_list.append(pred_new)

        length_all = len(pred_pyramid_list)

        for i in range(length_all):
            pred_pyramid_list[i] = self.dequant(pred_pyramid_list[i])
        return pred_pyramid_list

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        for module in self.edge_aware_refinements:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for m in self.filter:
            for m_ in m:
                if hasattr(m_, "fuse_model"):
                    m_.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        module_list = [
            self.disp_mul_op,
            self.disp_sum_op,
        ]
        for m in module_list:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

        self.softmax.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16},
            activation_calibration_qkwargs={"dtype": qint16},
        )

        self.quant_dispvalue.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16},
            activation_calibration_qkwargs={"dtype": qint16},
        )

        for module in self.edge_aware_refinements:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
