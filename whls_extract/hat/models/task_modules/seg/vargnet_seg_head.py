# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate
from horizon_plugin_pytorch.nn.functional import rle
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import OnePathResUnit
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.extend_container import ExtSequential
from hat.models.base_modules.separable_conv_module import (
    SeparableGroupConvModule2d,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FRCNNSegHead(nn.Module):
    """FRCNNSegHead module for segmentation task.

    Args:
        group_base: Group base of group conv
        in_strides: The strides corresponding to the inputs of
                    seg_head, the inputs usually come from backbone or neck.
        in_channels: Number of channels of each input stride.
        out_strides: List of output strides.
        out_channels: Number of channels of each output stride.
        bn_kwargs: Extra keyword arguments for bn layers.
        proj_channel_multiplier: Multiplier of channels of
                    pw conv in block.
        with_extra_conv: Whether to use extra conv module.
        use_bias: Whether to use bias in conv module.
        linear_out: Whether NOT to use to act of pw.
        argmax_output: Whether conduct argmax on output.
        with_score: Whether to keep score in argmax operation.
        rle_label: Whether to calculate rle representation of label output.
        dequant_output: Whether to dequant output.
        int8_output: If True, output int8, otherwise output int32.
        no_upscale_infer: Load params from x2 scale if True.
    """

    def __init__(
        self,
        group_base: int,
        in_strides: List,
        in_channels: List,
        out_strides: List,
        out_channels: List,
        bn_kwargs: Dict,
        proj_channel_multiplier: float = 1.0,
        with_extra_conv: bool = False,
        use_bias: bool = True,
        linear_out: bool = True,
        argmax_output: bool = False,
        with_score: bool = False,
        rle_label: bool = False,
        dequant_output: bool = True,
        int8_output: bool = False,
        no_upscale_infer: bool = False,
    ):
        super().__init__()
        if argmax_output:
            assert (
                not dequant_output
            ), "argmax output tensor, no need to dequant"
        else:
            assert not (
                with_score or rle_label
            ), "with_score and rle_label are not available if no argmax op"

        assert len(in_strides) == len(in_channels), "%d vs. %d" % (
            len(in_strides),
            len(in_channels),
        )
        assert len(out_strides) == len(out_channels), "%d vs. %d" % (
            len(out_strides),
            len(out_channels),
        )
        assert out_strides[-1] <= in_strides[-1], "%d vs. %d" % (
            out_strides[-1],
            in_strides[-1],
        )

        stride2channels = {s: c for s, c in zip(in_strides, in_channels)}

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.no_upscale_infer = no_upscale_infer

        self.dequant_output = dequant_output
        self.argmax_output = argmax_output
        self.with_score = with_score
        self.rle_label = rle_label
        self.take_strides = []
        self.output_blocks = nn.ModuleDict()

        for stride, channel in zip(out_strides, out_channels):
            extra_block = []
            assert (
                stride >= in_strides[0] // 4
            ), f"stride {stride} out of upscale 4 of in_strides[0]: {in_strides[0]}"  # noqa
            if stride == in_strides[0] // 4:
                upscale = Interpolate(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
                extra_block.append(upscale)
                gc_num_filter = stride2channels[in_strides[0]]
                pw_num_filter = stride2channels[in_strides[0]]
                self.take_strides.append(in_strides[0])
            elif stride == in_strides[0] // 2:
                upscale = Interpolate(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
                extra_block.append(upscale)
                gc_num_filter = stride2channels[in_strides[0]]
                pw_num_filter = stride2channels[in_strides[0]]
                self.take_strides.append(in_strides[0])
            else:
                # Add this redundant Interpolate op to deal with abnormal
                # backward behavior introduced in torch 1.9.1
                # TODO: remove this redundant op
                upscale = Interpolate(
                    scale_factor=1,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
                extra_block.append(upscale)
                gc_num_filter = stride2channels[stride]
                pw_num_filter = stride2channels[stride]
                self.take_strides.append(stride)

            kwargs = {}
            if extra_block:
                kwargs = {"extra_block": ExtSequential(extra_block)}

            block = OutputBlockEx(
                gc_num_filter=gc_num_filter,
                gc_group_base=group_base,
                pw_num_filter=int(pw_num_filter * proj_channel_multiplier),
                out_c=channel,
                bn_kwargs=bn_kwargs,
                use_bias=use_bias,
                linear_out=linear_out,
                with_extra_conv=with_extra_conv,
                int8_output=int8_output,
                **kwargs,
            )
            if self.out_strides[0] == stride and self.no_upscale_infer:
                self.output_blocks["stride_%d" % (stride // 2)] = block
            else:
                self.output_blocks[f"stride_{stride}"] = block
        self.dequant = DeQuantStub()

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:

        take_features = []
        for stride in self.take_strides:
            take_features.append(x[self.in_strides.index(stride)])

        outputs = []

        for feat, s in zip(take_features, self.out_strides):
            if self.out_strides[0] == s and self.no_upscale_infer:
                pred = self.output_blocks["stride_%d" % (s // 2)](feat)
            else:
                pred = self.output_blocks[f"stride_{s}"](feat)

            if self.dequant_output:
                pred = self.dequant(pred)

            if self.argmax_output:
                if self.with_score:
                    score, pred = pred.max(dim=1, keepdim=True)
                    res = [score, pred]
                else:
                    pred = pred.argmax(dim=1, keepdim=True)
                    res = [pred]
                if self.rle_label:
                    rle_label = rle(pred, torch.int8)
                    assert len(rle_label) == 1
                    res.append(rle_label[0])
                res = tuple(res)
                if len(res) == 1:
                    res = res[0]
            else:
                res = pred

            outputs.append(res)

        return tuple(outputs)

    def fuse_model(self):
        for m in self.output_blocks.values():
            m.fuse_model()


class OutputBlockEx(nn.Sequential):
    """Output block module for FRCNNSegHead.

    Args:
        gc_num_filter: Num filters of dw conv.
        gc_group_base: Group base of group conv.
        pw_num_filter: Num filters of pw conv.
        out_c: Out channels for conv module.
        bn_kwargs: Kwargs of BN layer.
        use_bias: Whether to use bias.
        linear_out: Whether NOT to use to act of pw.
        with_extra_conv: Whether to use extra conv module.
        extra_block: Define extra block.
        factor: Factor for channels expansion.Default is 2.
        int8_output:If True, output int8, otherwise output int32.
    """

    def __init__(
        self,
        gc_num_filter: int,
        gc_group_base: int,
        pw_num_filter: int,
        out_c: int,
        bn_kwargs: Dict,
        use_bias: bool = True,
        linear_out: bool = True,
        with_extra_conv: bool = False,
        extra_block: Optional[nn.Module] = None,
        factor: float = 2.0,
        int8_output: bool = False,
    ):
        if with_extra_conv:
            assert pw_num_filter % gc_group_base == 0

        self.int8_output = int8_output
        self.with_extra = extra_block is not None

        blocks = [
            OnePathResUnit(
                dw_num_filter=gc_num_filter,
                group_base=gc_group_base,
                pw_num_filter=pw_num_filter,
                pw_num_filter2=pw_num_filter,
                stride=1,
                is_dim_match=(gc_num_filter == pw_num_filter),
                use_bias=use_bias,
                bn_kwargs=bn_kwargs,
                pw_with_act=not linear_out,
                factor=factor,
            )
        ]
        if with_extra_conv:
            blocks.append(
                SeparableGroupConvModule2d(
                    in_channels=pw_num_filter
                    * (1 if gc_group_base == 8 else 2),
                    out_channels=pw_num_filter,
                    groups=int(pw_num_filter / gc_group_base),
                    dw_act_layer=nn.ReLU(inplace=True),
                    dw_norm_layer=nn.BatchNorm2d(
                        pw_num_filter * (1 if gc_group_base == 8 else 2),
                        **bn_kwargs,
                    ),
                    pw_norm_layer=nn.BatchNorm2d(pw_num_filter, **bn_kwargs),
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                )
            )
        blocks.append(
            ConvModule2d(
                in_channels=pw_num_filter,
                out_channels=out_c,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                act_layer=None,
                bias=use_bias,
            ),
        )
        if extra_block is not None:
            blocks.append(extra_block)

        super(OutputBlockEx, self).__init__(*blocks)

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        idx = -2 if self.with_extra else -1
        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self[idx].qconfig = qconfig_manager.get_default_qat_out_qconfig()
