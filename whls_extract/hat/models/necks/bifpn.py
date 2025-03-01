# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Union

import horizon_plugin_pytorch.nn as nnf
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.nn import ModuleDict, ModuleList, Sequential
from torch.nn.quantized import FloatFunctional

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.utils import _check_strides
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.logger import rank_zero_info

__all__ = ["BiFPN"]

logger = logging.getLogger(__name__)


class MaybeApply1x1(nn.Module):
    """Use conv1x1 and bn to change channel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_bn (bool): if use bn

    """

    def __init__(self, in_channels, out_channels, use_bn=False):
        super(MaybeApply1x1, self).__init__()
        if in_channels == out_channels:
            return None
        self.lateral_conv = ModuleList()
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.lateral_conv.append(conv)
        if use_bn:
            norm = nn.BatchNorm2d(out_channels)
            self.lateral_conv.append(norm)

    def forward(self, x):
        if hasattr(self, "lateral_conv"):
            for module in self.lateral_conv:
                x = module(x)
            return x
        else:
            return x


class Resize(nn.Module):
    """The layer is used to change the feather map size or keep shape.

    # TODO(min.du, 1.0): move to hat/ops #

    Args:
        sampling: Sampling way, the candidate is ['down', 'up', 'keep'].
            e.g. 'down' : downsampling.
                 'up'   : upsampling.
                 'keep' : keep shape unchanged.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        pooling_type: Pooling type, the candidate is ['max', 'avg'].
        use_bn: if use bn.
        conv_after_downsample: Whether 1X1 conv is placed after downsample.
        upsample_type: use module or function unsample, the candidate is
            ['module', 'function', 'torch_c'].
            This parameter is added for compatibility considerations.
            Due to historical reasons,there are two different
            implementations of resize in the horizon_plugin_pytorch. They are
            exactly the same except their saved checkpoints. For 'module', the
            state dict of its qat model will have one more fake quant than
            'function'.
            'torch_c' optimize for torchDynamo/tensorRT
            use origin c op due to horizon_plugin_pytorch
            hook lead to trace graph break.
    """

    def __init__(
        self,
        sampling: str,
        in_channels: int,
        out_channels: int,
        pooling_type: str = "max",
        use_bn: bool = True,
        conv_after_downsample: bool = False,
        upsample_type: str = "module",
    ):
        super(Resize, self).__init__()
        assert sampling in ["down", "up", "keep"]
        assert pooling_type in ["max", "avg"]
        assert upsample_type in ["module", "function", "torch_c"]
        self.sampling = sampling
        self.resize_layer = ModuleList()
        if sampling == "down":
            if not conv_after_downsample:
                lateral_conv = MaybeApply1x1(in_channels, out_channels, use_bn)
                if lateral_conv:
                    self.resize_layer.append(lateral_conv)
            if pooling_type == "max":
                pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            elif pooling_type == "avg":
                pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.resize_layer.append(pooling)
            if conv_after_downsample:
                lateral_conv = MaybeApply1x1(in_channels, out_channels, use_bn)
                if lateral_conv:
                    self.resize_layer.append(lateral_conv)
        else:
            lateral_conv = MaybeApply1x1(in_channels, out_channels, use_bn)
            if lateral_conv:
                self.resize_layer.append(lateral_conv)

        self.upsample_type = upsample_type
        if sampling == "up" and upsample_type == "module":
            self.upsampling = nnf.Interpolate(
                scale_factor=2, mode="bilinear", recompute_scale_factor=True
            )

    def forward(self, x):
        """Forward feature.

        Args:
            x (tensor): input tensor

        Returns (tensor): resized tensor

        """

        for module in self.resize_layer:
            x = module(x)
        if self.sampling == "up":
            if self.upsample_type == "module":
                x = self.upsampling(x)
            elif self.upsample_type == "torch_c":
                # optimize for torchDynamo/tensorRT
                # use origin c op due to horizon_plugin_pytorch
                # hook lead to trace graph break.
                # For tensorRT optimize,
                # Please run before:
                # from hat.utils.trt_fx_extension import load_extension
                # load_extension("c_interpolate")
                x = torch._C._nn.upsample_bilinear2d(
                    x,
                    output_size=None,
                    align_corners=False,
                    scale_factors=(2, 2),
                )
            else:
                x = nn.functional.interpolate(
                    x,
                    scale_factor=2,
                    mode="bilinear",
                    recompute_scale_factor=True,
                )
        return x


class Fusion(nn.Module):
    """Multi-level feature fusion.

    Args:
        weight_len: Number of the input tensors
        weight_method: sum or fastattn
        eps: Avoid except 0
        use_fx: Whether use fx mode qat. Default: False.

    """

    def __init__(
        self,
        in_channels: int,
        weight_len: int,
        weight_method: str = "sum",
        eps: float = 0.0001,
        use_fx: bool = False,
    ):
        super(Fusion, self).__init__()
        self.weight_method = weight_method
        self.eps = eps
        self.use_fx = use_fx
        if not use_fx:
            self.floatF = FloatFunctional()
        if weight_method == "fastattn":
            self.edge_weights = nn.Parameter(
                torch.ones(weight_len, dtype=torch.float32), requires_grad=True
            )
            self.relu = nn.ReLU(inplace=True)

    def float_function_sum(self, x):
        x = _as_list(x)
        for i, val in enumerate(x):
            if i == 0:
                res = val
            else:
                if self.use_fx:
                    # do not use "+="", since it is an inplace operation
                    res = res + val
                else:
                    res = self.floatF.add(res, val)
        return res

    def forward(self, x):
        x = list(x)
        if self.weight_method == "sum":
            return self.float_function_sum(x)
        elif self.weight_method == "fastattn":
            # edge_weights would be followed with relu to become positive
            relu_edge_weights = self.relu(self.edge_weights)
            weights_sum = self.float_function_sum(relu_edge_weights)
            for i in range(len(x)):
                x[i] = x[i] * relu_edge_weights[i]
                x[i] = x[i] / (weights_sum + self.eps)
            return self.float_function_sum(x)


class BifpnLayer(nn.Module):
    """The basic structure of BiFPN.

    Args:
        fpn_config: The dict is used for build the bifpn node
        out_index: Get final output tensor list
        use_fx: Whether use fx mode qat. Default: False.

    """

    def __init__(
        self,
        fpn_config: Dict,
        out_index: List[int] = None,
        use_fx: bool = False,
    ):

        super(BifpnLayer, self).__init__()
        self.fpn_config = fpn_config
        self.out_index = out_index
        level = fpn_config.level
        in_channels = fpn_config.in_channels
        out_channels = fpn_config.out_channels
        offset2inchannels = {
            0: out_channels[0],
            1: out_channels[1],
            2: out_channels[2],
            3: out_channels[3],
            4: out_channels[4],
            5: out_channels[3],
            6: out_channels[2],
            7: out_channels[1],
            8: out_channels[0],
            9: out_channels[1],
            10: out_channels[2],
            11: out_channels[3],
        }
        offset2out_channels = {
            0: {"keep": out_channels[0]},
            1: {"keep": out_channels[1]},
            2: {"keep": out_channels[2]},
            3: {"keep": out_channels[3]},
            4: {"keep": out_channels[4], "up": out_channels[3]},
            5: {"keep": out_channels[3], "up": out_channels[2]},
            6: {"keep": out_channels[2], "up": out_channels[1]},
            7: {"keep": out_channels[1], "up": out_channels[0]},
            8: {"down": out_channels[1]},
            9: {"down": out_channels[2]},
            10: {"down": out_channels[3]},
            11: {"down": out_channels[4]},
        }
        weight_method = fpn_config.weight_method
        self.all_nodes = ModuleDict()
        for i, fnode in enumerate(fpn_config.nodes):
            rank_zero_info(f"fnode {i} : {fnode}")
            node = ModuleList()
            out_ch = 0
            # resize
            for offset, sampling, upsample_type in zip(
                fnode["inputs_offsets"],
                fnode["sampling"],
                fnode["upsample_type"],
            ):
                in_ch = (
                    in_channels[offset]
                    if offset < level
                    else offset2inchannels[offset]
                )
                out_ch = offset2out_channels[offset][sampling]
                node.append(
                    Resize(
                        sampling,
                        in_ch,
                        out_ch,
                        upsample_type=upsample_type,
                    )
                )

            # fusion
            node.append(
                Fusion(
                    out_ch,
                    len(fnode["inputs_offsets"]),
                    weight_method,
                    use_fx=use_fx,
                )
            )

            # relu, conv, bn
            node.append(
                Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_ch,
                        out_ch,
                        kernel_size=3,
                        stride=1,
                        bias=False,
                        groups=out_ch,
                        padding=1,
                    ),
                    ConvModule2d(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        groups=1,
                        padding=0,
                        norm_layer=nn.BatchNorm2d(out_ch),
                    ),
                )
            )
            # append with name
            self.all_nodes[str(i)] = node

    def forward(self, x):
        x = list(x)
        assert len(x) == self.fpn_config.level
        for i, fnode in enumerate(self.fpn_config.nodes):
            nodes = []
            for idx, input_offset in enumerate(fnode["inputs_offsets"]):
                input_node = x[input_offset]
                input_node = self.all_nodes[str(i)][idx](input_node)  # resize
                nodes.append(input_node)
            # fusion
            new_node = self.all_nodes[str(i)][idx + 1](nodes)
            # activation + separable conv + bn
            new_node = self.all_nodes[str(i)][idx + 2](new_node)
            x.append(new_node)

        all_outs = [x[i] for i in range(-self.fpn_config.level, 0, 1)]
        if self.out_index is not None:
            return [all_outs[i] for i in self.out_index]
        return all_outs


def get_fpn_config(
    fpn_name="bifpn_sum", out_channels=64, upsample_type="module"
):
    assert fpn_name in ["bifpn_sum", "bifpn_fa"]
    fpn_config = edict()
    fpn_config.out_channels = out_channels
    fpn_config.level = 5
    # define connection method
    fpn_config.nodes = [
        {
            "inputs_offsets": [3, 4],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [2, 5],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [1, 6],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [0, 7],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [1, 7, 8],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type, upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [2, 6, 9],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type, upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [3, 5, 10],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type, upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [4, 11],
            "sampling": ["keep", "down"],
            "upsample_type": [upsample_type, upsample_type],
        },
    ]
    # define weighting method
    fpn_config.weight_method = "sum"
    if fpn_name == "bifpn_fa":
        fpn_config.weight_method = "fastattn"

    return fpn_config


@OBJECT_REGISTRY.register
class BiFPN(nn.Module):
    """Weighted Bi-directional Feature Pyramid Network(BiFPN).

    This is an implementation of - EfficientDet: Scalable and Efficient Object
    Detection (https://arxiv.org/abs/1911.09070)

    Args:
        in_strides: Stride of input feature map
        out_strides: Stride of output feature map
        stride2channels: The key:value is stride:channel ,
            the channles have been multipified by alpha
        out_channels: Channel number of output layer, the key:value
            is stride:channel.
        num_outs: Number of BifpnLayer's input, the value is must 5,
            because the bifpn layer is fixed
        stack: Number of BifpnLayer
        start_level: Index of the start input backbone level
            used to build the feature pyramid. Default: 0.
        end_level: Index of the end input backbone level (exclusive)
            to build the feature pyramid. Default: -1, means the last level.
        fpn_name: the value is mutst between with 'bifpn_sum', 'bifpn_fa'.
        upsample_type: use module or function unsample, the candidate is
            ['module', 'function'].
        use_fx: Whether use fx mode qat. Default: False.

    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: int,
        stride2channels: Dict,
        out_channels: Union[int, Dict],
        num_outs: int,
        stack: int = 3,
        start_level: int = 0,
        end_level: int = -1,
        fpn_name: str = "bifpn_sum",
        upsample_type: str = "module",
        use_fx: bool = False,
    ):
        super(BiFPN, self).__init__()

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.stride2channels = stride2channels
        self.in_channels = [stride2channels[stride] for stride in in_strides]
        assert isinstance(out_channels, int) or isinstance(out_channels, dict)
        self.out_channels = (
            [out_channels[stride] for stride in out_strides]
            if isinstance(out_channels, dict)
            else [out_channels for _ in out_strides]
        )
        self.num_ins = len(in_strides)
        self.num_outs = num_outs

        # assert in_strides in stride2channels
        self.in_strides = _check_strides(
            in_strides, self.stride2channels.keys()
        )

        assert len(self.out_strides) <= num_outs
        assert stack >= 1
        self.stack = stack
        self.fpn_config = get_fpn_config(
            fpn_name,
            self.out_channels,
            upsample_type,
        )

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        # add extra downsample layers (stride-2 pooling or conv + pooling)
        # to build extra input features that are not from backbone.
        extra_levels = (
            self.num_outs - self.backbone_end_level + self.start_level
        )
        # channels for multi-level feature-map used in bifpn
        self.fpn_config.in_channels = self.in_channels[
            self.start_level : self.backbone_end_level
        ] + [
            self.out_channels[self.backbone_end_level - self.start_level + i]
            for i in range(extra_levels)
        ]
        # bifpn total out strides and out_index
        self.total_out_strides = self.in_strides[
            self.start_level : self.backbone_end_level
        ] + [
            self.in_strides[self.backbone_end_level - 1] * (2 ** (i + 1))
            for i in range(extra_levels)
        ]
        self.out_index = [
            self.total_out_strides.index(stride) for stride in self.out_strides
        ]
        # build extra_downsample layers
        self.extra_downsamples = ModuleList()
        for i in range(extra_levels):
            out_channels = self.out_channels[
                self.backbone_end_level - self.start_level + i
            ]
            downsample = Resize(
                sampling="down",
                in_channels=self.in_channels[-1],
                out_channels=out_channels,
                pooling_type="max",
                use_bn=True,
                conv_after_downsample=False,
                upsample_type=upsample_type,
            )
            self.in_channels[-1] = out_channels
            self.extra_downsamples.append(downsample)
        # repeat build bifpn layer many times
        self.bifpn_layers = ModuleList()
        for i in range(self.stack):
            rank_zero_info("building bifpn cell %d" % (i))
            if i == self.stack - 1:
                out_index = self.out_index
            else:
                out_index = None
            bifpn_layer = BifpnLayer(
                edict(self.fpn_config.copy()),
                out_index,
                use_fx=use_fx,
            )
            self.bifpn_layers.append(bifpn_layer)
            # update fpn_in_channels, only need to update once
            if i == 0:
                self.fpn_config.in_channels = [
                    self.out_channels[i] for i in range(self.num_outs)
                ]
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of BiFPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, inputs):
        """Forward features.

        Args:
            inputs (list[tensor]): Input tensors

        Returns (list[tensor]): Output tensors

        """
        assert len(inputs) == len(self.in_channels)
        x = list(inputs[self.start_level : self.backbone_end_level])
        # build extra input features
        for downsample in self.extra_downsamples:
            x.append(downsample(x[-1]))
        # repeat bifpn
        for i in range(self.stack):
            x = self.bifpn_layers[i](x)
        return x

    def fuse_model(self):
        try:
            from horizon_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except ImportError:
            logging.warning(
                "Please install horizon_plugin_pytorch first, otherwise use "
                "pytorch official quantification"
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules

        total_fuse = 0
        for m in self.modules():
            if type(m) == MaybeApply1x1:
                if hasattr(m, "lateral_conv"):
                    torch.quantization.fuse_modules(
                        m,
                        ["lateral_conv.0", "lateral_conv.1"],
                        inplace=True,
                        fuser_func=fuser_func,
                    )
                    total_fuse += 1
            elif type(m) == ConvModule2d:
                m.fuse_model()
                total_fuse += 1
        rank_zero_info("neck total_fuse  {}".format(total_fuse))

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
