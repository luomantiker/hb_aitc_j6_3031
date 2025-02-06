# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Tuple, Union

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from .basic_efficientnet_module import SEBlock
from .conv_module import ConvModule2d

__all__ = [
    "RepBlock",
    "MultiBranchModule",
    "MultiBranchConvModule",
]


class ScaleOp(nn.Module):
    """A PyTorch module for scaling the input tensor.

    Args:
        scale: Either a float representing the constant scaling factor or a
            torch.Tensor representing a learnable scale parameter.
            If a torch.Tensor is provided, it will be registered as a learnable
            parameter and can be trained.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale_op = quantized.FloatFunctional()
        if isinstance(scale, torch.Tensor):
            self.scale = nn.Parameter(scale, requires_grad=True)
            self.scale_quant = QuantStub()
        else:
            self.scale = scale
            self.scale_quant = None

    def forward(self, x):
        if self.scale_quant:
            scale = self.scale_quant(self.scale)
            return self.scale_op.mul(x, scale)
        return self.scale_op.mul(x, self.scale)


class MultiBranchModule(nn.Module):
    """A module that combines multiple branches of computation.

    Args:
        branches: List of branches (sub-modules) to be combined.
        scale_list: List of scaling factors for each branch.
            If None, no scaling is applied. Defaults to None.
    """

    def __init__(
        self,
        branches: List[nn.Module],
        scale_list: List[int] = None,
    ):
        super(MultiBranchModule, self).__init__()
        self.branches = nn.ModuleList(branches)
        self.add_list = nn.ModuleList()
        for _ in range(len(self.branches) - 1):
            self.add_list.append(quantized.FloatFunctional())
        self.scale_list = scale_list
        if scale_list is not None:
            self.scale_op_list = nn.ModuleList()
            assert len(scale_list) == len(
                branches
            ), f"scale_list len:{len(scale_list)}, branch len:{len(branches)}"
            for scale in self.scale_list:
                if not isinstance(scale, torch.Tensor) and scale == 1:
                    self.scale_op_list.append(nn.Identity())
                else:
                    self.scale_op_list.append(ScaleOp(scale))

    def get_branch_scale(self, i):
        if self.scale_list is not None:
            if isinstance(self.scale_list[i], torch.Tensor):
                return self.scale_op_list[i].scale.data
            return self.scale_list[i]
        return 1

    def forward(self, x):
        out = self.branches[0](x)
        if self.scale_list:
            out = self.scale_op_list[0](out)
        for i in range(1, len(self.branches)):
            x1 = self.branches[i](x)
            if self.scale_list:
                x1 = self.scale_op_list[i](x1)
            out = self.add_list[i - 1].add(x1, out)
        return out


class MultiBranchConvModule(nn.Module):
    """A PyTorch Module that aggregates multiple convolutional branches.

    Each branch in the module contains a Conv2d layer with potentially
        different kernel sizes.
    Each branch may optionally contain a BatchNorm2d layer following the
        Conv2d layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        k_size_list: list of kernel_size of conv layer for each branch,
                Note: k_size=0 stands for identity layer.
        bn_flag_list: list of whether add BN layer behind the conv layer for
                each branch.
        stride: Stride to use for the Conv2d layers in the branches.
        dilation: Same as nn.Conv2d.
        groups: Same as nn.Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_size_list: List[int],
        bn_flag_list: List[bool],
        scale_list: List[int] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
    ):
        super(MultiBranchConvModule, self).__init__()
        self.k_size_list = k_size_list
        self.bn_flag_list = bn_flag_list
        assert len(k_size_list) == len(bn_flag_list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.dilation = dilation

        self.branches = nn.ModuleList()
        for k_size, bn_flag in zip(k_size_list, bn_flag_list):
            branch = self.build_branch(k_size, bn_flag)
            if branch is not None:
                self.branches.append(branch)
        self.add_list = nn.ModuleList()
        for _ in range(len(self.branches) - 1):
            self.add_list.append(quantized.FloatFunctional())

        self.scale_list = scale_list
        if scale_list is not None:
            self.scale_op_list = nn.ModuleList()
            assert len(self.scale_list) == len(k_size_list)
            for scale in self.scale_list:
                if not isinstance(scale, torch.Tensor) and scale == 1:
                    self.scale_op_list.append(nn.Identity())
                else:
                    self.scale_op_list.append(ScaleOp(scale))

    def get_branch_scale(self, i):
        if self.scale_list is not None:
            if isinstance(self.scale_list[i], torch.Tensor):
                return self.scale_op_list[i].scale.data
            return self.scale_list[i]
        return 1

    def build_branch(self, k_size, bn):
        if isinstance(k_size, tuple):
            assert len(k_size) == 2 and k_size[0] == k_size[1]
            k_size = k_size[0]
        if k_size == 0:
            if self.in_channels == self.out_channels and self.stride == 1:
                if bn:
                    branch = nn.BatchNorm2d(self.in_channels)
                else:
                    branch = nn.Identity()
            else:
                branch = None
        else:
            branch = ConvModule2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=k_size,
                stride=self.stride,
                padding=(k_size - 1) // 2,
                groups=self.groups,
                dilation=self.dilation,
                bias=False if bn else True,
                norm_layer=nn.BatchNorm2d(self.out_channels) if bn else None,
            )
        return branch

    def forward(self, x):
        out = self.branches[0](x)
        if self.scale_list:
            out = self.scale_op_list[0](out)
        for i in range(1, len(self.branches)):
            x1 = self.branches[i](x)
            if self.scale_list:
                x1 = self.scale_op_list[i](x1)
            out = self.add_list[i - 1].add(x1, out)
        return out


class RepBlock(nn.Module):
    """A Block that could reparameterize given module.

    Args:
        module: The module to be reparameterized.
            The given module should contain branches, and each branch has the
            structure of either ConvModule2d (Conv2d + BN(optional)),
            nn.Identity and BatchNorm2d.
            Recommend to use `MultiBranchConvModule`.
        norm_layer: Norm_layer following the module
        use_se: whether add SE module.
        act_layer: Activation layer.
    """

    def __init__(
        self,
        module: nn.Module,
        norm_layer: nn.Module = None,
        use_se: bool = False,
        act_layer: nn.Module = None,
        deploy=False,
    ):
        super(RepBlock, self).__init__()
        self.module = module
        assert hasattr(module, "branches")
        self.conv_args = self.check_branches(module)
        self.kernel_size = self.conv_args["kernel_size"]
        self.conv_args["padding"] = (
            (self.kernel_size[0] - 1) * self.conv_args["dilation"][0]
        ) // 2

        self.has_norm_layer = norm_layer is not None
        self.use_se = use_se
        self.has_act_layer = act_layer is not None

        out_channels = self.conv_args["out_channels"]

        post_layers = []
        if norm_layer:
            post_layers.append(norm_layer)
        if use_se:
            post_layers.append(
                SEBlock(
                    out_channels,
                    out_channels // 16,
                    out_channels,
                    nn.ReLU(),
                    adapt_pooling=True,
                )
            )
        if act_layer:
            post_layers.append(act_layer)
        self.post_layers = nn.Sequential(*post_layers)

        self.deploy = False
        if deploy:
            self.switch_to_deploy()

    def forward(self, x):
        x = self.module(x)
        x = self.post_layers(x)
        return x

    def check_branches(self, module):
        """Check given module branch could be reparameterized.

        Return: arguments for building the rep convolution layer.
        """

        def check_equal(arr):
            return all([elem == arr[0] for elem in arr[1:]])

        conv_args = [
            "in_channels",
            "out_channels",
            "stride",
            "kernel_size",
            "groups",
            "dilation",
        ]
        conv_args_dict = {k: [] for k in conv_args}
        for branch in module.branches:
            if isinstance(branch, nn.Conv2d):
                for arg in conv_args_dict:
                    conv_args_dict[arg].append(getattr(branch, arg))
            elif isinstance(branch, ConvModule2d):
                for arg in conv_args_dict:
                    conv_args_dict[arg].append(getattr(branch[0], arg))
                assert (
                    len(branch) <= 2
                ), "The ConvModule2d layer should not have nonlinear layer."
            elif isinstance(branch, (nn.Identity, nn.BatchNorm2d)):
                conv_args_dict["kernel_size"].append((1, 1))
            elif hasattr(branch, "get_conv_bn") and hasattr(
                branch, "conv_args"
            ):
                for arg in conv_args_dict:
                    conv_args_dict[arg].append(branch.conv_args[arg])
            else:
                raise TypeError(
                    "Each branch should be either ConvModule2d"
                    + " Identity or BatchNorm2d."
                )
        for arg in conv_args:
            if arg != "kernel_size":
                assert check_equal(
                    conv_args_dict[arg]
                ), f"arg {arg} should be consistent in multi conv branches."
        for k in conv_args_dict["kernel_size"]:
            assert k[0] % 2 == 1, "Kernel size should be odd."

        if conv_args_dict["in_channels"] == []:
            conv_args_dict["in_channels"].append(module.in_channels)
            conv_args_dict["out_channels"].append(module.out_channels)
            conv_args_dict["stride"].append((1, 1))
            conv_args_dict["groups"].append(1)
            conv_args_dict["dilation"].append((1, 1))
        conv_args = {k: conv_args_dict[k][0] for k in conv_args_dict.keys()}
        k_size = [
            max([k[i] for k in conv_args_dict["kernel_size"]])
            for i in range(2)
        ]
        conv_args["kernel_size"] = tuple(k_size)

        return conv_args

    def get_conv_bn(self):
        assert self.deploy, "Only deploy mode RepBlock could get conv weights"
        assert self.use_se is False
        assert self.has_act_layer is False
        if self.has_norm_layer:
            return self.module, self.post_layers[0]
        else:
            return self.module, None

    def switch_to_deploy(self):
        if not self.deploy:
            for m in self.module.branches:
                if hasattr(m, "switch_to_deploy"):
                    m.switch_to_deploy()
            module = self.convert_rep(self.module)
            del self.module
            self.module = module
            if not self.training:
                self.module.eval()
        self.deploy = True

    def convert_rep(self, module):
        """Convert module to reparameterized conv layer."""
        weight_list = []
        bias_list = []
        scale_list = []
        for i, branch in enumerate(module.branches):
            weight, bias = self._fuse_bn_tensor(branch)
            if hasattr(module, "get_branch_scale"):
                scale = module.get_branch_scale(i)
            else:
                scale = 1
            weight_list.append(weight)
            bias_list.append(bias)
            scale_list.append(scale)

        kernel, bias = 0, 0
        kernel_size = self.conv_args["kernel_size"]
        for i, (weight, b) in enumerate(zip(weight_list, bias_list)):
            padded_k = self._pad_kernel(
                weight,
                kernel_size,
            )

            if isinstance(scale_list[i], torch.Tensor):
                kernel += padded_k * scale_list[i].unsqueeze(1)
                bias += b * scale_list[i].squeeze()
            else:
                kernel += padded_k * scale_list[i]
                bias += b * scale_list[i]
        rep_conv = nn.Conv2d(**self.conv_args)
        rep_conv.weight.data = kernel
        rep_conv.bias.data = bias
        return rep_conv

    def _get_id_tensor(self):
        in_channels = self.conv_args["in_channels"]
        groups = self.conv_args["groups"]
        input_dim = in_channels // groups
        kernel_value = torch.zeros((in_channels, input_dim, 1, 1))
        for i in range(in_channels):
            kernel_value[i, i % input_dim, 0, 0] = 1
        return kernel_value

    def _fuse_bn_tensor(self, branch):
        """Given branch, fuse BN and return rep conv weight and bias."""
        if branch is None:
            return 0, 0
        elif isinstance(branch, nn.Identity):
            id_tensor = self._get_id_tensor()
            return id_tensor, 0
        elif isinstance(branch, nn.Conv2d):
            if branch.bias is None:
                bias = 0
            else:
                bias = branch.bias.data
            return branch.weight.data, bias
        elif isinstance(branch, ConvModule2d) and len(branch) == 1:
            return branch[0].weight.data, branch[0].bias.data
        elif isinstance(branch, ConvModule2d) and len(branch) > 1:
            kernel = branch[0].weight  # branch.conv
            bias = branch[0].bias
            if bias is None:
                bias = 0
            bn = branch[1]
        elif isinstance(branch, nn.BatchNorm2d):
            kernel = self._get_id_tensor().to(branch.weight.device)
            bias = 0
            bn = branch
        elif hasattr(branch, "get_conv_bn"):
            conv, bn = branch.get_conv_bn()
            if bn is None:
                return conv.weight.data, conv.bias.data
            kernel = conv.weight
            bias = 0 if conv.bias is None else conv.bias

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return (kernel * t).data, (
            beta + (bias - running_mean) * gamma / std
        ).data

    def _pad_kernel(self, kernel, k_size):
        if not isinstance(kernel, int):
            pad_size = (k_size[0] - kernel.shape[-2]) // 2
            padding = [pad_size, pad_size, pad_size, pad_size]
            kernel = torch.nn.functional.pad(kernel, padding)
        return kernel
