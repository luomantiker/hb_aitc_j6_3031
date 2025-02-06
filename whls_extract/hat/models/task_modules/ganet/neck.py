# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Tuple

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.embeddings import PositionEmbeddingSine
from hat.models.weight_init import kaiming_init, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["GaNetNeck"]


class AttentionLayer(nn.Module):
    """
    Position attention module for ganet neck.

    Args:
        in_channel: Channel of input feature map.
        out_channel: Channel of output feature maps.
        ratio: Ratio of channel in hidden layer.
        pos_shape: Shape of pos embed.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        ratio: int = 4,
        pos_shape: Tuple[int, int, int] = (1, 10, 25),
    ):
        super(AttentionLayer, self).__init__()
        self.pre_conv = ConvModule2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(out_channel),
            act_layer=nn.ReLU(inplace=True),
        )

        self.query_conv = ConvModule2d(
            in_channels=out_channel,
            out_channels=out_channel // ratio,
            kernel_size=1,
            stride=1,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )

        self.key_conv = ConvModule2d(
            in_channels=out_channel,
            out_channels=out_channel // ratio,
            kernel_size=1,
            stride=1,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )

        self.value_conv = ConvModule2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            norm_layer=None,
            act_layer=None,
        )

        self.final_conv = ConvModule2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(out_channel),
            act_layer=nn.ReLU(inplace=True),
        )

        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.matmul_1 = quantized.FloatFunctional()
        self.matmul_2 = quantized.FloatFunctional()
        self.add_1 = quantized.FloatFunctional()
        self.add_2 = quantized.FloatFunctional()
        self.mul = quantized.FloatFunctional()
        self.pos_embed_quant = QuantStub(scale=None)
        self.gamma_quant = QuantStub(scale=None)

        self.pos_embed = self._build_position_encoding(out_channel, pos_shape)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of GaNetHead module."""
        for m in self.pre_conv.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(
                    m,
                    mode="fan_out",
                    nonlinearity="relu",
                    bias=0,
                    distribution="normal",
                )
        for m in self.query_conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=0.0)

        for m in self.key_conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=0.0)

        for m in self.value_conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=0.0)

        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(
                    m,
                    mode="fan_out",
                    nonlinearity="relu",
                    bias=0,
                    distribution="normal",
                )

    def _build_position_encoding(self, hidden_dim, shape):
        mask = torch.zeros(shape, dtype=torch.bool)
        pos_module = PositionEmbeddingSine(hidden_dim // 2)
        pos_embs = pos_module(mask)
        return pos_embs

    @fx_wrap()
    def _proj(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, 1, -1, height * width
        )
        proj_key = self.key_conv(x).permute(0, 2, 3, 1)

        energy = self.matmul_1.matmul(proj_key, proj_query)
        energy = energy.view(m_batchsize, -1, height * width)
        attention = self.softmax(energy)
        attention = attention.view(m_batchsize, 1, -1, height * width)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, 1, width * height
        )
        out = self.matmul_2.matmul(proj_value, attention)
        out = out.view(m_batchsize, -1, height, width)
        return out

    def forward(self, x):
        pos_embed = self.pos_embed_quant(self.pos_embed.to(x.device))
        x = self.pre_conv(x)
        x = self.add_1.add(x, pos_embed)
        out = self._proj(x)
        gamma_quant = self.gamma_quant(self.gamma)
        feat_mul = self.mul.mul(gamma_quant, out)
        out_feat = self.add_2.add(feat_mul, x)

        out_feat = self.final_conv(out_feat)
        return out_feat

    def fuse_model(self):
        module_list = [
            self.pre_conv,
            self.query_conv,
            self.key_conv,
            self.value_conv,
            self.final_conv,
        ]
        for m in module_list:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class GaNetNeck(nn.Module):
    """
    Neck for ganet.

    Args:
        fpn_module: fpn module for ganet neck.
        attn_in_channels: channels of attention layer input.
        attn_out_channels: channels of attention layer input.
        attn_ratios: ratios of channel in hidden layer of each attention layer.
        pos_shape: Shape of pos embed.
        num_feats: The number of feat map.
    """

    def __init__(
        self,
        fpn_module: nn.Module,
        attn_in_channels: List[int],
        attn_out_channels: List[int],
        attn_ratios: List[int],
        pos_shape: Tuple[int, int, int] = (1, 10, 25),
        num_feats: int = 3,
    ):
        super(GaNetNeck, self).__init__()

        assert (
            len(attn_in_channels) == len(attn_out_channels) == len(attn_ratios)
        ), (
            "the length of attn_in_channels, attn_out_channels and",
            "attn_ratios must equal.",
        )
        self.fpn_module = fpn_module
        self.pos_shape = pos_shape
        self.num_feats = num_feats
        self.attn_module = self._build_attn_module(
            attn_in_channels,
            attn_out_channels,
            attn_ratios,
            self.pos_shape,
        )

        self._init_weights()

    def _init_weights(self):

        if hasattr(self.fpn_module, "_init_weights"):
            self.fpn_module._init_weights()

        for m in self.attn_module:
            if hasattr(m, "_init_weights"):
                m._init_weights()

    def _build_attn_module(
        self, attn_in_channels, attn_out_channels, attn_ratios, pos_shape
    ):
        attn_layers = []
        for attn_in_channel, attn_out_channel, attn_ratio in zip(
            attn_in_channels, attn_out_channels, attn_ratios
        ):
            attn_layers.append(
                AttentionLayer(
                    attn_in_channel,
                    attn_out_channel,
                    attn_ratio,
                    pos_shape,
                )
            )

        return nn.ModuleList(attn_layers)

    def forward(self, feats):

        feats = feats[-self.num_feats :]
        for m in self.attn_module:
            feats[-1] = m(feats[-1])

        feats = self.fpn_module(feats)

        return feats

    def fuse_model(self):

        if hasattr(self.fpn_module, "fuse_model"):
            self.fpn_module.fuse_model()
        for m in self.attn_module:
            m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        for m in self.attn_module:
            m.set_qconfig()

        if hasattr(self.fpn_module, "set_qconfig"):
            self.fpn_module.set_qconfig()
        else:
            self.fpn_module.qconfig = qconfig_manager.get_default_qat_qconfig()
