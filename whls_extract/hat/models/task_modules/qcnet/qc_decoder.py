import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from .fourier_embedding import FourierConvEmbedding, FourierEmbedding
from .rattention import RAttentionLayer
from .utils import weight_init

__all__ = ["QCNetOEDecoder"]


@OBJECT_REGISTRY.register
class QCNetOEDecoder(nn.Module):
    """
    QCNetOEDecoder for decoding predictions in the OE version of QCNet.

    Args:
        input_dim: Dimension of the input data.
        hidden_dim: Dimension of the hidden layers.
        output_dim: Dimension of the output data.
        output_head: Flag indicating if an output head is used.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps.
        num_modes: Number of prediction modes.
        num_recurrent_steps: Number of recurrent steps.
        num_t2m_steps: Number of time steps for the decoder's cross attention
                along the time axis .
        num_freq_bands: Number of frequency bands.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        dropout: Dropout rate.
        split_rec_modules: Flag indicating whether to split recurrent modules.
            (recommend for QAT stage)
        deploy: Flag indicating if the model is being deployed.
        reuse_agent_rembs: Flag to reuse agent rembs for the decoder.
        quant_infer_cold_start: Indicates whether to use cold start
            for streaming inference.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 2,
        output_head: bool = False,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        num_modes: int = 6,
        num_recurrent_steps: int = 3,
        num_t2m_steps: Optional[int] = 30,
        num_freq_bands: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.1,
        split_rec_modules: bool = True,
        reuse_agent_rembs: bool = False,
        deploy: bool = False,
        quant_infer_cold_start: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.HT = num_historical_steps
        self.FT = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = (
            num_t2m_steps
            if num_t2m_steps is not None
            else num_historical_steps
        )
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.deploy = deploy
        self.reuse_agent_rembs = reuse_agent_rembs
        self.quant_infer_cold_start = quant_infer_cold_start

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierConvEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pl2m_emb = FourierConvEmbedding(
            input_dim=input_dim_r_pl2m,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2m_emb = FourierConvEmbedding(
            input_dim=input_dim_r_a2m,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.y_emb = FourierEmbedding(
            input_dim=output_dim + output_head,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.traj_emb = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        """
        self.a2m_propose_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )"""
        self.m2m_propose_attn_layer = RAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            has_pos_emb=False,
        )

        self.t2m_refine_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.m2m_refine_attn_layer = RAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            has_pos_emb=False,
        )
        self.to_loc_propose_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(
                hidden_dim,
                num_future_steps * output_dim // num_recurrent_steps,
            ),
        )
        self.to_scale_propose_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(
                hidden_dim,
                num_future_steps * output_dim // num_recurrent_steps,
            ),
        )
        self.to_loc_refine_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_future_steps * output_dim),
        )
        self.to_scale_refine_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_future_steps * output_dim),
        )
        self.to_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )

        pt = self.HT - self.num_t2m_steps
        ST = self.num_t2m_steps
        self.t = torch.arange(pt, self.HT).reshape([1, 1, ST])
        x = torch.arange(pt, self.HT).reshape([1, ST]).repeat(ST, 1)
        y = torch.arange(pt, self.HT).reshape([ST, 1]).repeat(1, ST)
        self.diff_t = y - x

        self.loc_cumsum_conv = nn.Conv2d(
            self.num_future_steps,
            self.num_future_steps,
            kernel_size=1,
            bias=False,
        )
        self.scale_cumsum_conv = nn.Conv2d(
            self.num_future_steps,
            self.num_future_steps,
            kernel_size=1,
            bias=False,
        )
        for param in self.loc_cumsum_conv.parameters():
            param.requires_grad = False
        for param in self.scale_cumsum_conv.parameters():
            param.requires_grad = False

        self.apply(weight_init)
        self.loc_cumsum_conv.weight.data = torch.tril(
            torch.ones(self.num_future_steps, self.num_future_steps)
        ).reshape([self.num_future_steps, self.num_future_steps, 1, 1])
        self.scale_cumsum_conv.weight.data = torch.tril(
            torch.ones(self.num_future_steps, self.num_future_steps)
        ).reshape([self.num_future_steps, self.num_future_steps, 1, 1])

        self.traj_emb_h0_quant = QuantStub()
        self.mode_emb_quant = QuantStub()
        self.dequant = DeQuantStub()
        self.propose_quant = QuantStub()
        self.quant_r_t_cur = QuantStub()
        self.quant_x_a = QuantStub()
        if split_rec_modules:
            self.new_module()

        self.r_t2m_quant = nn.ModuleList([QuantStub() for i in range(4)])
        self.r_pl2m_quant = nn.ModuleList([QuantStub() for i in range(3)])
        self.r_a2m_quant = nn.ModuleList([QuantStub() for i in range(3)])

        self.locs_propose_pos_cat = FloatFunctional()

    def new_module(self):
        self.pl2m_propose_attn_layers_t = nn.ModuleList(
            [
                copy.deepcopy(self.pl2m_propose_attn_layers)
                for i in range(self.num_recurrent_steps)
            ]
        )
        # del self.a2m_propose_attn_layers
        self.t2m_propose_attn_layers_t = nn.ModuleList(
            [
                copy.deepcopy(self.t2m_propose_attn_layers)
                for i in range(self.num_recurrent_steps)
            ]
        )
        # del self.t2m_propose_attn_layers
        self.m2m_propose_attn_layer_t = nn.ModuleList(
            [
                copy.deepcopy(self.m2m_propose_attn_layer)
                for i in range(self.num_recurrent_steps)
            ]
        )
        # del self.m2m_propose_attn_layer
        self.to_loc_propose_pos_t = nn.ModuleList(
            [
                copy.deepcopy(self.to_loc_propose_pos)
                for i in range(self.num_recurrent_steps)
            ]
        )
        # del self.to_loc_propose_pos
        self.to_scale_propose_pos_t = nn.ModuleList(
            [
                copy.deepcopy(self.to_scale_propose_pos)
                for i in range(self.num_recurrent_steps)
            ]
        )
        # del self.to_scale_propose_pos

    def build_r_emb(self, data, scene_enc):
        if not self.reuse_agent_rembs:
            r_t2m = data["decoder"]["r_t2m"]
            r_t2m = [self.r_t2m_quant[i](a) for i, a in enumerate(r_t2m)]
            r_t2m = self.r_t2m_emb(
                continuous_inputs=r_t2m, categorical_embs=None
            )  # [B, D, A, qt]
            r_t2m = r_t2m.permute(0, 2, 3, 1)  # [B, A, qt, D]

            r_pl2m = data["decoder"]["r_pl2m"]
            r_pl2m = [self.r_pl2m_quant[i](a) for i, a in enumerate(r_pl2m)]
            r_pl2m = self.r_pl2m_emb(
                continuous_inputs=r_pl2m, categorical_embs=None
            )
            r_pl2m = r_pl2m.permute(0, 2, 3, 1)  # [B, A, pl, D]

            r_a2m = data["decoder"]["r_a2m"]
            r_a2m = [self.r_a2m_quant[i](a) for i, a in enumerate(r_a2m)]
            r_a2m = self.r_a2m_emb(
                continuous_inputs=r_a2m, categorical_embs=None
            )
            r_a2m = r_a2m.permute(0, 2, 3, 1)
        else:
            r_t2m = scene_enc["r_t_cur"]  # [B, A, time_span, D]
            r_pl2m = scene_enc["r_pl2a_cur"]  # [B, A, pl, D]
            r_a2m = scene_enc["r_a2a_cur"]  # [B, A, A, D]

        return {"r_t2m": r_t2m, "r_pl2m": r_pl2m, "r_a2m": r_a2m}

    def forward(self, data: dict, scene_enc: dict):
        B, A = data["decoder"]["mask_dst"].shape[:2]
        M = self.num_modes
        HT = self.HT
        QT = self.num_t2m_steps
        pt = HT - QT

        r_embs = self.build_r_emb(data, scene_enc)
        r_t2m = r_embs["r_t2m"]  # [B, A, qt, D]
        r_pl2m = r_embs["r_pl2m"]  # [B, A, pl, D]
        r_a2m = r_embs["r_a2m"]  # [B, A, A, D]

        mask_dst = data["decoder"]["mask_dst"]

        mask_t2m = mask_dst.unsqueeze(2).repeat(
            1, 1, 1, self.HT - pt
        )  # [B, A, 1, T]
        mask_pl2m = mask_dst.repeat(1, 1, r_pl2m.shape[2])

        x_t = scene_enc["x_a"]  # [B, A, HT, D]
        x_a = scene_enc["x_a"][:, :, -1]

        x_pl = scene_enc["x_pl"]
        D = x_t.shape[-1]

        mask_a2m = data["decoder"]["mask_a2m"]  # [B, A, A, D]

        m = (
            self.mode_emb_quant(self.mode_emb.weight)
            .reshape(1, 1, M, D)
            .repeat(B, A, 1, 1)
        )  # [B,A,M,D]
        r_pl2m6 = r_pl2m.repeat(1, M, 1, 1)
        mask_pl2m6 = mask_pl2m.repeat(1, M, 1)

        r_a2m6 = r_a2m.repeat(1, M, 1, 1)
        mask_a2m6 = mask_a2m.repeat(1, M, 1)
        r_t2m6 = r_t2m.unsqueeze(2).repeat(1, 1, M, 1, 1)  # [B, A, M, qt, D]
        mask_t2m6 = mask_t2m.repeat(1, 1, M, 1)

        locs_propose_pos = [None] * self.num_recurrent_steps
        scales_propose_pos = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                # [B, A, HT, D],[B, A, M, D],  [B, A, M, HT, D]
                m = self.t2m_propose_attn_layers_t[t][i](
                    (x_t, m), r_t2m6, extra_dim=True, mask=mask_t2m6
                )
                # [B, A, M, D]
                m = m.transpose(1, 2)
                m = m.reshape([B, M * A, -1])
                # [B, pl, D] [B, M*A, D], [B, M*A, pl, D]
                m = self.pl2m_propose_attn_layers_t[t][i](
                    (x_pl, m), r_pl2m6, mask=mask_pl2m6
                )
                m = m.reshape(B, M, A, D).transpose(1, 2)
            # [B, A, M, D]
            m = self.m2m_propose_attn_layer_t[t](
                m, None, extra_dim=True
            )  # [B, A, M, D]
            locs_propose_pos[t] = self.to_loc_propose_pos_t[t](
                m
            )  # [B, A, M, 24]
            if not self.deploy:
                scales_propose_pos[t] = self.to_scale_propose_pos_t[t](
                    self.dequant(m)
                )  # [B, A, M, 24]

        loc_propose_pos = self.locs_propose_pos_cat.cat(
            locs_propose_pos, dim=-1
        ).view(B * A, self.num_modes, self.FT, self.output_dim)
        loc_propose_pos = (
            self.loc_cumsum_conv(loc_propose_pos.permute(0, 2, 1, 3))
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if not self.deploy:
            scale_propose_pos = (
                F.elu(
                    torch.cat(scales_propose_pos, dim=-1).view(
                        B * A, self.num_modes, self.FT, self.output_dim
                    ),
                    alpha=1.0,
                )
                + 1.0
            )
            scale_propose_pos = (
                self.scale_cumsum_conv(scale_propose_pos.permute(0, 2, 1, 3))
                .permute(0, 2, 1, 3)
                .contiguous()
                + 0.1
            )
        # [B, A, M, FT, 2]  [B, A, M, FT, 2]
        m = self.y_emb(loc_propose_pos.detach())  # [B*A, 6, FT, D]
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim)
        traj_emb_h0 = self.traj_emb_h0_quant(self.traj_emb_h0)
        h0 = traj_emb_h0.unsqueeze(1).repeat(1, m.shape[0], 1)
        # [ B*A*M, FT, D] | [B*A*M, 1, D]
        m = self.traj_emb(m, h0)[1]
        m = m.reshape(B, A, M, D)  # [B, A, M, D]

        for i in range(self.num_layers):
            # [B, A, HT, D],[B, A, M, D],  [B, A, M, HT, D]
            m = self.t2m_refine_attn_layers[i](
                (x_t, m), r_t2m6, extra_dim=True, mask=mask_t2m6
            )
            # [B, A, M, D]
            m = m.transpose(1, 2)
            m = m.reshape([B, M * A, -1])
            # [B, pl, D] [B, M*A, D], [B, M*A, pl, D]
            m = self.pl2m_refine_attn_layers[i](
                (x_pl, m), r_pl2m6, mask=mask_pl2m6
            )
            # [B, A,  D], [B, M*A, D],  [B, M*A, A, D]
            m = self.a2m_refine_attn_layers[i](
                (x_a, m), r_a2m6, mask=mask_a2m6
            )
            m = m.reshape(B, M, A, D).transpose(1, 2)
        m = self.m2m_refine_attn_layer(m, None, extra_dim=1)  # [B, A, M, D]
        loc_refine_pos = self.to_loc_refine_pos(m).view(
            B * A, self.num_modes, self.num_future_steps, self.output_dim
        )
        loc_refine_pos = self.dequant(loc_refine_pos) + self.dequant(
            loc_propose_pos.detach()
        )
        pi = self.to_pi(m).squeeze(-1).reshape(B * A, M)

        if not self.deploy or self.training:
            scale_refine_pos = (
                F.elu(
                    self.to_scale_refine_pos(self.dequant(m)).view(
                        B * A,
                        self.num_modes,
                        self.num_future_steps,
                        self.output_dim,
                    ),
                    alpha=1.0,
                )
                + 1.0
                + 0.1
            )
            return {
                "loc_propose_pos": self.dequant(loc_propose_pos),
                "loc_refine_pos": loc_refine_pos,
                "pi": self.dequant(pi),
                "scale_propose_pos": scale_propose_pos,
                "scale_refine_pos": scale_refine_pos,
            }
        else:
            return {
                "loc_refine_pos": loc_refine_pos,
                "pi": self.dequant(pi),
            }

    def set_qconfig(self):
        # Set the fix-scale to cover the actual trajectory position range
        # of (-300, 300) meters (with inputs scaled down by a factor of 10).
        self.propose_quant.qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 30 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 30 / 32768,
                "dtype": qint16,
            },
        )

        self.r_a2m_emb.set_qconfig()
        self.r_pl2m_emb.set_qconfig()
        self.r_t2m_emb.set_qconfig()
        self.y_emb.set_qconfig()
        self.to_pi[3].qconfig = qconfig_manager.get_default_out_qconfig()
        self.to_loc_refine_pos[
            3
        ].qconfig = qconfig_manager.get_default_out_qconfig()
        self.loc_cumsum_conv.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16},
            activation_calibration_qkwargs={"dtype": qint16},
        )
