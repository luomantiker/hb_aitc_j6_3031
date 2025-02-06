from typing import Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from .fourier_embedding import FourierConvEmbedding
from .rattention import RAttentionLayer
from .utils import weight_init

__all__ = ["QCNetOEAgentEncoderStream"]


@OBJECT_REGISTRY.register
class QCNetOEAgentEncoderStream(nn.Module):
    """
    QCNetOEAgentEncoderStream encodes agent trajectories in a streaming manner.

    Args:
        input_dim: Dimension of the input data.
        hidden_dim: Dimension of the hidden layers.
        num_historical_steps: Number of historical time steps.
        time_span: Time span for the model.
        num_freq_bands: Number of frequency bands.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads .
        head_dim: Dimension of each attention head.
        num_t2m_steps: Number of time steps for the decoder's cross attention
            along the time axis.
        num_pl2a: Number of polygons for map polygon to agent cross attention
                (default is 32, works only when save_memory=True).
        num_a2a: Number of agents for agent to agent cross attention.
                (default is 36, works only when save_memory=True).
        dropout: Dropout rate.
        save_memory: Flag to save memory during processing.
        stream_infer: Flag for streaming inference.
        deploy: Flag to indicate deployment mode. Default is False.
        reuse_agent_rembs: Flag to reuse agent rembs for the decoder.
        quant_infer_cold_start:Indicates whether to use cold start
            for streaming inference.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_historical_steps: int = 50,
        time_span: Optional[int] = 10,
        num_freq_bands: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        head_dim: int = 16,
        num_t2m_steps: int = 30,
        num_pl2a: int = 32,
        num_a2a: int = 36,
        dropout: float = 0.1,
        save_memory: bool = False,
        stream_infer: bool = True,
        reuse_agent_rembs: bool = False,
        deploy: bool = False,
        quant_infer_cold_start: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.ht = num_historical_steps
        self.time_span = (
            time_span if time_span is not None else num_historical_steps
        )
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_t2m_steps = num_t2m_steps
        self.dropout = dropout
        self.save_memory = save_memory
        self.num_pl2a = num_pl2a
        self.num_a2a = num_a2a
        self.stream_infer = stream_infer
        self.reuse_agent_rembs = reuse_agent_rembs
        self.deploy = deploy
        self.quant_infer_cold_start = quant_infer_cold_start

        input_dim_x_a = 4
        input_dim_r_t = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(10, hidden_dim)

        self.x_a_emb = FourierConvEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierConvEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pl2a_emb = FourierConvEmbedding(
            input_dim=input_dim_r_pl2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierConvEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.t_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pl2a_attn_layers = nn.ModuleList(
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
        self.a2a_attn_layers = nn.ModuleList(
            [
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        x = torch.arange(self.ht).reshape([1, self.ht]).repeat(self.ht, 1)
        y = torch.arange(self.ht).reshape([self.ht, 1]).repeat(1, self.ht)
        self.diff_t = y - x
        self.t_mask = (y > x) & ((y - x) <= self.time_span)

        self.x_a_cat = nn.ModuleList(
            [FloatFunctional() for i in range(num_layers)]
        )
        self.mask_t_key_cat = FloatFunctional()
        self.his_cat = FloatFunctional()

        self.type_quant = QuantStub()

        self.diff_t_quant = QuantStub()
        self.quant = QuantStub()
        self.x_a_his_quant = QuantStub()
        self.dequant = DeQuantStub()

        self.x_a_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_x_a)]
        )
        self.r_pl2a_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_r_pl2a)]
        )
        self.r_t_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_r_t)]
        )
        self.r_a2a_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_r_a2a)]
        )

        self.mid_his_quant = nn.ModuleList(
            [QuantStub() for i in range(num_layers)]
        )

        self.apply(weight_init)

    def build_cur_embs(
        self,
        data: dict,
        cur: int,
        map_data: dict,
        x_a_his: torch.Tensor,
        categorical_embs: torch.Tensor,
    ):
        """Build current time step agent embedding.

        Args:
            data: Dict contains input data.
            cur: current time step.
            map_data: Output dict of map encoder.
            x_a_his: Historical time steps agent embeddings.q
            categorical_embs: Agent type categorical embeddings.
        """
        B, A = data["agent"]["valid_mask"].shape[:2]
        D = self.hidden_dim
        ST = self.time_span
        pl_N = map_data["x_pl"].shape[1]
        mask_a_cur = data["agent"]["mask_a_cur"]  # [B, A]
        x_a_cur = data["agent"]["x_a_cur"]
        r_pl2a_cur = data["agent"]["r_pl2a_cur"]
        r_t_cur = data["agent"]["r_t_cur"]
        r_a2a_cur = data["agent"]["r_a2a_cur"]
        pl_idx_cur = data["agent"].get("pl_idx_cur")
        a2a_idx_cur = data["agent"].get("a2a_idx_cur")
        A_K = r_a2a_cur[0].shape[3]

        x_a_cur = [self.x_a_quant[i](a) for i, a in enumerate(x_a_cur)]
        r_pl2a_cur = [
            self.r_pl2a_quant[i](a) for i, a in enumerate(r_pl2a_cur)
        ]
        r_t_cur = [self.r_t_quant[i](a) for i, a in enumerate(r_t_cur)]
        r_a2a_cur = [self.r_a2a_quant[i](a) for i, a in enumerate(r_a2a_cur)]

        categorical_embs = self.type_quant(categorical_embs).permute(
            0, 3, 1, 2
        )
        # [B, 4, A, 1], [B, A, 1, D]
        x_a_cur = self.x_a_emb(
            continuous_inputs=x_a_cur, categorical_embs=categorical_embs
        )
        # [B, D, A, 1]
        r_t_cur = self.r_t_emb(
            continuous_inputs=r_t_cur, categorical_embs=None
        )  # [B, D, A, 6]

        x_a_cur_emb = x_a_cur.permute(0, 2, 3, 1)  # [B, A, 1, D]
        x_a_cur_emb = self.dequant(x_a_cur_emb)

        if pl_idx_cur is not None:
            pl_K = self.num_pl2a
            pl_N = pl_K
            x_pl = (
                map_data["x_pl"]
                .gather(1, pl_idx_cur.reshape(B, -1, 1).repeat(1, 1, D))
                .reshape(B, A, pl_K, D)
            )

        else:
            x_pl = map_data["x_pl"]

        r_pl2a_cur = self.r_pl2a_emb(
            continuous_inputs=r_pl2a_cur, categorical_embs=None
        )
        # [B, D, A, pl_N]
        r_a2a_cur = self.r_a2a_emb(
            continuous_inputs=r_a2a_cur, categorical_embs=None
        )

        mask_a2a_cur = data["agent"]["mask_a2a_cur"]
        x_a_cur = x_a_cur.permute(0, 2, 3, 1).squeeze(2)  # [B, A, 1, D]
        r_t_cur = r_t_cur.permute(0, 2, 3, 1)  # [B, A, ST, D]

        mask_t_key = data["agent"]["mask_t_key"]  # [B, A, 2]
        r_pl2a_cur = r_pl2a_cur.permute(0, 2, 3, 1)  # [B, A, pl_N, D]
        mask_pl2a_cur = mask_a_cur[:, :, None].repeat(
            1, 1, pl_N
        )  # (B, A,  pl_N)

        r_a2a_cur = r_a2a_cur.permute(0, 2, 3, 1)  # [B, A, A, D]

        for i in range(self.num_layers):
            x_a_query = x_a_cur
            x_a_key = self.mid_his_quant[i](x_a_his[i]).reshape(B, A, -1, D)[
                :, :, -ST:
            ]
            x_a_cur = self.t_attn_layers[i](
                (x_a_key, x_a_query), r_t_cur[:, :, -ST:], mask=mask_t_key
            )  # [B, A, st, D] [B, A, D] [B, A, st, D]

            x_a_cur = self.pl2a_attn_layers[i](
                (x_pl, x_a_cur), r_pl2a_cur, mask=mask_pl2a_cur
            )
            # [B, pl_N, D], [B, A, D] , [B, A, pl_N, D]

            if a2a_idx_cur is not None:
                x_a_src = x_a_cur.gather(
                    1, a2a_idx_cur.reshape(B, A * A_K, 1).repeat(1, 1, D)
                ).reshape(B, A, A_K, D)
            else:
                x_a_src = x_a_cur

            x_a_cur = self.a2a_attn_layers[i](
                (x_a_src, x_a_cur),
                r_a2a_cur,
                mask=mask_a2a_cur.reshape(B, A, -1),
            )  # [B, A, D]
        outputs = {
            "x_a_cur": x_a_cur,
            "x_a_cur_emb": x_a_cur_emb,
        }

        if self.reuse_agent_rembs:
            r_cur_embs = {
                "r_t_cur": r_t_cur,
                "r_pl2a_cur": r_pl2a_cur,
                "r_a2a_cur": r_a2a_cur,
            }
            outputs.update(r_cur_embs)
        else:
            pass

        return outputs

    def build_his_embs(self, data: dict, map_data: dict, ht: int, qt: int):
        """Build multi-steps historical agent embeddings.

        Args:
            data: Dict contains input data.
            map_data: Output dict of map encoder.
            ht: Historical time steps which indeed indicates current time step
                index, and this function build embeddings before ht steps(:ht).
            qt: Time steps that we need to compute for agents.
        """
        B, A = data["agent"]["valid_mask"].shape[:2]
        D = self.hidden_dim
        st = self.time_span
        bt = ht - qt
        pl_N = map_data["x_pl"].shape[1]
        mask_a = data["agent"]["valid_mask"][:, :, bt:ht].reshape(B, A, qt)

        type_embed = self.type_a_emb(data["agent"]["agent_type"].long())
        type_embed_q = self.type_quant(type_embed)  # [B, A, 1]
        categorical_embs = type_embed_q.repeat(1, 1, st + qt, 1)
        categorical_embs = categorical_embs.permute(0, 3, 1, 2)

        x_a = data["agent"]["x_a"]
        r_pl2a = data["agent"]["r_pl2a"]
        r_t = data["agent"]["r_t"]  # [B*A, 1, num_t2m, time_span]*4
        r_a2a = data["agent"]["r_a2a"]
        pl_idx = data["agent"].get("pl_idx", None)
        a2a_idx = data["agent"].get("a2a_idx", None)
        A_K = r_a2a[0].shape[3]

        x_a = [self.x_a_quant[i](a) for i, a in enumerate(x_a)]
        r_pl2a = [self.r_pl2a_quant[i](a) for i, a in enumerate(r_pl2a)]
        r_t = [self.r_t_quant[i](a) for i, a in enumerate(r_t)]
        r_a2a = [self.r_a2a_quant[i](a) for i, a in enumerate(r_a2a)]

        x_a = self.x_a_emb(
            continuous_inputs=x_a, categorical_embs=categorical_embs
        )

        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        if pl_idx is not None:
            pl_K = self.num_pl2a
            pl_N = pl_K
            x_pl = (
                map_data["x_pl"]
                .gather(1, pl_idx.reshape(B, -1, 1).repeat(1, 1, D))
                .reshape(B, A * qt, pl_K, D)
            )
            pl_expand = True

        else:
            x_pl = map_data["x_pl"]
            pl_expand = False

        r_pl2a = self.r_pl2a_emb(
            continuous_inputs=r_pl2a, categorical_embs=None
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        mask_a2a = data["agent"]["valid_mask_a2a"][
            :, bt:ht, :, :A_K
        ]  # [B, 5, A, A_K]
        x_a = x_a.permute(0, 2, 3, 1).reshape(B * A, st + qt, -1)
        # [B*A, D, qt, st] --> [B*A,qt, st, D]
        r_t = r_t.permute(0, 2, 3, 1)  # [B*A, D, HT, HT] ->[B*A, HT, HT, D])
        mask_t_key = (
            self.mask_t_key_cat.cat(
                [
                    data["agent"]["valid_mask"][
                        :, :, bt - i : bt + qt - i
                    ].unsqueeze(3)
                    for i in range(st, 0, -1)
                ],
                dim=3,
            )
            .contiguous()
            .reshape(B * A, qt, st)
        )  # [B, A, QT, ST]

        r_pl2a = r_pl2a.permute(
            0, 2, 3, 1
        )  # [B*A, D, HT, pl_N] -> [B*A, HT, pl_N, D]
        mask_pl2a = mask_a[:, :, :, None].repeat(
            1, 1, 1, pl_N
        )  # .view(B, A*QT, pl_N)  # (B, A, QT, pl_N)

        r_a2a = r_a2a.permute(0, 2, 3, 1)  # [B*HT, A, A, D]
        x_a_his_list = []
        for i in range(self.num_layers):
            if i >= 1:
                qt = qt - st
                r_t = r_t[:, st * i :]
                mask_t_key = mask_t_key[:, st * i :]
            x_a_query = x_a[:, st:]

            x_a = self.dequant(x_a)
            x_a = self.mid_his_quant[i](x_a)

            x_a_his_list.append(
                x_a[:, -st:].reshape(B, A, -1, D)
            )  # [B, A, ST, D]
            x_a_key = self.x_a_cat[i].cat(
                [
                    x_a[:, st - i : st + qt - i].unsqueeze(2)
                    for i in range(st, 0, -1)
                ],
                dim=2,
            )
            x_a = self.t_attn_layers[i](
                (x_a_key, x_a_query), r_t, mask=mask_t_key
            )

            x_a = x_a.reshape(B, A, qt, -1).reshape(B, A * qt, -1)

            if i == 0:
                r_pl2a1 = r_pl2a.reshape(B, A * qt, pl_N, -1)
                mask_pl2a_i = mask_pl2a.view(B, A * qt, pl_N)
            else:
                r_pl2a1 = r_pl2a[:, st:].reshape(B, A * qt, pl_N, -1)
                mask_pl2a_i = (
                    mask_pl2a[:, :, st:].contiguous().view(B, A * qt, pl_N)
                )
                if pl_expand:
                    x_pl = x_pl.reshape(B, A, -1, pl_N, D)[:, :, st:].reshape(
                        B, A * qt, pl_N, D
                    )
            x_a = self.pl2a_attn_layers[i](
                (x_pl, x_a), r_pl2a1, mask=mask_pl2a_i
            )
            # -> [B, A*HT, D]
            x_a = (
                x_a.reshape(B, A, qt, -1)
                .permute(0, 2, 1, 3)
                .reshape([B * qt, A, -1])
            )

            if i >= 1:
                if B == 1:
                    r_a2a = r_a2a[st:]
                    mask_a2a = mask_a2a[:, st * i :]
                else:
                    r_a2a = r_a2a.reshape(B, -1, A, A_K, D)[:, st:].reshape(
                        B * qt, A, A_K, D
                    )
                    mask_a2a = mask_a2a[:, st * i :]

            if a2a_idx is not None:
                x_a_src = x_a.gather(
                    1,
                    a2a_idx[:, st * i :]
                    .reshape(B * qt, A * A_K, 1)
                    .repeat(1, 1, D),
                ).reshape(B * qt, A, A_K, D)
            else:
                x_a_src = x_a

            x_a = self.a2a_attn_layers[i](
                (x_a_src, x_a), r_a2a, mask=mask_a2a.reshape(B * qt, A, -1)
            )  # [B*HT, A, D]
            x_a = (
                x_a.reshape(B, qt, A, -1)
                .permute(0, 2, 1, 3)
                .reshape(B * A, qt, -1)
            )
        x_a = x_a.reshape(B, A, qt, -1)
        outputs = {
            "x_a": x_a,
            "x_a_mid_emb": x_a_his_list,
            "agent_type_embs": type_embed,
        }
        if self.reuse_agent_rembs and not self.stream_infer:
            r_cur_embs = {
                "r_t_cur": r_t[:, -1].reshape(
                    B, A, -1, D
                ),  # [B, A, time_span, D]
                "r_pl2a_cur": r_pl2a[:, -1].reshape(
                    B, A, -1, D
                ),  # [B, A, pl, D]
                "r_a2a_cur": r_a2a.reshape(B, qt, A, A_K, D)[:, -1].reshape(
                    B, A, -1, D
                ),
            }  # [B, A, A, D]
            outputs.update(r_cur_embs)
        return outputs

    def forward(self, data, map_data):
        if self.quant_infer_cold_start:
            x_a_his = data["agent"]["x_a_his"]  # [B, A, 5, D]
            x_a_mid_emb = data["agent"][
                "x_a_mid_emb"
            ]  # [[B, A, 2, D], [[B, A, 2, D]]
            if "agent_type_embs" in data["agent"]:
                agent_type_embs = data["agent"]["agent_type_embs"]
            else:
                agent_type_embs = self.type_a_emb(
                    data["agent"]["agent_type"].long()
                )  # [B, A, 2, D]

            cur_embs = self.build_cur_embs(
                data, self.ht, map_data, x_a_mid_emb, agent_type_embs
            )
            x_a_cur = cur_embs["x_a_cur"]
            x_a_cur = x_a_cur.unsqueeze(2)
            x_a_his = self.x_a_his_quant(x_a_his)
            outputs = {
                "x_a": self.his_cat.cat([x_a_his, x_a_cur], dim=2),
                "x_a_cur": x_a_cur,
                "x_a_cur_emb": cur_embs["x_a_cur_emb"],
            }
            if self.reuse_agent_rembs:
                outputs["r_t_cur"] = cur_embs["r_t_cur"]
                outputs["r_pl2a_cur"] = cur_embs["r_pl2a_cur"]
                outputs["r_a2a_cur"] = cur_embs["r_a2a_cur"]
            return outputs
        else:
            if not self.stream_infer and not self.deploy:
                QT = self.num_t2m_steps + self.time_span * (
                    self.num_layers - 1
                )

                out = self.build_his_embs(data, map_data, self.ht, QT)
                return out
            elif self.deploy and not self.stream_infer:
                # export his model(hot start)
                ht = data["agent"]["x_a"][0].shape[-1]
                qt = data["agent"]["r_pl2a"][0].shape[-2]
                out = self.build_his_embs(data, map_data, ht, qt)
                return out
            else:
                if "x_a_his" not in data["agent"]:
                    # predict and hot start
                    if "r_pl2a" in data["agent"]:
                        qt = data["agent"]["r_pl2a"][0].shape[-2]
                    else:
                        qt = (
                            self.num_t2m_steps
                            + self.time_span * (self.num_layers - 1)
                            - 1
                        )
                    out = self.build_his_embs(data, map_data, self.ht - 1, qt)

                    x_a_his = out["x_a"]  # [B, A, 5, D]
                    x_a_his = self.dequant(x_a_his)
                    x_a_mid_emb = out["x_a_mid_emb"]  # [B, A, ST, D]
                    agent_type_embs = out["agent_type_embs"]
                else:
                    # export cur model(hot start)
                    x_a_his = data["agent"]["x_a_his"]  # [B, A, T-1, D]
                    x_a_mid_emb = data["agent"][
                        "x_a_mid_emb"
                    ]  # [[B, A, 10, D], [[B, A, 10, D]]
                    if "agent_type_embs" in data["agent"]:
                        agent_type_embs = data["agent"]["agent_type_embs"]
                    else:
                        agent_type_embs = self.type_a_emb(
                            data["agent"]["agent_type"].long()
                        )

                cur_embs = self.build_cur_embs(
                    data, self.ht, map_data, x_a_mid_emb, agent_type_embs
                )
                x_a_cur = cur_embs["x_a_cur"]
                x_a_cur = x_a_cur.unsqueeze(2)
                x_a_his = self.x_a_his_quant(x_a_his)
                outputs = {"x_a": self.his_cat.cat([x_a_his, x_a_cur], dim=2)}
                if self.reuse_agent_rembs:
                    outputs["r_t_cur"] = cur_embs["r_t_cur"]
                    outputs["r_pl2a_cur"] = cur_embs["r_pl2a_cur"]
                    outputs["r_a2a_cur"] = cur_embs["r_a2a_cur"]
                return outputs

    def set_qconfig(self):
        # x_a_quant[2] is the vel_norm (velocity) input.
        # Set the fixed-scale to support a maximum velocity input of 50m/s
        # (with both velocity and position scaled down by a factor of 10).
        self.x_a_quant[2].qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 5 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 5 / 32768,
                "dtype": qint16,
            },
        )
        # According to the sensitivity tool, set the fix-scale to cover the
        # range (-1, 1), otherwise the calibration error will be large.
        self.x_a_emb.freqs[2].qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 1.0 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 1.0 / 32768,
                "dtype": qint16,
            },
        )
        self.r_a2a_emb.set_qconfig()
        self.r_pl2a_emb.set_qconfig()
        self.r_t_emb.set_qconfig()
        self.x_a_emb.set_qconfig()
