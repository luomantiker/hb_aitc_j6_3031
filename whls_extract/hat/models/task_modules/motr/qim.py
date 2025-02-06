import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn.quantized as quantized
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.quantization import FixedScaleObserver, QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.registry import OBJECT_REGISTRY

qint8_fixscale_qconfig = horizon.quantization.get_default_qat_qconfig(
    dtype="qint8",
    activation_qkwargs={
        "observer": FixedScaleObserver,
        "scale": 1.0,
    },
)


@OBJECT_REGISTRY.register
class QueryInteractionModule(nn.Module):
    def __init__(self, dim_in, hidden_dim, dropout=0.0):
        super().__init__()
        self._build_layers(dim_in, hidden_dim, dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() == 4:
                nn.init.xavier_uniform_(p)

    def _build_layers(self, dim_in, hidden_dim, dropout):

        self.self_attn = MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Conv2d(dim_in, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(hidden_dim, dim_in, 1)
        self.linear_pos1 = nn.Conv2d(dim_in, hidden_dim, 1)
        self.linear_pos2 = nn.Conv2d(hidden_dim, dim_in, 1)
        self.dropout_pos1 = nn.Dropout(dropout)
        self.dropout_pos2 = nn.Dropout(dropout)
        self.norm_pos = LayerNorm2d(normalized_shape=[dim_in, 1, 1], dim=1)
        self.norm1 = LayerNorm2d(normalized_shape=[dim_in, 1, 1], dim=1)
        self.norm2 = LayerNorm2d(normalized_shape=[dim_in, 1, 1], dim=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.embed_quant = QuantStub()
        self.query_pos_quant = QuantStub()
        self.query_mask_quant = QuantStub(scale=1.0)
        self.dequant = DeQuantStub()
        self.add_q_out = quantized.FloatFunctional()
        self.add_tgt_tgt2 = quantized.FloatFunctional()
        self.add_tgt_tgt2_1 = quantized.FloatFunctional()
        self.add_pos_pos2 = quantized.FloatFunctional()
        self.mask_matmul = quantized.FloatFunctional()
        self.mask_add_scalar = quantized.FloatFunctional()
        self.mask_mul_scalar = quantized.FloatFunctional()

    def forward(self, query_pos_all, output_embedding, query_mask):
        bs = 1
        out_embed = self.embed_quant(output_embedding)
        query_pos_all = self.query_pos_quant(query_pos_all)
        query_mask = self.query_mask_quant(query_mask)

        query_pos = (
            query_pos_all.permute(0, 1, 3, 2)
            .contiguous()
            .view(bs, query_pos_all.shape[3], 1, query_pos_all.shape[2])
        )

        out_embed = (
            out_embed.permute(0, 1, 3, 2)
            .contiguous()
            .view(bs, out_embed.shape[3], 1, out_embed.shape[2])
        )
        tgt_mask1 = query_mask.reshape(bs, 1, 1, query_mask.shape[3])
        tgt_mask2 = query_mask.reshape(bs, 1, query_mask.shape[3], 1)

        tgt_mask = self.mask_matmul.matmul(tgt_mask2, tgt_mask1)

        tgt_mask = self.mask_mul_scalar.mul_scalar(tgt_mask, 100)
        tgt_mask = self.mask_add_scalar.add_scalar(tgt_mask, -100)
        tgt_mask = tgt_mask.squeeze(1).squeeze(0)
        q = k = self.add_q_out.add(query_pos, out_embed)
        tgt = out_embed
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
        )[0]

        tgt = self.add_tgt_tgt2.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation1(self.linear1(tgt))))

        tgt = self.add_tgt_tgt2_1.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)

        query_pos2 = self.linear_pos2(
            self.dropout_pos1(self.activation2(self.linear_pos1(tgt)))
        )

        query_pos = self.add_pos_pos2.add(
            query_pos, self.dropout_pos2(query_pos2)
        )
        query_pos = self.norm_pos(query_pos)

        output = self.dequant(query_pos)

        return output

    def set_qconfig(self):

        modules_list = [
            self.self_attn.matmul,
            self.self_attn.attn_matmul,
            self.self_attn.out_proj,
            self.linear1,
            self.linear2,
            self.linear_pos1,
            self.linear_pos2,
            self.norm1,
            self.norm2,
            self.activation1,
            self.activation2,
            self.add_q_out,
            self.add_tgt_tgt2,
            self.add_tgt_tgt2_1,
            self.add_pos_pos2,
            self.norm_pos,
            self.embed_quant,
            self.query_pos_quant,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        self.query_mask_quant.qconfig = qint8_fixscale_qconfig
        self.self_attn.attn_mask_quant.qconfig = qint8_fixscale_qconfig

    def fuse_model(self):
        pass
