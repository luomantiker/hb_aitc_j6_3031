from mmdet3d.ops.bev_pool_v2.bev_pool import QuickCumsumCuda, bev_pool_v2
from onnxruntime_extensions import PyCustomOpDef, onnx_op
import torch

__all__ = ["BevPoolV2"]


@onnx_op(
    op_type="BevPoolV2",
    inputs=[
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_int32,
        PyCustomOpDef.dt_int32,
        PyCustomOpDef.dt_int32,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int32,
        PyCustomOpDef.dt_int32,
    ],
    outputs=[PyCustomOpDef.dt_float],
)
def bev_pooling_v2_func(
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    bev_feat_shape,
    interval_starts,
    interval_lengths,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = torch.from_numpy(depth).to(device)
    feat = torch.from_numpy(feat).to(device)
    ranks_depth = torch.from_numpy(ranks_depth).int().to(device)
    ranks_feat = torch.from_numpy(ranks_feat).int().to(device)
    ranks_bev = torch.from_numpy(ranks_bev).int().to(device)
    interval_starts = torch.from_numpy(interval_starts).int().to(device)
    interval_lengths = torch.from_numpy(interval_lengths).int().to(device)
    bev_feat_shape = bev_feat_shape.tolist()
    bev_feat = bev_pool_v2(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    )
    return bev_feat.cpu().numpy()


class BevPoolV2Function(QuickCumsumCuda):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
        ctx,  # noqa: ARG004
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    ):
        return bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )

    @staticmethod
    def symbolic(
        g,
        depth,
        feat,
        ranks_depths,
        ranks_feats,
        ranks_bevs,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    ):
        assert len(bev_feat_shape) == 5, "bev_feat_shape must be 5D"
        output_shape = list(bev_feat_shape)
        # 在公版算子中, bev_pool_v2 输出末尾会执行一次 permute(0, 4, 1, 2, 3),
        # 将输出的布局从 (B, H, W, Z, C) 转换为 (B, C, H, W, Z), 此处与该行为保持一致.
        # 详见 https://horizonrobotics.feishu.cn/wiki/GJ20wMaNeicoYYkV3fIcpZyBnWc
        output_shape[1], output_shape[2], output_shape[3], output_shape[4] = (
            output_shape[4],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        )
        bev_feat_shape_tensor = g.op("Constant", value_t=torch.tensor(bev_feat_shape))
        return g.op(
            "ai.onnx.contrib::BevPoolV2",
            depth,
            feat,
            ranks_depths,
            ranks_feats,
            ranks_bevs,
            bev_feat_shape_tensor,
            interval_starts,
            interval_lengths,
        ).setType(depth.type().with_dtype(torch.float32).with_sizes(output_shape))


class BevPoolV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_pool_v2 = BevPoolV2Function.apply

    def forward(
        self,
        depth,
        feat,
        ranks_depths,
        ranks_feats,
        ranks_bevs,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    ):
        return self.bev_pool_v2(
            depth,
            feat,
            ranks_depths,
            ranks_feats,
            ranks_bevs,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )
