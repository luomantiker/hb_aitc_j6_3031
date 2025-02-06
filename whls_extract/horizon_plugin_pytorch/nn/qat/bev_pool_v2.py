from torch import Tensor

from ..bev_pool_v2 import BevPoolV2 as FloatBevPoolV2
from ..bev_pool_v2 import bev_pool_v2
from .qat_meta import QATModuleMeta


class BevPoolV2(FloatBevPoolV2, metaclass=QATModuleMeta, input_num=2):
    def forward(
        self,
        depth: Tensor,
        feat: Tensor,
        ranks_depth: Tensor,
        ranks_feat: Tensor,
        ranks_bev: Tensor,
        interval_starts: Tensor,
        interval_lengths: Tensor,
        bev_feat_shape=None,
    ):
        if self.activation_pre_process is not None:
            depth = self.activation_pre_process[0](depth)
            feat = self.activation_pre_process[1](feat)

        if bev_feat_shape is None:
            bev_feat_shape = self.bev_feat_shape

        output = bev_pool_v2(
            depth.as_subclass(Tensor),
            feat.as_subclass(Tensor),
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            bev_feat_shape,
        )

        if self.activation_post_process is None:
            return output
        else:
            return self.activation_post_process(output)
