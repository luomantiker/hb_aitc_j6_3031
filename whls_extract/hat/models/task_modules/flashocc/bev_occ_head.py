# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["BEVOCCHead2D"]

nusc_class_frequencies = np.array(
    [
        944004,
        1897170,
        152386,
        2391677,
        16957802,
        724139,
        189027,
        2074468,
        413451,
        2384460,
        5916653,
        175883646,
        4275424,
        51393615,
        61411620,
        105975596,
        116424404,
        1892500630,
    ]
)


@OBJECT_REGISTRY.register
class BEVOCCHead2D(nn.Module):
    """BEVOCCHead2D module.

    Args:
        in_dim: In channels for occ_head.
        out_dim: The out channels for final conv.
        share_conv_channels: Channels for share conv.
        use_predicter: Whether to use two-layer linear layers for prediction.
        loss_occ: Loss function module.
        use_upsample: whether to upsample the features to the same dimension
                      as gt, 200 * 200.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        Dz: int = 16,
        # use_mask: bool = True,
        num_classes: bool = 18,
        use_predicter: bool = True,
        loss_occ: nn.Module = None,
        use_upsample: bool = False,
    ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule2d(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        # self.use_mask = use_mask
        self.num_classes = num_classes
        self.use_upsample = use_upsample

        self.loss_occ = loss_occ

    def forward(self, img_feats: Tensor):
        """Forward mould.

        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:
            occ_pred:tensor
        """
        occ_pred = self.final_conv(img_feats)
        if self.use_upsample is True:
            occ_pred = F.interpolate(
                occ_pred, size=(200, 200), mode="bilinear", align_corners=False
            )
        # (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = occ_pred.permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        return occ_pred
