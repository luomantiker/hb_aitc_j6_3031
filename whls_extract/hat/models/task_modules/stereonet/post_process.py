from typing import List, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hat.registry import OBJECT_REGISTRY

__all__ = ["StereoNetPostProcess", "StereoNetPostProcessPlus"]


@OBJECT_REGISTRY.register
class StereoNetPostProcess(nn.Module):
    """
    A basic post process for StereoNet.

    Args:
        maxdisp: The max value of disparity.
    """

    def __init__(self, maxdisp: int = 192):
        super(StereoNetPostProcess, self).__init__()
        self.maxdisp = maxdisp

    def forward(
        self,
        pred_disps: List[Tensor],
        gt_disps: List[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """Perform the forward pass of the model.

        Args:
            pred_disps: The model outputs.
            gt_disps: The gt disparitys.

        Returns:
            pred_disps: The prediction disparitys.
        """

        if self.training:
            assert gt_disps is not None
            disp_tmp = F.interpolate(
                pred_disps[-2],
                size=pred_disps[-1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_disps[-1] = F.relu(disp_tmp + pred_disps[-1])
            for i in range(len(pred_disps)):
                pred_disp_w = pred_disps[i].size()[-1]
                gt_disp_w = gt_disps.size()[-1]
                pred_disps[i] = pred_disps[i] * self.maxdisp
                if pred_disp_w != gt_disp_w:
                    pred_disps[i] = F.interpolate(
                        pred_disps[i], size=gt_disps.shape[1:], mode="bilinear"
                    )
                pred_disps[i] = pred_disps[i].squeeze(1)

            return pred_disps

        else:
            disp_tmp = F.interpolate(
                pred_disps[-2],
                size=pred_disps[-1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pred_disps[-1] = F.relu(disp_tmp + pred_disps[-1])
            pred_disps[-1] = pred_disps[-1].squeeze(1) * self.maxdisp

            return pred_disps[-1]


@OBJECT_REGISTRY.register
class StereoNetPostProcessPlus(nn.Module):
    """
    An advanced post process for StereoNet.

    Args:
        maxdisp: The max value of disparity.
        low_max_stride: The max stride of lowest disparity.
    """

    def __init__(self, maxdisp: int = 192, low_max_stride: int = 8):
        super(StereoNetPostProcessPlus, self).__init__()
        self.maxdisp = maxdisp
        self.low_max_stride = low_max_stride

    def forward(
        self,
        modelouts: List[Tensor],
        gt_disps: List[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """Perform the forward pass of the model.

        Args:
            modelouts: The model outputs.
            gt_disps: The gt disparitys.

        """

        if len(modelouts) == 3:
            disp_low = modelouts[0]
        else:
            disp_low = None

        disp_low_unfold = modelouts[-2]
        spg = modelouts[-1]

        disp_1 = F.interpolate(
            disp_low_unfold, scale_factor=self.low_max_stride, mode="nearest"
        )

        disp_1 = (spg * disp_1).sum(1)
        disp_1 = disp_1.squeeze(1) * self.maxdisp

        if self.training:
            disp_low = F.interpolate(
                disp_low, size=gt_disps.shape[1:], mode="bilinear"
            )

            disp_low = disp_low.squeeze(1) * self.maxdisp
            return [disp_low, disp_1]
        else:
            return disp_1.squeeze(1)
