from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hat.models.task_modules.centerpoint.target import clip_sigmoid
from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterPointLoss"]


@OBJECT_REGISTRY.register
class CenterPointLoss(nn.Module):
    """CenterPoint loss module.

    Args:
        loss_cls: Classification loss module. Default: None.
        loss_bbox: Regression loss module. Default: None.
        with_velocity: Whether velocity information is included.
        code_weights: Weights for the regression loss. Default: None.
    """

    def __init__(
        self,
        loss_cls: Optional[nn.Module] = None,
        loss_bbox: Optional[nn.Module] = None,
        with_velocity: bool = False,
        code_weights: Optional[list] = None,
    ):
        super(CenterPointLoss, self).__init__()

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.with_velocity = with_velocity
        self.code_weights = code_weights

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat: Feature map with the shape of [B, H*W, dim].
            ind: Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask: Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, dim].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def forward(
        self,
        heatmaps: List[torch.Tensor],
        anno_boxes: List[torch.Tensor],
        inds: List[torch.Tensor],
        masks: List[torch.Tensor],
        preds_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute CenterPoint loss.

        Args:
            heatmaps: List of heatmap tensors.
            anno_boxes: List of ground truth annotation boxes.
            inds: List of indexes indicating the position
                of the valid boxes.
            masks: List of masks indicating which boxes are valid.
            preds_dicts: List of predicted tensors.

        Returns:
            Dict: A dictionary containing loss components.
        """
        loss_dict = dict()  # noqa C408
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict["heatmap"] = clip_sigmoid(preds_dict["heatmap"])
            loss_heatmap = self.loss_cls(
                preds_dict["heatmap"],
                heatmaps[task_id],
            )
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.with_velocity:
                preds_dict["anno_box"] = torch.cat(
                    (
                        preds_dict["reg"],
                        preds_dict["height"],
                        preds_dict["dim"],
                        preds_dict["rot"],
                        preds_dict["vel"],
                    ),
                    dim=1,
                )
            else:
                preds_dict["anno_box"] = torch.cat(
                    (
                        preds_dict["reg"],
                        preds_dict["height"],
                        preds_dict["dim"],
                        preds_dict["rot"],
                    ),
                    dim=1,
                )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.code_weights
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)
            )
            loss_dict[f"task{task_id}.loss_heatmap"] = loss_heatmap
            loss_dict[f"task{task_id}.loss_bbox"] = loss_bbox

        return loss_dict
