import torch
import torch.nn as nn

from hat.core.box_utils import box_center_to_corner, recover_ori_bbox
from hat.registry import OBJECT_REGISTRY

__all__ = ["DeformDetrPostProcess"]


@OBJECT_REGISTRY.register
class DeformDetrPostProcess(nn.Module):
    """Convert Deform Detr's output into the format expected by evaluation.

    Args:
        select_box_nums_for_evaluation: Selected bbox nums for evaluation.
    """

    def __init__(self, select_box_nums_for_evaluation: int = 300):
        super().__init__()
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

    @torch.no_grad()
    def forward(
        self,
        batched_inputs: dict,
        box_cls: torch.Tensor,
        box_pred: torch.Tensor,
    ):
        prob = box_cls.sigmoid()
        box_pred = box_pred.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1),
            self.select_box_nums_for_evaluation,
            dim=1,
        )
        scores = topk_values
        topk_boxes = torch.div(
            topk_indexes, box_cls.shape[2], rounding_mode="floor"
        )
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(
            box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
        )
        boxes = box_center_to_corner(boxes)

        pred_boxes = recover_ori_bbox(
            boxes,
            batched_inputs["scale_factor"],
            batched_inputs["resized_shape"],
        )
        pred_boxes = torch.stack(pred_boxes)
        det_results = torch.cat(
            [pred_boxes, scores.unsqueeze(2), labels.unsqueeze(2)], dim=-1
        )

        results = {}
        results["pred_bboxes"] = det_results
        results["img_name"] = batched_inputs["img_name"]
        results["img_id"] = batched_inputs["img_id"]
        return results
