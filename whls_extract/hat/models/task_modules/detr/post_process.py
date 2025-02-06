import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.core.box_utils import box_center_to_corner
from hat.registry import OBJECT_REGISTRY

__all__ = ["DetrPostProcess"]


@OBJECT_REGISTRY.register
class DetrPostProcess(nn.Module):
    """Convert model's output into the format expected by evaluation."""

    @torch.no_grad()
    def forward(self, outs, targets):
        outputs_class, outputs_coord = outs
        outputs_coord = outputs_coord.sigmoid()
        out_logits, out_bbox = outputs_class[-1], outputs_coord[-1]

        assert len(out_logits) == len(targets["ori_img"])

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_center_to_corner(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        pred_boxes = []
        ori_imgs = targets["ori_img"]
        for ori_img, boxes_per_image in zip(ori_imgs, boxes):
            h, w, _ = ori_img.shape
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=boxes_per_image.device
            )
            pred_boxes_per_img = boxes_per_image * image_size_xyxy
            pred_boxes.append(pred_boxes_per_img)

        pred_boxes = torch.stack(pred_boxes)
        det_results = torch.cat(
            [pred_boxes, scores.unsqueeze(2), labels.unsqueeze(2)], dim=-1
        )

        results = {}
        results["pred_bboxes"] = det_results
        results["img_name"] = targets["img_name"]
        results["img_id"] = targets["img_id"]

        return results
