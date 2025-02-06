# Copyright (c) Horizon Robotics. All rights reserved.
import torch
from horizon_plugin_pytorch.nn.functional import decode, get_top_n, nms

from hat.registry import OBJECT_REGISTRY

__all__ = ["RetinaNetPostProcess"]


# TODO(kongtao.hu, 0.1): Modify the class name, it should be universal
@OBJECT_REGISTRY.register
class RetinaNetPostProcess(torch.nn.Module):
    """The postprocess of RetinaNet.

    Args:
        score_thresh (float): Filter boxes whose score is lower than this.
        nms_thresh (float): thresh for nms.
        detections_per_img (int): Get top n boxes by score after nms.
        topk_candidates (int): Get top n boxes by score after decode.
    """

    def __init__(
        self,
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        topk_candidates: int = 1000,
    ):
        super(RetinaNetPostProcess, self).__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk_candidates = topk_candidates
        self.post_nms_top_n = detections_per_img

    @torch.no_grad()
    def forward(self, boxes, preds, image_shapes):
        num_level = len(preds)
        batch_size = len(preds[0])

        all_image_coords_list = []
        all_image_scores_list = []
        all_image_regressions_list = []
        all_image_bboxes_list = []

        for batch_id in range(batch_size):
            each_image_coords_list = []
            each_image_scores_list = []
            each_image_regressions_list = []
            each_iamge_bboxes_list = []
            for level in range(num_level):
                each_level_coords = preds[level][batch_id][0]
                each_level_scores = preds[level][batch_id][1]
                each_level_regressions = preds[level][batch_id][2]
                each_level_bboxes = boxes[level][batch_id]

                each_image_coords_list.append(each_level_coords)
                each_image_scores_list.append(each_level_scores)
                each_image_regressions_list.append(each_level_regressions)
                each_iamge_bboxes_list.append(each_level_bboxes)

            all_image_coords_list.append(each_image_coords_list)
            all_image_scores_list.append(each_image_scores_list)
            all_image_regressions_list.append(each_image_regressions_list)
            all_image_bboxes_list.append(each_iamge_bboxes_list)

        ret_boxes, ret_scores, ret_labels = [], [], []

        for batch_id in range(batch_size):
            each_image_coords_list = all_image_coords_list[batch_id]
            each_image_scores_list = all_image_scores_list[batch_id]
            each_image_regressions_list = all_image_regressions_list[batch_id]
            each_iamge_bboxes_list = all_image_bboxes_list[batch_id]

            all_level_boxes, all_level_scores, all_level_labels = [], [], []
            for (
                each_level_coords,
                each_level_scores,
                each_level_regressions,
                each_level_bboxes,
            ) in zip(
                each_image_coords_list,
                each_image_scores_list,
                each_image_regressions_list,
                each_iamge_bboxes_list,
            ):
                each_level_bboxes = each_level_bboxes.permute(1, 2, 0)
                each_level_coords = each_level_coords.long()
                each_level_bboxes = each_level_bboxes[
                    each_level_coords[:, 0], each_level_coords[:, 1], :
                ]

                anchor_num = int(each_level_bboxes.size(-1) / 4)
                num_classes = int(each_level_scores.size(-1) / anchor_num)
                each_level_scores = each_level_scores.reshape(-1, num_classes)
                each_level_scores = each_level_scores.sigmoid()
                each_level_regressions = each_level_regressions.reshape(-1, 4)
                each_level_bboxes = each_level_bboxes.reshape(-1, 4)

                (
                    each_level_bboxes,
                    each_level_scores,
                    each_level_labels,
                ) = decode(
                    each_level_bboxes,
                    each_level_regressions,
                    each_level_scores,
                    regression_scale=None,
                    background_class_idx=None,
                    clip_size=image_shapes[batch_id],
                    size_threshold=None,
                    abs_offset=False,
                )

                (
                    each_level_scores,
                    each_level_bboxes,
                    each_level_labels,
                ) = get_top_n(
                    each_level_scores,
                    [each_level_bboxes, each_level_labels],
                    self.topk_candidates,
                    None,
                )
                all_level_boxes.append(each_level_bboxes)
                all_level_scores.append(each_level_scores)
                all_level_labels.append(each_level_labels)

            pred_per_image_boxes = torch.cat(all_level_boxes, dim=0)
            pred_per_image_scores = torch.cat(all_level_scores, dim=0)
            pred_per_image_labels = torch.cat(all_level_labels, dim=0)

            (
                pred_per_image_boxes,
                pred_per_image_scores,
                pred_per_image_labels,
            ) = nms(
                pred_per_image_boxes,
                pred_per_image_scores,
                pred_per_image_labels,
                self.nms_thresh,
                self.score_thresh,
                None,
                self.post_nms_top_n,
            )

            ret_boxes.append(pred_per_image_boxes)
            ret_scores.append(pred_per_image_scores)
            ret_labels.append(pred_per_image_labels)
        predictions = []
        for (ret_box, ret_score, ret_label) in zip(
            tuple(ret_boxes), tuple(ret_scores), tuple(ret_labels)
        ):
            ret_score = ret_score.unsqueeze(-1)
            ret_label = ret_label.unsqueeze(-1)
            pred = torch.cat([ret_box, ret_score, ret_label], dim=-1)
            predictions.append(pred)
        return predictions
