from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from hat.core.box3d_utils import bbox3d2result, points_img2cam, xywhr2xyxyr
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.core.nms.box3d_nms import box2d3d_multiclass_nms
from hat.models.task_modules.fcos.target import get_points
from hat.registry import OBJECT_REGISTRY
from hat.utils.config import Config
from hat.utils.model_helpers import fx_wrap


@OBJECT_REGISTRY.register
class FCOS3DPostProcess(nn.Module):
    """Post-process for FOCS3D.

    Args:
        num_classes: Number of categories excluding the background category.
        use_direction_classifier: Whether to add a direction classifier.
        strides: Downsample factor of each feature map.
        group_reg_dims: The dimension of each regression
            target group. Default: (2, 1, 3, 1, 2).
        pred_attrs: Whether to predict attributes.
            Defaults to False.
        num_attrs: The number of attributes to be predicted.
            Default: 9.
        attr_background_label: background label.
        bbox_coder: bbox coder class.
        bbox_code_size: Dimensions of predicted bounding boxes.
        dir_offset: Parameter used in direction
            classification. Defaults to 0.
        test_cfg: Testing config of anchor head.
        pred_bbox2d: Whether to predict 2D boxes.
            Defaults to False.
    """

    def __init__(
        self,
        num_classes: int,
        use_direction_classifier: bool,
        strides: Tuple[int],
        group_reg_dims: Tuple[int],
        pred_attrs: bool,
        num_attrs: int,
        attr_background_label: int,
        bbox_coder: Dict,
        bbox_code_size: int,
        dir_offset: float,
        test_cfg: Dict,
        pred_bbox2d: bool = False,
    ):
        super(FCOS3DPostProcess, self).__init__()
        self.cls_out_channels = num_classes
        self.use_direction_classifier = use_direction_classifier
        self.strides = strides
        self.group_reg_dims = group_reg_dims
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.attr_background_label = attr_background_label
        self.bbox_coder = bbox_coder
        self.bbox_code_size = bbox_code_size
        self.dir_offset = dir_offset
        self.test_cfg = test_cfg
        self.pred_bbox2d = pred_bbox2d

    @fx_wrap()
    def forward(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        img_metas,
        cfg=None,
        rescale=None,
    ):
        assert (
            len(cls_scores)
            == len(bbox_preds)
            == len(dir_cls_preds)
            == len(centernesses)
            == len(attr_preds)
        )

        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        result_list = []
        for img_id in range(cls_scores[0].size(0)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id]
                    .new_full([2, *cls_scores[i][img_id].shape[1:]], 0)
                    .detach()
                    for i in range(num_levels)
                ]

            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id]
                    .new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label,
                    )
                    .detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            # input_meta = img_metas[img_id]
            input_meta = {}
            input_meta["box_type_3d"] = CameraInstance3DBoxes
            for key in img_metas.keys():
                input_meta[key] = img_metas[key][img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                dir_cls_pred_list,
                attr_pred_list,
                centerness_pred_list,
                mlvl_points,
                input_meta,
                cfg,
                rescale,
            )
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        mlvl_points,
        input_meta,
        cfg,
        rescale=False,
    ):
        view = np.array(input_meta["cam2img"])
        # scale_factor = input_meta["scale_factor"]
        cfg = self.test_cfg if cfg is None else cfg
        cfg = Config(cfg_dict=cfg)
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for (
            cls_score,
            bbox_pred,
            dir_cls_pred,
            attr_pred,
            centerness,
            points,
        ) in zip(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            attr_preds,
            centernesses,
            mlvl_points,
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.cls_out_channels)
                .sigmoid()
            )
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(
                -1, sum(self.group_reg_dims)
            )
            bbox_pred = bbox_pred[:, : self.bbox_code_size]
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            # if rescale:
            #    bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # change local yaw to global yaw for 3D nms
        cam2img = mlvl_centers2d.new_zeros((4, 4))
        cam2img[: view.shape[0], : view.shape[1]] = mlvl_centers2d.new_tensor(
            view
        )
        mlvl_bboxes = self.bbox_coder.decode_yaw(
            mlvl_bboxes,
            mlvl_centers2d,
            mlvl_dir_scores,
            self.dir_offset,
            cam2img,
        )

        mlvl_bboxes_for_nms = xywhr2xyxyr(
            input_meta["box_type_3d"](
                mlvl_bboxes,
                box_dim=self.bbox_code_size,
                origin=(0.5, 0.5, 0.5),
            ).bev
        )

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box2d3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        results = box2d3d_multiclass_nms(
            mlvl_bboxes3d=mlvl_bboxes,
            mlvl_bboxes3d_for_nms=mlvl_bboxes_for_nms,
            mlvl_scores3d=mlvl_nms_scores,
            score_thr=cfg.score_thr,
            max_num=cfg.max_per_img,
            nms_thr=cfg.nms_thr,
            mlvl_dir_scores=mlvl_dir_scores,
            mlvl_attr_scores=mlvl_attr_scores,
            do_nms_bev=True,
        )
        # print(len(results))
        labels, _, _, scores, bboxes, _, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta["box_type_3d"](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)
        )
        bbox_img = bbox3d2result(bboxes, scores, labels, attrs)
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        box_center = bboxes.gravity_center
        box_dims = bboxes.dims
        box_yaw = bboxes.yaw
        box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
        box_yaw = -box_yaw
        velocity = bboxes.tensor[..., 7:9]
        if self.pred_attrs:
            ret = torch.cat(
                [
                    box_center,
                    box_dims,
                    box_yaw.view(-1, 1),
                    velocity,
                    scores.view(-1, 1),
                    labels.view(-1, 1),
                    attrs.view(-1, 1),
                ],
                dim=-1,
            )
        else:
            ret = torch.cat(
                [
                    box_center,
                    box_dims,
                    box_yaw.view(-1, 1),
                    velocity,
                    scores.view(-1, 1),
                    labels.view(-1, 1),
                ],
                dim=-1,
            )
        return {
            "ret": ret,
            "bboxes": bbox_img["boxes_3d"],
            "scores": bbox_img["scores_3d"],
            "labels": bbox_img["labels_3d"],
            "attrs": bbox_img["attrs_3d"],
        }
