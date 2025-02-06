from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY
from .bbox_coder import limit_period


@OBJECT_REGISTRY.register
class FCOS3DLoss(nn.Module):
    """Loss for FCOS3D.

    Args:
        num_classes: Number of categories excluding the background
            category.
        pred_attrs: Whether to predict attributes.
            Defaults to False.
        num_attrs: The number of attributes to be predicted.
            Default: 9.
        group_reg_dims: The dimension of each regression
            target group. Default: (2, 1, 3, 1, 2).
        pred_velo: Whether to predict velocity.
            Defaults to False.
        use_direction_classifier: Whether to add a direction classifier.
        dir_offset: Parameter used in direction
            classification. Defaults to 0.
        dir_limit_offset: Parameter used in direction
            classification. Defaults to 0.
        diff_rad_by_sin: Whether to change the difference
            into sin difference for box regression loss. Defaults to True.
        loss_cls: Config of classification loss.
        loss_bbox: Config of localization loss.
        loss_dir: Config of direction classifier loss.
        loss_attr: Config of attribute classifier loss,
            which is only active when `pred_attrs=True`.
        loss_centerness: Config of centerness loss.
        train_cfg: Training config of anchor head.
    """

    def __init__(
        self,
        num_classes: int,
        pred_attrs: False,
        num_attrs: int,
        group_reg_dims: Tuple[int],
        pred_velo: bool,
        use_direction_classifier: bool,
        dir_offset: float,
        dir_limit_offset: float,
        diff_rad_by_sin: bool,
        loss_cls: Dict,
        loss_bbox: Dict,
        loss_dir: Dict,
        loss_attr: Dict,
        loss_centerness: Dict,
        train_cfg: Dict,
    ):
        super(FCOS3DLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.group_reg_dims = group_reg_dims
        self.pred_velo = pred_velo
        self.use_direction_classifier = use_direction_classifier
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.diff_rad_by_sin = diff_rad_by_sin
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_dir = loss_dir
        self.loss_attr = loss_attr
        self.loss_centerness = loss_centerness
        self.train_cfg = train_cfg

    @staticmethod
    def get_direction_target(
        reg_targets,
        dir_offset=0,
        dir_limit_offset=0.0,
        num_bins=2,
        one_hot=True,
    ):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int, optional): Direction offset. Default to 0.
            dir_limit_offset (float, optional): Offset to set the direction
                range. Default to 0.0.
            num_bins (int, optional): Number of bins to divide 2*PI.
                Default to 2.
            one_hot (bool, optional): Whether to encode as one hot.
                Default to True.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(
            rot_gt - dir_offset, dir_limit_offset, 2 * np.pi
        )
        dir_cls_targets = torch.floor(
            offset_rot / (2 * np.pi / num_bins)
        ).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device
            )
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7]
        )
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(
            boxes2[..., 6:7]
        )
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1
        )
        return boxes1, boxes2

    def forward(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        labels_3d,
        bbox_targets_3d,
        centerness_targets,
        attr_targets,
    ):
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, dir_cls_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = (
            ((flatten_labels_3d >= 0) & (flatten_labels_3d < bg_class_ind))
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d.to(torch.int64),
            avg_factor=num_pos + num_imgs,
        )  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            flatten_attr_targets = torch.cat(attr_targets)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims)
            )
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape
            )

            code_weight = self.train_cfg.get("code_weight", None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight
                )

            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False,
                )

            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d
                )

            loss_offset = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum(),
            )
            loss_depth = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum(),
            )
            loss_size = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum(),
            )
            loss_rotsin = self.loss_bbox(
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum(),
            )
            loss_velo = None
            if self.pred_velo:
                loss_velo = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum(),
                )

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets
            )
            # direction classification loss
            loss_dir = None
            # TODO: add more check for use_direction_classifier
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum(),
                )

            # attribute classification loss
            loss_attr = None
            if self.pred_attrs:
                loss_attr = self.loss_attr(
                    pos_attr_preds,
                    pos_attr_targets,
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum(),
                )

        else:
            # need absolute due to possible negative delta x/y
            loss_offset = pos_bbox_preds[:, :2].sum()
            loss_depth = pos_bbox_preds[:, 2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            loss_velo = None
            if self.pred_velo:
                loss_velo = pos_bbox_preds[:, 7:9].sum()
            loss_centerness = pos_centerness.sum()
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()
            loss_attr = None
            if self.pred_attrs:
                loss_attr = pos_attr_preds.sum()

        loss_dict = {
            "loss_cls": loss_cls,
            "loss_offset": loss_offset,
            "loss_depth": loss_depth,
            "loss_size": loss_size,
            "loss_rotsin": loss_rotsin,
            "loss_centerness": loss_centerness,
        }

        if loss_velo is not None:
            loss_dict["loss_velo"] = loss_velo

        if loss_dir is not None:
            loss_dict["loss_dir"] = loss_dir

        if loss_attr is not None:
            loss_dict["loss_attr"] = loss_attr

        return loss_dict
