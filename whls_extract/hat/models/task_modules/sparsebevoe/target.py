from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from hat.registry import OBJECT_REGISTRY

__all__ = ["SparseBEVOETarget"]


@OBJECT_REGISTRY.register
class SparseBEVOETarget(object):
    """
    Configuration and parameters for Sparse BEVOE Target generation.

    Args:
        cls_weight: Weight for classification loss. Defaults to 2.0.
        alpha: Focal loss alpha parameter. Defaults to 0.25.
        gamma: Focal loss gamma parameter. Defaults to 2.
        eps: Small epsilon value for numerical stability. Defaults to 1e-12.
        box_weight: Weight for regression loss. Defaults to 0.25.
        reg_weights: List of weights for each regression target.
                     Defaults to [1.0] * 8 + [0.0] * 2.
        cls_wise_reg_weights: List of class-wise regression weights.
                              Defaults to None.
        ignore_reg_weight: Weight for ignored regression targets.
                           Defaults to 0.0.
        ignore_cls_weight: Weight for ignored classification targets.
                           Defaults to 0.0.
        num_dn_groups: Number of groups for denoise sampling.
                       Defaults to 0.
        num_temp_dn_groups: Number of temporal groups for denosis sampling.
                            Defaults to 0.
        dn_noise_scale: Scale factor for denoise sampling .
                        Defaults to 0.5.
        max_dn_gt: Maximum number of ground truth samples for denoise sampling.
                   Defaults to 30.
        add_neg_dn: Whether to add negative denoise samples.
                    Defaults to True.
    """

    def __init__(
        self,
        cls_weight: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2,
        eps: float = 1e-12,
        box_weight: float = 0.25,
        reg_weights: List[float] = None,
        cls_wise_reg_weights: List[float] = None,
        ignore_reg_weight: float = 0.0,
        ignore_cls_weight: float = 0.0,
        num_dn_groups: int = 0,
        num_temp_dn_groups: int = 0,
        dn_noise_scale: float = 0.5,
        max_dn_gt: int = 30,
        add_neg_dn: bool = True,
    ):
        super(SparseBEVOETarget, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.ignore_reg_weight = ignore_reg_weight
        self.ignore_cls_weight = ignore_cls_weight
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.dn_metas = None

    def encode_reg_target(
        self,
        box_target: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """
        Encode regression targets into a standardized format.

        Args:
            box_target: List of tensors representing regression targets.
            device: Device to move the outputs to. Defaults to None.

        Returns:
            List of encoded regression targets.
        """

        outputs = []
        for box in box_target:
            if len(box) == 0:
                output = torch.zeros([1, 10])
            else:
                output = torch.cat(
                    [
                        box[..., 0:3],
                        box[..., 3:6].log(),
                        torch.sin(box[..., 6]).unsqueeze(-1),
                        torch.cos(box[..., 6]).unsqueeze(-1),
                        torch.where(
                            torch.isnan(box[..., 6 + 1 :]),
                            box.new_tensor(0),
                            box[..., 6 + 1 :],
                        ),
                    ],
                    dim=-1,
                )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def __call__(
        self,
        cls_pred: torch.Tensor,
        box_pred: torch.Tensor,
        cls_target: torch.Tensor,
        box_target: List[torch.Tensor],
        gt_ignore: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Compute target components based on predicted and target values.

        Args:
            cls_pred: Predicted class scores, shape (bs, num_pred, num_cls).
            box_pred: Predicted box coordinates, shape (bs, num_pred, 4).
            cls_target: Target class labels, shape (bs, num_pred).
            box_target: List of target box coordinates tensors.
            gt_ignore: Ignored ground truth indicator, shape (bs, num_pred).

        Returns:
            Tuple of :
                - output_cls_target: Classification target.
                - output_box_target: Regression target.
                - output_cls_weights: Classification loss weight.
                - output_reg_weights: Regression loss weight.
        """

        bs, num_pred, num_cls = cls_pred.shape
        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)
        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)

        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for i in range(bs):
            cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
            cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
            indices.append(
                [
                    cls_pred.new_tensor(x, dtype=torch.int64)
                    for x in linear_sum_assignment(cost)
                ]
            )

        output_cls_target = cls_target[0].new_ones([bs, num_pred]) * num_cls
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        ignore = box_pred.new_zeros([bs, num_pred])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
            if gt_ignore is not None and len(gt_ignore) != 0:
                ignore[i, pred_idx] = gt_ignore[i][target_idx].to(
                    dtype=ignore.dtype
                )

        output_reg_weights = torch.where(
            ignore[..., None] == 0,
            output_reg_weights,
            output_reg_weights * self.ignore_reg_weight,
        )
        output_cls_weights = ignore.new_ones(ignore.shape)
        output_cls_weights = torch.where(
            ignore == 0,
            output_cls_weights,
            output_cls_weights * self.ignore_cls_weight,
        )
        return (
            output_cls_target,
            output_box_target,
            output_cls_weights,
            output_reg_weights,
        )

    def _cls_cost(
        self, cls_pred: torch.Tensor, cls_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification cost based on predicted and target class scores.

        Args:
            cls_pred: Predicted class scores, shape (bs, num_pred, num_cls).
            cls_target: Target class labels, shape (bs, num_pred).

        Returns:
            Classification cost.
        """
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            neg_cost = (
                -(1 - cls_pred[i] + self.eps).log()
                * (1 - self.alpha)
                * cls_pred[i].pow(self.gamma)
            )
            pos_cost = (
                -(cls_pred[i] + self.eps).log()
                * self.alpha
                * (1 - cls_pred[i]).pow(self.gamma)
            )
            cost.append(
                (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                * self.cls_weight
            )
        return cost

    def _box_cost(
        self,
        box_pred: torch.Tensor,
        box_target: List[torch.Tensor],
        instance_reg_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute regression cost based on predicted and target box coordinates.

        Args:
            box_pred : Predicted box coordinates, shape (bs, num_pred, 4).
            box_target: List of target box coordinates tensors.
            instance_reg_weights: List of instance-wise regression weights.

        Returns:
            reg_cost : Regression cost.
        """
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            cost.append(
                torch.sum(
                    torch.abs(box_pred[i, :, None] - box_target[i][None])
                    * instance_reg_weights[i][None]
                    * box_pred.new_tensor(self.reg_weights),
                    dim=-1,
                )
                * self.box_weight
            )
        return cost

    def get_dn_anchors(
        self,
        cls_target: List[torch.Tensor],
        box_target: List[torch.Tensor],
        gt_instance_id: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate denosis anchor for training based on ground truth targets.

        Args:
            cls_target : List of tensors containing target class labels.
            box_target : List of tensors containing target box coordinates.
            gt_instance_id : List of tensors containing ground truth
                             instance IDs.Defaults to None.

        Returns:
            Tensor containing denosis anchor points or None
            if num_dn_groups <= 0.
        """

        if self.num_dn_groups <= 0:
            return None

        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None
        # put all non-ignore gt to front and delete the gt out of max_dn_gt
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])

        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = torch.stack(
            self.encode_reg_target(
                [
                    F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
                    for x in box_target
                ],
                device=cls_target.device,
            )
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape

        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)
        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)

            num_gt *= 2
        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3

        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]

            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )

        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None

        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        dn_reg_target: torch.Tensor,
        dn_cls_target: torch.Tensor,
        valid_mask: torch.Tensor,
        dn_id_target: torch.Tensor,
        num_noraml_anchor: torch.Tensor,
        temporal_valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Update Denosis anchors with temporal denosis anchor.

        Args:
            instance_feature: Instance features, shape (bs, num_anchor, ...)
            anchor: Anchor points, shape (bs, num_anchor, ...)
            dn_reg_target: DN regression targets,
                           shape (bs, num_anchor, ...)
            dn_cls_target: DN class targets, shape (bs, num_anchor)
            valid_mask: Valid mask indicating active DN instances,
                        shape (bs, num_anchor)
            dn_id_target: Optional DN instance IDs,
                          shape (bs, num_anchor)
            num_normal_anchor: Number of normal anchors (non-DN) in each batch.
            temporal_valid_mask: Optional temporal valid mask.

        Returns:
            Updated instance_feature, anchor, dn_reg_target, dn_cls_target,
            valid_mask, dn_id_target
        """
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]
        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)
        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]
        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for _, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature: torch.Tensor,
        dn_anchor: torch.Tensor,
        dn_cls_target: torch.Tensor,
        valid_mask: torch.Tensor,
        dn_id_target: torch.Tensor,
    ) -> None:
        """
        Cache Denosis features and anchors for future use.

        Args:
            dn_instance_feature: DN instance features,
                                 shape (bs, num_dn, ...)
            dn_anchor: DN anchor points, shape (bs, num_dn, ...)
            dn_cls_target: DN class targets, shape (bs, num_dn)
            valid_mask: Valid mask indicating active DN instances,
                        shape (bs, num_dn)
            dn_id_target: DN instance IDs, shape (bs, num_dn)
        """
        if self.num_temp_dn_groups <= 0 or dn_anchor.shape[1] == 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]

        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = {
            "dn_instance_feature": dn_instance_feature,
            "dn_anchor": dn_anchor,
            "dn_cls_target": dn_cls_target,
            "valid_mask": valid_mask,
            "dn_id_target": dn_id_target,
        }
