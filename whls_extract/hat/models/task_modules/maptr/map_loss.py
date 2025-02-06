# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.functional import smooth_l1_loss
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["PtsDirCosLoss", "PtsL1Loss", "PtsL1Cost", "OrderedPtsL1Cost"]


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(
        pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs
    ):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weight_dir_reduce_loss(
    loss, weight=None, reduction="mean", avg_factor=None
):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_dir
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError("avg_factor should not be none for OrderedPtsL1Loss")
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # import pdb;pdb.set_trace()
            # loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum()
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def custom_weighted_dir_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    Example:
    >>> import torch
    >>> @weighted_loss
    ... def l1_loss(pred, target):
    ...     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs
    ):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(
            loss, weight, reduction, avg_factor
        )
        return loss

    return wrapper


@weighted_loss
def _smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float): The smooth threshold. Defaults is 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta
    )
    return loss


@weighted_loss
def pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """Dir cosine similiarity loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
        target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction="none")
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


@OBJECT_REGISTRY.register
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction
        )
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_dir


@OBJECT_REGISTRY.register
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        beta (float): The smooth threshold. Defaults is 0.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, beta=0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction
        )
        # import pdb;pdb.set_trace()
        if self.beta > 0:
            loss_bbox = _smooth_l1_loss(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                beta=self.beta,
            )

        else:
            loss_bbox = pts_l1_loss(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        return self.loss_weight * loss_bbox


@OBJECT_REGISTRY.register
class PtsL1Cost(object):
    """OrderedPtsL1Cost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """Call function.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.view(num_gts, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.

    Args:
        weight (int | float, optional): loss_weight
        beta (float): The smooth threshold. Defaults is 0.
    """

    def __init__(self, weight=1.0, beta=0):
        self.weight = weight
        self.beta = beta

    def __call__(self, bbox_pred, gt_bboxes):
        """Call function.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        if self.beta > 0:
            num_pred = len(bbox_pred)
            bbox_pred = bbox_pred.unsqueeze(1).repeat(1, len(gt_bboxes), 1)
            gt_bboxes = gt_bboxes.unsqueeze(0).repeat(num_pred, 1, 1)
            bbox_cost = smooth_l1_loss(
                bbox_pred, gt_bboxes, reduction="none", beta=self.beta
            ).sum(-1)
        else:
            bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class SimpleLoss(torch.nn.Module):
    def __init__(
        self,
        pos_weight,
        loss_weight,
        reduction: str = "mean",
    ):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]), reduction="none"
        )
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.dequant = DeQuantStub()

    def forward(
        self,
        ypred,
        ytgt,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ):
        if isinstance(ypred, (list, tuple)):
            assert isinstance(
                ytgt, (list, tuple)
            ), "ypred and ytgt should have the same type"
            total_loss = 0
            for pred, gt in zip(ypred, ytgt):
                loss = self.loss_fn(pred, gt)
                loss = self.dequant(loss)
                loss = weight_reduce_loss(
                    loss.float(), weight, self.reduction, avg_factor
                )
                total_loss += loss * self.loss_weight
            return total_loss

        loss = self.loss_fn(ypred, ytgt)
        loss = self.dequant(loss)
        loss = weight_reduce_loss(
            loss.float(), weight, self.reduction, avg_factor
        )
        return loss * self.loss_weight
