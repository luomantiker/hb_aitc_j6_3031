# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import numpy
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = ["Accuracy", "TopKAccuracy", "AccuracySeg"]


@OBJECT_REGISTRY.register
class Accuracy(EvalMetric):
    """Computes accuracy classification score.

    Args:
        axis (int): The axis that represents classes
        name (str):  Name of this metric instance for display.
    """

    def __init__(
        self,
        axis=1,
        name="accuracy",
    ):
        self.axis = axis
        super().__init__(name)

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, self.axis)

            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            pred_label_num = (label >= 0).sum()
            self.num_inst += pred_label_num


@OBJECT_REGISTRY.register
class AccuracySeg(EvalMetric):
    """Computes seg accuracy."""

    def __init__(
        self,
        name="accuracy",
        axis=1,
    ):
        super().__init__(name)
        self.axis = axis

    def update(self, output):
        labels = output["gt_seg"]
        preds = output["pred_seg"]

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            # different with Accuracy
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, self.axis)

            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            pred_label_num = (label >= 0).sum()
            self.num_inst += pred_label_num


@OBJECT_REGISTRY.register
class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    `TopKAccuracy` differs from Accuracy in that it considers the prediction
    to be ``True`` as long as the ground truth label is in the top K
    predicated labels.

    If `top_k` = ``1``, then `TopKAccuracy` is identical to `Accuracy`.

    Args:
        top_k (int): Whether targets are in top k predictions.
        name (str):  Name of this metric instance for display.
    """

    def __init__(self, top_k, name="top_k_accuracy"):
        super().__init__(name)
        self.top_k = top_k
        assert self.top_k > 1, "Please use Accuracy if top_k=1"
        self.name += "_%d" % self.top_k

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):
            assert len(pred_label.shape) == 2, "Predictions should be 2 dims"
            # Using argpartition here instead of argsort is safe because
            # we do not care about the order of top k elements. It is
            # much faster, which is important since that computation is
            # single-threaded due to Python GIL.
            pred_label = numpy.argpartition(
                pred_label.detach().cpu().float().numpy(),
                -self.top_k,
            )
            label = label.detach().cpu().numpy().astype("int32")

            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    num_correct = (
                        pred_label[:, num_classes - 1 - j].flat == label.flat
                    ).sum()
                    self.sum_metric += num_correct
            self.num_inst += num_samples


@OBJECT_REGISTRY.register
class AccuracyAttrMultiLabel(EvalMetric):
    """Computes multi-label accuracy classification score.

    Args:
        name (str):  Name of this metric instance for display.
        attr_type_name (str): Name of the specific type for display.
        attr_type_list (List): List of all types.
        attr_type_numcls (List): Number of categories for each type.
        ignore_idx (List): The index of the category to be ignored.
    """

    def __init__(
        self,
        name: str = "accuracy",
        attr_type_name: str = "",
        attr_type_list: List = None,
        attr_type_numcls: List = None,
        ignore_idx: List = None,
    ):
        name = name + "_" + attr_type_name
        super().__init__(name)
        self.attr_type_name = attr_type_name
        self.attr_type_list = attr_type_list
        self.attr_type_numcls = attr_type_numcls
        self.ignore_idx = ignore_idx

    def cal_acc(self, idx, label, pred_label, type_name):

        if idx == 0:
            t_label = label[0 : self.attr_type_numcls[idx]]
            t_pred_label = pred_label[0 : self.attr_type_numcls[idx]]
        else:
            t_label = label[
                sum(self.attr_type_numcls[:idx]) : sum(
                    self.attr_type_numcls[: idx + 1]
                )
            ]
            t_pred_label = pred_label[
                sum(self.attr_type_numcls[:idx]) : sum(
                    self.attr_type_numcls[: idx + 1]
                )
            ]

        if (
            not (
                type_name != "ignore"
                and type_name != "occlusion"
                and "confidence" not in type_name
                and self.ignore_idx is not None
                and label[self.ignore_idx].sum() > 0
            )
            and not t_label.sum() == 0
        ):

            t_label = torch.argmax(t_label)
            t_pred_label = torch.argmax(t_pred_label)

            num_correct = (t_label == t_pred_label).sum()
            self.sum_metric += num_correct
            pred_label_num = (t_label >= 0).sum()
            self.num_inst += pred_label_num

    def update(self, labels, preds):

        for label, pred_label in zip(labels, preds):
            if self.attr_type_name != "all_attribute":
                idx = self.attr_type_list.index(self.attr_type_name)
                self.cal_acc(idx, label, pred_label, self.attr_type_name)
            else:
                for idx, type_name in enumerate(self.attr_type_list):
                    self.cal_acc(idx, label, pred_label, type_name)
