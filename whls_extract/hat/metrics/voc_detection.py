# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, convert_numpy
from .metric import EvalMetric
from .utils import cat_tensor_to_numpy

logger = logging.getLogger(__name__)

__all__ = ["VOCMApMetric", "VOC07MApMetric"]


@OBJECT_REGISTRY.register
class VOCMApMetric(EvalMetric):
    """Calculate mean AP for object detection task.

    Args:
        num_classes: Num classs.
        iou_thresh: IOU overlap threshold for TP.
        class_names: If provided, will print out AP for each class.
        ignore_ioa_thresh: The IOA threshold for ignored GTs.
        score_threshs: If provided, will print recall/precision at
            each score threshold.
        max_iou_thresh: If provided, will calculate average AP at each iou
            threshold from 'iou_thresh' to 'max_iou_thresh', and the step
            is 'iou_thresh_interval'. Must be larger than iou_thresh.
        iou_thresh_interval: The step to generate a list of iou thresholds.
            Default is 0.05. You need to make sure
            ``max_iou_thresh - iou_thresh`` can be devided by this value.
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresh: Union[float, List] = 0.5,
        class_names: Optional[List[str]] = None,
        ignore_ioa_thresh: float = 0.2,
        score_threshs: Optional[List[float]] = None,
        max_iou_thresh: Optional[float] = None,
        iou_thresh_interval: float = 0.05,
        cls_idx_mapping: bool = False,
    ):
        self.num_classes = num_classes
        self.iou_thresh = _as_list(iou_thresh)
        if max_iou_thresh is not None:
            num_gap = Decimal(str(max_iou_thresh)) - Decimal(str(iou_thresh))
            rem = num_gap % Decimal(str(iou_thresh_interval))
            assert (
                max_iou_thresh > iou_thresh and rem == 0
            ), "Make sure 'max_iou_thresh - iou_thresh' is larger than 0 and "
            "can be devided by 'iou_thresh_interval'"
            self.iou_thresh = np.arange(
                iou_thresh,
                max_iou_thresh + iou_thresh_interval,
                iou_thresh_interval,
            )

        if class_names is None:
            self.num = None
            name = "MeanAP"
        else:
            assert isinstance(class_names, (list, tuple))
            assert len(class_names) == num_classes
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.num = num + 1
            name = list(class_names) + ["mAP"]
            for i, namei in enumerate(name):
                thresh_str = str(iou_thresh)
                if max_iou_thresh is not None:
                    thresh_str += ":{}:{}".format(
                        max_iou_thresh, iou_thresh_interval
                    )
                name[i] = namei + "@[{}]".format(thresh_str)

        self.score_threshs = score_threshs

        super(VOCMApMetric, self).__init__(name)

        self.reset()
        self.ignore_ioa_thresh = ignore_ioa_thresh
        self.class_names = class_names
        self.cls_idx_mapping = cls_idx_mapping

    def _init_states(self):
        self.add_state(
            "_n_pos",
            default=torch.zeros(self.num_classes),
            dist_reduce_fx="sum",
        )
        for cls in range(self.num_classes):
            for it in self.iou_thresh:
                self.add_state(
                    "_%s_%d_match" % (str(it).replace(".", "_"), cls),
                    default=[],
                    dist_reduce_fx="cat",
                )

            self.add_state(
                "_%d_score" % (cls),
                default=[],
                dist_reduce_fx="cat",
            )

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, "num", None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        super().reset()

    def compute(self):
        self.gather_metrics()
        if self.num is None:
            if self.num_inst == 0:
                return float("nan")
            else:
                return self.sum_metric / self.num_inst
        else:
            values = [
                x / y if y != 0 else float("nan")
                for x, y in zip(self.sum_metric, self.num_inst)
            ]
            return values

    def update(self, model_outs: Dict):
        """model_outs is a dict, the meaning of it's key is as following.

        pred_bboxes(List): Each element of pred_bboxes is the predict result
            of an image. It's shape is (N, 6), where 6 means
            (x1, y1, x2, y2, label, score).
        gt_bboxes(List): Each element of gt_bboxes is the bboxes' coordinates
            of an image. It's shape is (N, 4), where 4 means (x1, y1, x2, y2).
        gt_classes(List): Each element of gt_classes is the bboxes' classes
            of an image. It's shape is (N).
        gt_difficult(List): Each element of gt_difficult is the bboxes'
            difficult flag of an image. It's shape is (N).
        """

        gt_bboxes = model_outs["gt_bboxes"]
        gt_labels = model_outs["gt_classes"]
        gt_real_labels = model_outs.get("gt_labels")
        gt_difficults = model_outs["gt_difficult"]
        ig_bboxes = model_outs.get("ig_bboxes", None)
        outputs = model_outs["pred_bboxes"]
        pred_bboxes = [pred[:, :4] for pred in outputs]
        pred_scores = [pred[:, 4] for pred in outputs]
        pred_labels = [pred[:, 5] for pred in outputs]

        if gt_difficults is None:
            gt_difficults = [None for _ in gt_labels]

        for batch_id, (
            pred_bbox,
            pred_label,
            pred_score,
            gt_bbox,
            gt_label,
            gt_difficult,
        ) in enumerate(
            zip(
                *[
                    convert_numpy(x)
                    for x in [
                        pred_bboxes,
                        pred_labels,
                        pred_scores,
                        gt_bboxes,
                        gt_labels,
                        gt_difficults,
                    ]
                ]
            )
        ):
            if gt_bbox.shape[0] == 0:
                continue
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]

            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])

            possible_choices = [pred_label, gt_label]
            if gt_real_labels is not None:
                possible_choices.append(
                    convert_numpy(gt_real_labels[batch_id])
                )
            for ln in np.unique(np.concatenate(possible_choices).astype(int)):
                self._update_for_class(
                    ln,
                    gt_label,
                    gt_bbox,
                    convert_numpy(ig_bboxes[batch_id]) if ig_bboxes else None,
                    gt_real_labels[batch_id] if gt_real_labels else None,
                    gt_difficult,
                    pred_label,
                    pred_bbox,
                    pred_score,
                )

    def _update_for_class(
        self,
        class_id,
        gt_label,
        gt_bbox,
        ig_bbox,
        gt_real_label,
        gt_difficult,
        pred_label,
        pred_bbox,
        pred_score,
    ):
        if class_id < 0:  # The label is ignored.
            return
        if self.cls_idx_mapping:
            class_id = 0
        if (  # The predicted label is not annotated in the image.
            gt_real_label is not None and class_id not in gt_real_label
        ):
            return
        ln_score = []
        score_attr_name = "_%d_score" % (class_id)

        pred_mask_l = pred_label == class_id
        pred_bbox_l = pred_bbox[pred_mask_l]
        pred_score_l = pred_score[pred_mask_l]
        # sort by score
        order = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order]
        pred_score_l = pred_score_l[order]

        gt_mask_l = np.logical_or(gt_label == class_id, gt_label < 0)
        gt_bbox_l = gt_bbox[gt_mask_l]
        gt_difficult_l = gt_difficult[gt_mask_l]

        self._n_pos[class_id] += np.logical_not(gt_difficult_l).sum()
        device = self._n_pos.device
        ln_score.extend(pred_score_l)
        pre_score = getattr(self, score_attr_name)
        cur_score = pre_score + [torch.tensor(ln_score, device=device)]
        setattr(self, score_attr_name, cur_score)

        if len(pred_bbox_l) == 0:  # no prediction matched to ln.
            return
        if len(gt_bbox_l) == 0:
            for it in self.iou_thresh:
                ln_match = (0,) * pred_bbox_l.shape[0]
                match_attr_name = "_%s_%d_match" % (
                    str(it).replace(".", "_"),
                    class_id,
                )
                pre_match = getattr(self, match_attr_name)
                cur_match = pre_match + [torch.tensor(ln_match, device=device)]
                setattr(self, match_attr_name, cur_match)
            return

        for it in self.iou_thresh:
            iou = self._cal_iou_fn(pred_bbox_l.copy(), gt_bbox_l.copy())

            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < it] = -1

            if ig_bbox is not None and len(ig_bbox) > 0:
                ioa = self._cal_iou_fn(
                    pred_bbox_l.copy(), ig_bbox.copy(), "ioa"
                )
                ig_index = ioa.argmax(axis=1)
                ig_index[iou.max(axis=1) >= self.ignore_ioa_thresh] = 1
            else:
                ig_index = np.zeros_like(gt_index)

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            ln_match = []
            for ig_status, gt_idx in zip(ig_index, gt_index):
                is_fp = False
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        ln_match.append(-1)
                    else:
                        if not selec[gt_idx]:
                            ln_match.append(1)
                        else:
                            is_fp = True
                    selec[gt_idx] = True
                else:
                    is_fp = True
                if is_fp:
                    if ig_status > 0:
                        ln_match.append(-2)
                    else:
                        ln_match.append(0)

            match_attr_name = "_%s_%d_match" % (
                str(it).replace(".", "_"),
                class_id,
            )
            pre_match = getattr(self, match_attr_name)
            cur_match = pre_match + [torch.tensor(ln_match, device=device)]
            setattr(self, match_attr_name, cur_match)

    def gather_metrics(self):
        """Update num_inst and sum_metric."""
        aps = []
        self.rec_prec_on_threshs = {}
        for lp in range(self.num_classes):
            ap = []
            for it in self.iou_thresh:
                recall, prec = self._recall_prec(lp, 0, it)
                api = self._average_precision(recall, prec)
                ap.append(api)
            ap = np.nanmean(ap)
            aps.append(ap)
            if self.num is not None and lp < (self.num - 1):
                self.sum_metric[lp] = ap
                self.num_inst[lp] = 1
            if self.score_threshs:
                cn = self.class_names[lp]
                self.rec_prec_on_threshs[cn] = {}
                for st in self.score_threshs:
                    recall_st, prec_st = [], []
                    for it in self.iou_thresh:
                        recall_sti, prec_sti = self._recall_prec(lp, st, it)
                        recall_st.append(recall_sti[-1])
                        prec_st.append(prec_sti[-1])
                    recall_st = np.nanmean(recall_st)
                    prec_st = np.nanmean(prec_st)
                    self.rec_prec_on_threshs[cn][st] = "{:.3f}/{:.3f}".format(
                        recall_st, prec_st
                    )

        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)
        if self.score_threshs:
            self._print_recall_precision_on_thresholds()

    def _recall_prec(self, class_i, score_thresh, iou_thresh):
        """Get recall and precision from internal records."""
        prec = None
        rec = None

        score_lk = getattr(self, "_%d_score" % (class_i))
        match_lk = getattr(
            self, "_%s_%d_match" % (str(iou_thresh).replace(".", "_"), class_i)
        )

        score_l = cat_tensor_to_numpy(score_lk)
        match_l = cat_tensor_to_numpy(match_lk)
        match_l = match_l.astype(np.int32)

        valid = score_l >= score_thresh
        score_l = score_l[valid]
        match_l = match_l[valid]

        if len(score_l) == 0:
            return [0], [0]

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec is nan.
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = tp / (fp + tp)
        # If n_pos is 0, rec is None.
        n_pos = self._n_pos[class_i].cpu().numpy()
        if n_pos > 0:
            rec = tp / n_pos

        return rec, prec

    def _average_precision(self, rec, prec):
        """Calculate average precision.

        Args:
            rec (numpy.array): cumulated recall
            prec (numpy.array): cumulated precision
        Returns:
            ap as float
        """

        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], np.nan_to_num(prec), [0.0]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _cal_iou_fn(self, bbox_a, bbox_b, itype="iou"):
        # VOC evaluation follows integer typed bounding boxes.
        bbox_a[:, 2:] += 1
        bbox_b[:, 2:] += 1

        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2], axis=1)
        area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2], axis=1)
        if itype == "iou":
            result = area_i / (area_a[:, None] + area_b - area_i)
        elif itype == "ioa":
            result = area_i / area_a[:, None]
        elif itype == "iob":
            result = area_i / area_b
        else:
            raise NotImplementedError

        return result

    def _print_recall_precision_on_thresholds(self):
        logger.info(
            "{}Begin Recall/Precision Result{}".format("=" * 50, "=" * 50)
        )
        df = pd.DataFrame(self.rec_prec_on_threshs)
        df.from_dict(self.rec_prec_on_threshs)
        logger.info("\n{}".format(df))
        logger.info(
            "{}End Recall/Precision Result{}".format("=" * 50, "=" * 50)
        )


@OBJECT_REGISTRY.register
class VOC07MApMetric(VOCMApMetric):
    """Mean average precision metric for PASCAL V0C 07 dataset.

    Args:
        num_classes: Num classs.
        iou_thresh: IOU overlap threshold for TP
        class_names: if provided, will print out AP for each class
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresh: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        super(VOC07MApMetric, self).__init__(
            num_classes=num_classes,
            iou_thresh=iou_thresh,
            class_names=class_names,
        )

    def _average_precision(self, rec, prec):
        """Calculate average precision, override the default one.

           special 11-point metric

        Args:
            rec (numpy.array): cumulated recall
            prec (numpy.array): cumulated precision
        Returns:
            ap as float
        """

        if rec is None or prec is None:
            return np.nan
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.0
        return ap
