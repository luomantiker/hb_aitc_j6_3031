# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import os
from typing import Sequence

import numpy as np

from hat.metrics.map_utils.mean_ap import eval_map, format_res_gt_by_classes
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from hat.utils.package_helper import require_packages
from .metric import EvalMetric

logger = logging.getLogger(__name__)

__all__ = ["NuscenesMapMetric"]


MAPCLASSES = ("divider",)


@OBJECT_REGISTRY.register
class NuscenesMapMetric(EvalMetric):
    """Evaluation Nuscenes Detection.

    Args:
        name: Name of this metric instance for display.
        eval_use_same_gt_sample_num_flag: Whether to use same gt sample number
            for evaluation.
        save_prefix: Path to save result.
        fixed_ptsnum_per_line: Number of fixed points per line.
        pc_range: Range of point cloud.
        classes: Classes for evaluation.
        metric: Metric used for evaluation.
        map_ann_file: Map annotation file.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self,
        name: str = "NuscenesMapMetric",
        eval_use_same_gt_sample_num_flag=False,
        save_prefix: str = "./WORKSPACE/results",
        fixed_ptsnum_per_line: int = -1,
        pc_range: Sequence[float] = None,
        classes: Sequence[str] = None,
        metric: str = "chamfer",
        map_ann_file: str = None,
    ):
        super(NuscenesMapMetric, self).__init__(name)

        self.save_prefix = save_prefix
        self.pc_range = pc_range

        self.res_path = os.path.join(self.save_prefix, "results_nusc.json")

        self.pred_annos = []
        self.gt_annos = []
        self.ret = ["MAP", 0.0]

        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = (
            eval_use_same_gt_sample_num_flag
        )
        if classes is None:
            self.classes = MAPCLASSES
        else:
            self.classes = classes
        self.NUM_MAPCLASSES = len(self.classes)

        self.metric = metric
        self.map_ann_file = map_ann_file

    def compute(self):
        pass

    def output_to_vecs(self, detection):
        box3d = detection["boxes_3d"].numpy()
        scores = detection["scores_3d"].numpy()
        labels = detection["labels_3d"].numpy()
        pts = detection["pts_3d"].numpy()

        vec_list = []
        for i in range(box3d.shape[0]):
            vec = {
                "bbox": box3d[i],  # xyxy
                "label": labels[i],
                "score": scores[i],
                "pts": pts[i],
            }
            vec_list.append(vec)
        return vec_list

    def update(
        self,
        batch_data,
        pred_results,
    ):
        mapped_class_names = self.classes

        if "seq_meta" in batch_data:
            tokens = batch_data["seq_meta"][0]["sample_token"]
            gt_vecs = batch_data["seq_meta"][0]["gt_instances"]
            gt_labels = batch_data["seq_meta"][0]["gt_labels_map"]
        else:
            tokens = batch_data["sample_token"]
            gt_vecs = batch_data["gt_instances"]
            gt_labels = batch_data["gt_labels_map"]

        for sample_token, res, vecs_gt, labels_gt in zip(
            tokens, pred_results, gt_vecs, gt_labels
        ):
            pred_anno = {}
            gt_anno = {}

            vecs = self.output_to_vecs(res)

            pred_anno["sample_token"] = sample_token
            gt_anno["sample_token"] = sample_token

            pred_vec_list = []
            for _, vec in enumerate(vecs):
                name = mapped_class_names[vec["label"]]
                anno = {
                    "pts": vec["pts"],
                    "pts_num": len(vec["pts"]),
                    "cls_name": name,
                    "type": vec["label"],
                    "confidence_level": vec["score"],
                }
                pred_vec_list.append(anno)
            pred_anno["vectors"] = pred_vec_list
            self.pred_annos.append(pred_anno)

            gt_labels = labels_gt.cpu().numpy()
            gt_vecs = vecs_gt.instance_list
            gt_vec_list = []
            for _, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                name = mapped_class_names[gt_label]
                anno = {
                    "pts": np.array(list(gt_vec.coords)),
                    "pts_num": len(list(gt_vec.coords)),
                    "cls_name": name,
                    "type": gt_label,
                }
                gt_vec_list.append(anno)
            gt_anno["vectors"] = gt_vec_list
            self.gt_annos.append(gt_anno)

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output_pred = [None for _ in range(global_world_size)]
        global_output_gt = [None for _ in range(global_world_size)]
        all_gather_object(global_output_pred, self.pred_annos)
        all_gather_object(global_output_gt, self.gt_annos)
        return global_output_pred, global_output_gt

    def reset(self):
        self.pred_annos = []
        self.gt_annos = []
        self.ret = ["MAP", 0.0]

    def get(self):
        if len(self.pred_annos) != 0:
            preds, gts = self._gather()
            self.pred_annos = preds[0]
            self.gt_annos = gts[0]
            tokens = {pred["sample_token"] for pred in preds[0]}
            for pred, gt in zip(preds[1:], gts[1:]):
                for pred_dict, gt_dict in zip(pred, gt):
                    assert (
                        pred_dict["sample_token"] == gt_dict["sample_token"]
                    ), "sample_token not match"
                    if pred_dict["sample_token"] not in tokens:
                        tokens.add(pred_dict["sample_token"])
                        self.pred_annos.append(pred_dict)
                        self.gt_annos.append(gt_dict)

            self._get()
        return self.ret[0], self.ret[1]

    def _dump(self):
        modality = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        }
        nusc_sub = {
            "meta": modality,
            "results": self.pred_annos,
        }
        logger.info(f"Results writes to {self.res_path}")
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        with open(self.res_path, "w") as fs:
            json.dump(nusc_sub, fs)

    @rank_zero_only
    def _get(self):
        # self._dump()
        logger.info(
            f"The length of self.pred_annos is: {len(self.pred_annos)}, "
            f"The length of self.gt_annos is: {len(self.gt_annos)}"
        )
        gen_results = self.pred_annos
        annotations = self.gt_annos
        if self.map_ann_file is not None:
            with open(self.map_ann_file, "r") as ann_f:
                gt_anns = json.load(ann_f)
            annotations = gt_anns["GTs"]
            logger.info(f"Use gts from {self.map_ann_file}")

        logger.info(f"Results writes to {self.res_path}")
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        cls_gens, cls_gts = format_res_gt_by_classes(
            self.res_path,
            gen_results,
            annotations,
            cls_names=self.classes,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,  # noqa E501
            pc_range=self.pc_range,
        )

        metric = self.metric
        allowed_metrics = ["chamfer", "iou"]
        assert metric in allowed_metrics, f"metric {metric} is not supported"

        logger.info("-*" * 10 + f"use metric:{metric}" + "-*" * 10)
        if metric == "chamfer":
            thresholds = [0.5, 1.0, 1.5]
        elif metric == "iou":
            thresholds = np.linspace(
                0.5,
                0.95,
                int(np.round((0.95 - 0.5) / 0.05)) + 1,
                endpoint=True,
            )
        cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

        for i, thr in enumerate(thresholds):
            print("-*" * 10 + f"threshhold:{thr}" + "-*" * 10)
            mAP, cls_ap = eval_map(
                gen_results,
                annotations,
                cls_gens,
                cls_gts,
                threshold=thr,
                cls_names=self.classes,
                logger=logger,
                num_pred_pts_per_instance=self.fixed_num,
                pc_range=self.pc_range,
                metric=metric,
            )
            for j in range(self.NUM_MAPCLASSES):
                cls_aps[i, j] = cls_ap[j]["ap"]

        log_info = ""
        log_info += "mAP: {:.4f}\n".format(cls_aps.mean(0).mean())
        log_info += "NuscMap_{}/mAP: {}\n".format(
            metric, cls_aps.mean(0).mean()
        )
        for i, name in enumerate(self.classes):
            print("{}: {}".format(name, cls_aps.mean(0)[i]))
            log_info += "NuscMap_{}/{}_AP: {}".format(
                metric, name, cls_aps.mean(0)[i]
            )
            log_info += "\n"

        for i, name in enumerate(self.classes):
            for j, thr in enumerate(thresholds):
                if metric == "chamfer":
                    log_info += "NuscMap_{}/{}_AP_thr_{}: {}".format(
                        metric, name, thr, cls_aps[j][i]
                    )
                    log_info += "\n"
                elif metric == "iou":
                    if thr == 0.5 or thr == 0.75:
                        log_info += "NuscMap_{}/{}_AP_thr_{}: {}".format(
                            metric, name, thr, cls_aps[j][i]
                        )
                        log_info += "\n"

        logger.info(log_info)
        self.ret[1] = cls_aps.mean(0).mean()
