import logging
import warnings
from typing import List, Optional

import numba
import numpy as np

from hat.core.box_np_ops import box3d_to_bbox, box_lidar_to_camera
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy, limit_period

try:
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPerformanceWarning,
        NumbaWarning,
    )

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    warnings.filterwarnings("ignore", category=NumbaWarning)
except ImportError:
    pass


__all__ = ["Kitti3DMetricDet"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Kitti3DMetricDet(EvalMetric):
    def __init__(
        self,
        current_classes: List[str],
        compute_aos: bool = False,
        name: str = "kitti3dAPDet",
        difficultys: Optional[List] = None,  # noqa B006
    ):
        super().__init__(name)

        if difficultys is None:
            difficultys = [0, 1, 2]

        overlap_0_7 = np.array(
            [
                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
            ]
        )
        overlap_0_5 = np.array(
            [
                [0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
            ]
        )
        min_overlaps = np.stack(
            [overlap_0_7, overlap_0_5], axis=0
        )  # [2, 3, 5]
        class_to_name = {
            0: "Car",
            1: "Pedestrian",
            2: "Cyclist",
            3: "Van",
            4: "Person_sitting",
        }
        name_to_class = {v: n for n, v in class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        current_classes = current_classes_int
        min_overlaps = min_overlaps[:, :, current_classes]

        # TODO: move alpha-checking outside
        # check whether alpha is valid
        # compute_aos = False
        # for anno in dt_annos:
        #     if anno['alpha'].shape[0] != 0:
        #         if anno['alpha'][0] != -10:
        #             compute_aos = True
        #         break

        self.class_to_name = class_to_name
        self.min_overlaps = min_overlaps
        self.current_classes = current_classes
        self.compute_aos = compute_aos
        self.difficultys = difficultys
        self.mAPbbox = None
        self.mAPbev = None
        self.mAP3d = None
        self.mAPaos = None

        self.labels = []
        self.preds = []

    def reset(self):
        self.labels = []
        self.preds = []

    def update(self, preds, labels):

        preds_annos = self.convert_detection_to_kitti_annos(
            batch_detections=preds,
            batch_labels=labels,
        )
        gt_labels = self.convert_gt_annos(labels)

        self.preds += preds_annos
        self.labels += gt_labels

    def _update(self):
        assert len(self.preds) == len(
            self.labels
        ), f"{len(self.preds)} vs {len(self.labels)}"
        num_parts = min(len(self.preds), 50)

        mAPbbox, mAPbev, mAP3d, mAPaos = do_eval_v3(
            self.labels,
            self.preds,
            self.current_classes,
            self.min_overlaps,
            self.compute_aos,
            self.difficultys,
            num_parts=num_parts,
        )
        self.mAPbbox = mAPbbox
        self.mAPbev = mAPbev
        self.mAP3d = mAP3d
        self.mAPaos = mAPaos

    def get(self):
        logger.info("Calculate Kitti metric, please wait ...")
        self._update()
        names, values = ([], [])
        d3_names, d3_values = ([], [])
        for j, curcls in enumerate(self.current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            for i in range(self.min_overlaps.shape[0]):
                names.append(f"{self.class_to_name[curcls]} AP@")
                overlap_str = ", ".join(
                    [str(x) for x in self.min_overlaps[i, :, j].tolist()]
                )
                values.append(f"{overlap_str}:")

                names.append("bbox AP:")
                bbox_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAPbbox[j, 0, i],
                    self.mAPbbox[j, 1, i],
                    self.mAPbbox[j, 2, i],
                )
                values.append(bbox_ap_value)
                names.append("bev  AP:")
                bev_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAPbev[j, 0, i],
                    self.mAPbev[j, 1, i],
                    self.mAPbev[j, 2, i],
                )
                values.append(bev_ap_value)
                names.append("3d   AP:")
                d3_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAP3d[j, 0, i],
                    self.mAP3d[j, 1, i],
                    self.mAP3d[j, 2, i],
                )
                values.append(d3_ap_value)
                if self.compute_aos:
                    names.append("aos  AP:")
                    bev_ap_value = "%.2f, %.2f, %.2f" % (
                        self.mAPaos[j, 0, i],
                        self.mAPaos[j, 1, i],
                        self.mAPaos[j, 2, i],
                    )
                    values.append(bev_ap_value)

                if i == 0:
                    d3_iou = str(self.min_overlaps[i, :, j].tolist()[1])
                    d3_names.append(
                        f"{self.class_to_name[curcls]}_3D_AP@{d3_iou}_moderate"
                    )
                    d3_mAP = self.mAP3d[j, 1, i]
                    d3_values.append(d3_mAP)

        d3_names.insert(0, "mAP_3D_moderate")
        overall_mAP = sum(d3_values) / len(self.current_classes)
        d3_values.insert(0, overall_mAP)

        log_info = "\n"
        for k, v in zip(names, values):
            if isinstance(v, (int, float)):
                log_info += "%s[%.2f] \n" % (k, v)
            else:
                log_info += "%s[%s] \n" % (str(k), str(v))
        logger.info(log_info)
        return (d3_names, d3_values)

    def convert_detection_to_kitti_annos(self, batch_detections, batch_labels):
        class_names = self.current_classes

        assert "calib" in batch_labels
        batch_meta_data = batch_labels["metadata"]
        batch_rect = batch_labels["calib"]["R0_rect"]
        batch_Trv2c = batch_labels["calib"]["Tr_velo_to_cam"]
        batch_P2 = batch_labels["calib"]["P2"]

        assert (
            len(batch_detections)
            == len(batch_rect)
            == len(batch_Trv2c)
            == len(batch_P2)
            == len(batch_meta_data)
        )

        predict_annos = []
        for detection, rect, Trv2c, P2, meta_data in zip(
            batch_detections,
            batch_rect,
            batch_Trv2c,
            batch_P2,
            batch_meta_data,
        ):
            (
                box3d_lidar_preds,
                labels_preds,
                scores_preds,
            ) = detection

            box3d_lidar_preds = convert_numpy(box3d_lidar_preds)
            labels_preds = convert_numpy(labels_preds)
            image_idx = convert_numpy(meta_data["image_idx"])
            image_shape = convert_numpy(meta_data["image_shape"])
            scores_preds = convert_numpy(scores_preds)

            rect = convert_numpy(rect)
            Trv2c = convert_numpy(Trv2c)
            P2 = convert_numpy(P2)

            box3d_lidar = box3d_lidar_preds
            box3d_lidar[:, -1] = limit_period(
                box3d_lidar[:, -1],
                offset=0.5,
                period=np.pi * 2,
            )
            box3d_lidar[:, 2] -= box3d_lidar[:, 5] / 2

            # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
            # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera  # noqa
            box3d_camera = box_lidar_to_camera(box3d_lidar, rect, Trv2c)
            box_2d = box3d_to_bbox(box3d_camera, rect, Trv2c, P2)

            anno = self._get_start_result_anno()
            num_example = 0
            for j in range(box3d_lidar_preds.shape[0]):
                if (
                    box_2d[j, 0] > image_shape[1]
                    or box_2d[j, 1] > image_shape[0]
                ):
                    continue
                if box_2d[j, 2] < 0 or box_2d[j, 3] < 0:
                    continue
                box_2d[j, 2:] = np.minimum(
                    box_2d[j, 2:], image_shape[::-1][1:]
                )
                box_2d[j, :2] = np.maximum(box_2d[j, :2], [0, 0])
                anno["bbox"].append(box_2d[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(
                    -np.arctan2(
                        -box3d_lidar_preds[j, 1], box3d_lidar_preds[j, 0]
                    )
                    + box3d_camera[j, 6]
                )
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(
                    self.class_to_name[class_names[int(labels_preds[j])]]
                )
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores_preds[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                anno["image_idx"] = str(int(image_idx))
            else:
                anno = self._get_empty_result_anno()
                anno["image_idx"] = np.array([str(int(image_idx))])

            predict_annos.append(anno)

        return predict_annos

    def convert_gt_annos(self, batch_examples):
        labels = []
        for meta_data in batch_examples["metadata"]:
            annos = self._get_start_result_anno()
            for k in list(annos.keys()):
                if k in meta_data:
                    annos[k] = meta_data[k]
            annos["image_idx"] = str(int(meta_data["image_idx"]))
            labels.append(annos)
        return labels

    def _get_start_result_anno(self):
        return dict(
            {
                # 'image_idx': -10000,
                "name": [],
                "truncated": [],
                "occluded": [],
                "alpha": [],
                "bbox": [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
                "score": [],
            }
        )

    def _get_empty_result_anno(self):
        annotations = {}
        annotations.update(
            {
                "name": np.array([]),
                "truncated": np.array([]),
                "occluded": np.array([]),
                "alpha": np.array([]),
                "bbox": np.zeros([0, 4]),
                "dimensions": np.zeros([0, 3]),
                "location": np.zeros([0, 3]),
                "rotation_y": np.array([]),
                "score": np.array([]),
            }
        )
        return annotations


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = [
        "car",
        "pedestrian",
        "cyclist",
        "van",
        "person_sitting",
    ]

    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = str(gt_anno["name"][i]).lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        elif (
            current_cls_name == "Pedestrian".lower()
            and "Person_sitting".lower() == gt_name
        ):
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if (
            (gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
            or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
            or (height <= MIN_HEIGHT[difficulty])
        ):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        for i in range(num_gt):
            if (str(gt_anno["name"][i]) == "DontCare") or (
                str(gt_anno["name"][i]) == "ignore"
            ):
                dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        if str(dt_anno["name"][i]).lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def do_eval_v3(
    gt_annos,
    dt_annos,
    current_classes,
    min_overlaps,
    compute_aos=False,
    difficultys=(0, 1, 2),
    z_axis=1,
    z_center=1.0,
    num_parts=50,
):

    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        ret = eval_class_v3(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            i,
            min_overlaps,
            compute_aos,
            z_axis=z_axis,
            z_center=z_center,
            num_parts=num_parts,
        )
        metrics[types[i]] = ret

    mAP_bbox = get_mAP(metrics["bbox"]["precision"])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP(metrics["bbox"]["orientation"])
    mAP_bev = get_mAP(metrics["bev"]["precision"])
    mAP_3d = get_mAP(metrics["3d"]["precision"])

    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (
            i < (len(scores) - 1)
        ):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    metric,
    min_overlap,
    thresholds,
    compute_aos=False,
):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[
                dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]
            ]

            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num : dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def prepare_data(
    gt_annos, dt_annos, current_class, difficulty=None, clean_data=None
):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1
        )
        dt_datas = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        total_dc_num,
        total_num_valid_gt,
    )


def eval_class_v3(
    gt_annos,
    dt_annos,
    current_classes,
    difficultys,
    metric,
    min_overlaps,
    compute_aos=False,
    z_axis=1,
    z_center=1.0,
    num_parts=50,
):
    """Kitti eval.

    support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm
    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    split_parts = [i for i in split_parts if i != 0]

    rets = calculate_iou_partly(
        dt_annos, gt_annos, metric, num_parts, z_axis=z_axis, z_center=z_center
    )
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    )
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    )
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    all_thresholds = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    )
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):  # noqa E741
            rets = prepare_data(
                gt_annos,
                dt_annos,
                current_class,
                difficulty=difficulty,
                clean_data=clean_data,
            )
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                # print(thresholds)
                all_thresholds[m, l, k, : len(thresholds)] = thresholds
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx : idx + num_part], 0
                    )
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx : idx + num_part], 0
                    )
                    dc_datas_part = np.concatenate(
                        dontcares[idx : idx + num_part], 0
                    )
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx : idx + num_part], 0
                    )
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx : idx + num_part], 0
                    )
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx : idx + num_part],
                        total_dt_num[idx : idx + num_part],
                        total_dc_num[idx : idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos,
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    # recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1
                    )
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)

    ret_dict = {
        "recall": recall,  # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  # noqa
        "precision": precision,
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
    }
    return ret_dict


def get_mAP2(prec):
    sums = 0
    interval = 4
    for i in range(0, prec.shape[-1], interval):
        sums = sums + prec[..., i]
    return sums / int(prec.shape[-1] / interval) * 100


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    dc_bboxes,
    metric,
    min_overlap,
    thresh=0,
    compute_fp=False,
    compute_aos=False,
):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (
                not compute_fp
                and (overlap > min_overlap)
                and dt_score > valid_detection
            ):
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (valid_detection == NO_DETECTION)
                and ignored_det[j] == 1
            ):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (
            ignored_gt[i] == 1 or ignored_det[det_idx] == 1
        ):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (
                assigned_detection[i]
                or ignored_det[i] == -1
                or ignored_det[i] == 1
                or ignored_threshold[i]
            ):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def calculate_iou_partly(
    gt_annos, dt_annos, metric, num_parts=50, z_axis=1, z_center=1.0
):
    """Fast iou algorithm.

    this function can be used independently to do result analysis.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
        z_axis: height axis. kitti camera use 1, lidar use 2.
    """
    assert len(gt_annos) == len(dt_annos)

    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    split_parts = [i for i in split_parts if i != 0]
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        if metric == 0:

            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)

            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in gt_annos_part], 0
            )
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in gt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1
            )
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in dt_annos_part], 0
            )
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in dt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1
            )
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64
            )
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1
            )
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1
            )
            overlap_part = box3d_overlap(
                gt_boxes, dt_boxes, z_axis=z_axis, z_center=z_center
            ).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][
                    gt_num_idx : gt_num_idx + gt_box_num,
                    dt_num_idx : dt_num_idx + dt_box_num,
                ]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 3] - query_boxes[k, 1]
        )
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]
                )
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0])
                            * (boxes[n, 3] - boxes[n, 1])
                            + qbox_area
                            - iw * ih
                        )
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (
                            boxes[n, 3] - boxes[n, 1]
                        )
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """Kitti camera format z_axis=1."""
    from hat.core.rotate_box_utils_v2 import rotate_iou_v2

    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_v2(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    box3d_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


@numba.jit(nopython=True, parallel=True)
def box3d_overlap_kernel(
    boxes, qboxes, rinc, criterion=-1, z_axis=1, z_center=1.0
):
    """Box3d_overlap_kernel.

    z_axis: the z (height) axis.
    z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center),
                )
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center,
                )
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = area1 + area2 - inc
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def bev_box_overlap(boxes, qboxes, criterion=-1, stable=False):
    from hat.core.rotate_box_utils_v2 import rotate_iou_v2

    if stable:
        # riou = box_np_ops.riou_cc(boxes, qboxes)
        raise ValueError("stable bev_box_overlap not implemented.")
    else:
        riou = rotate_iou_v2(boxes, qboxes, criterion)
    return riou
