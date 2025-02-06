# Copyright (c) Horizon Robotics. All rights reserved.

import json
from collections import defaultdict
from io import BytesIO
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib
import numpy as np
import pylab

from hat.core.box3d_utils import rotate_iou
from hat.core.rotate_box_utils import let_iou_2d

EPSILON = 1e-8

__all__ = [
    "bev3d_bbox_eval",
    "rotate_iou_matching",
    "calap",
]


def collect_data(
    batch,
    output,
    gt_cids,
    det_cids,
    save_real3d_res=False,
    eval_occlusion=False,
    ann_name="annos_bev_3d",
    eval_cutlane_cls=False,
    eval_cutlane_y=False,
    eval_cutlane_binding_lane_id=False,
    eval_velo=False,
):
    """Collect data for metric calculate in update stage.

    Args:
        batch (Dict): batch data of dataset.
        output (Dict): the model's output.
        gt_cids (List): eval_category_ids of ground truth.
        det_cids (List): eval_category_ids of pred's output.
        save_real3d_res (bool, optional): use to control the name of timestamp.
            will be remove future. Defaults to False.
        eval_occlusion (bool): whether need to evaluate occlusion attribute
        ann_name (str): anno name.
        eval_velo (bool): whether need to evaluate velocities attribute
            prediction.

    """

    assert "timestamp" in batch
    timestamps = np.array(batch["timestamp"].cpu())
    tag_info = batch.get("tag_info", {})
    if save_real3d_res:
        timestamps = [str(int(_time)) for _time in timestamps]
        rec_date = batch["pack_dir"]
        timestamps = [
            date + "__" + time for date, time in zip(rec_date, timestamps)
        ]
    else:
        timestamps = [str(int(_time * 1000)) for _time in timestamps]
    _gt_group_by_cid = {cid: [] for cid in gt_cids}
    _det_group_by_cid = {cid: [] for cid in det_cids}
    _timestamps = []
    with_gt = batch.get("with_gt", None)
    result_keys = [
        "bev3d_ct",
        "bev3d_loc_z",
        "bev3d_dim",
        "bev3d_rot",
        "bev3d_score",
        "bev3d_cls_id",
    ]
    if eval_occlusion:
        result_keys.append("bev3d_occlusion_id")
    if eval_cutlane_cls:
        result_keys.append("bev3d_cutlane_cls_id")
    if eval_cutlane_y:
        result_keys.append("bev3d_cutlane_y")
    if eval_cutlane_binding_lane_id:
        result_keys.append("bev3d_cutlane_binding_lane_id")
    if eval_velo:
        result_keys.append("bev3d_velocities")
    results = {k: output[k] for k in result_keys}
    batch_size, num_objs = output[result_keys[0]].shape[:2]

    # process model's pred
    for bs in range(batch_size):
        for obj_idx in range(num_objs):
            pred_items = {
                key: val[bs][obj_idx].cpu().numpy()
                for key, val in results.items()
            }
            pred_items["timestamp"] = timestamps[bs]
            pred_items["image_idx"] = bs
            pred_items.update({key: val[bs] for key, val in tag_info.items()})
            if with_gt is not None:
                pred_items["with_gt"] = with_gt[bs]
            else:
                pred_items["with_gt"] = True
            cid = pred_items["bev3d_cls_id"].tolist()
            if cid in _det_group_by_cid:
                _det_group_by_cid[cid] += [pred_items]

        # getting image timastamp
        _timestamps.append(timestamps[bs])

    # process ground truth
    assert (
        ann_name in batch
    ), "Please confirm the annos in batch, \
                check Bev3dTargetGenerator in auto3dv"
    annotations = batch[ann_name]

    for cid in det_cids:
        for bs in range(batch_size):
            cur_cid_idxs = (
                (annotations["vcs_cls_"][bs] == cid).nonzero().squeeze(-1)
            )
            if with_gt is not None and with_gt[bs] is False:
                assert (
                    len(cur_cid_idxs) == 0
                ), "GT should be empty while with_gt is False"
            gt_cur_cid = {
                key: out[bs][cur_cid_idxs] for key, out in annotations.items()
            }
            num_objs_gt = gt_cur_cid[list(gt_cur_cid.keys())[0]].shape[0]
            for gt_obj_idx in range(num_objs_gt):
                gt_items = {
                    key: val[gt_obj_idx].cpu().numpy()
                    for key, val in gt_cur_cid.items()
                }
                if "time_diff" in batch:
                    gt_items["time_diff"] = (
                        batch["time_diff"][bs].cpu().numpy()
                    )
                gt_items["timestamp"] = timestamps[bs]
                gt_items["image_idx"] = bs
                gt_items.update(
                    {key: val[bs] for key, val in tag_info.items()}
                )
                _gt_group_by_cid[cid] += [gt_items]

    return _gt_group_by_cid, _det_group_by_cid, _timestamps


def tag_by_range(
    data: float,
    value: Optional[Mapping[str, Sequence[Sequence[float]]]],
):
    for cls_id, key in enumerate(value):
        for range in value[key]:
            assert range[0] < range[1]
            if data >= range[0] and data < range[1]:
                return cls_id
    return -99  # ignore


def tag_by_category(
    cid: Union[float, str],
    mapper: Optional[Mapping[str, Sequence[Sequence[float]]]],
):
    for cls_id, key in enumerate(mapper):
        if cid in mapper[key]:
            return cls_id
    return -99  # ignore


def tag_by_vcs(
    data: Union[float, str],
    vcs_range: Optional[Mapping[str, Sequence[Sequence[float]]]],
):
    for cls_id, key in enumerate(vcs_range):
        for range in vcs_range[key]:
            if (
                data[0] >= range[0]
                and data[0] <= range[2]
                and data[1] >= range[1]
                and data[1] <= range[3]
            ):
                return cls_id
    return -99  # ignore


def obj_tagger(value, mapper, tagger_fn, tansforms=None):
    if value is None:
        return -99
    if isinstance(tagger_fn, str):
        tagger_fn = eval(tagger_fn)
    if callable(tansforms):
        value = tansforms(value)
    return tagger_fn(value, mapper)


def and_tag_combiner(base_tag_names, base_tags):
    combined_tag = None
    for child_tag_i in base_tag_names:
        if base_tags:
            if combined_tag is None:
                combined_tag = base_tags[child_tag_i] >= 0
            else:
                combined_tag *= base_tags[child_tag_i] >= 0

    if base_tags:
        combined_tag = combined_tag.astype(int)
        combined_tag[combined_tag == 0] = -99
    return combined_tag


def combine_tags(tag_names, base_tags, combiner="and_tag_combiner"):
    combined_tag = None
    if isinstance(combiner, str):
        combiner = eval(combiner)
    if callable(combiner):
        combined_tag = combiner(tag_names, base_tags)
    return combined_tag


def bev3d_bbox_eval(
    det_res: Mapping,
    annotation: Mapping,
    dep_thresh: Optional[Sequence[str]],
    score_threshold: float,
    iou_threshold: float,
    gt_max_depth: float,
    eval_vcs_range: Optional[Sequence[float]],
    enable_ignore: bool,
    vis_intervals: Optional[Sequence[str]],
    ego_ignore_range: Optional[Sequence[float]],
    eval_occlusion: bool,
    occlusion_ignore_id: int = -99,
    eval_mode: str = "bev_iou",
    let_iou_param: Optional[Mapping[str, float]] = None,
    ct_rot_size_threshold: Optional[Mapping] = None,
) -> Mapping:
    """Eval the metric between GT and pred boxes of bev3d.

    Args:
        det_res (Dict): the predict 3d boxes info.
        annotation (Dict): the ground truth 3d boxes info.
        dep_thresh (tuple of int, default: None): Depth range to
            validation.
        score_threshold (float): Threshold for score.
        iou_threshold (float): Threshold for IoU.
        gt_max_depth (float): Max depth for gts.
        eval_vcs_range (tuple of float): Max vcs
            (bottom, right, top, left) for gts & preds.
        enable_ignore (bool): Whether to use ignore_mask.
        ego_ignore_range (tuple of float): Ego range
            (bottom, right, top, left) to be ignored,(-0.6, -0.5, 2.0, 0.5)
            recommended based on the minimum tire diameter, wheelbase and track
        vis_intervals (tuple of str): Piecewise interval of visibility.
        eval_occlusion (bool): whether need to evaluate occlusion attribute
            prediction.
        eval_mode (str): which box matching scheme to use. only support
            bev_iou and let_iou.
        let_iou_param (Dict): Enable when eval_mode = let_iou. Contains various
            parameters related to let iou. Including "p_t", "min_t", "max_t".
        ct_rot_size_threshold: threshold for match by center/rot/size.
    Returns:
        (Dict): Dict contains the results.
    """
    assert eval_mode in ["bev_iou", "let_iou"]

    timestamp_count = 0
    total_timestamp_count = len(annotation["timestamps"])

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    for det in det_res:
        if det["bev3d_score"] < score_threshold:
            continue
        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < det["bev3d_ct"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < det["bev3d_ct"][1] < eval_vcs_range[-1]
            ):
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0]
                <= det["bev3d_ct"][0]
                <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= det["bev3d_ct"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_dets[det["timestamp"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_loc_"][0])  # vcs: abs(x)=depth

        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < gt["vcs_loc_"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < gt["vcs_loc_"][1] < eval_vcs_range[-1]
            ):
                continue
        else:
            if gt_depth > gt_max_depth:
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0] <= gt["vcs_loc_"][0] <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= gt["vcs_loc_"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_gts[gt["timestamp"]].append(gt)

    all_metric = [
        "dx",
        "dxp",
        "dy",
        "dyp",
        "dxy",
        "dxyp",
        "dw",
        "dwp",
        "dl",
        "dlp",
        "dh",
        "dhp",
        "drot",
    ]
    if eval_occlusion:
        all_metric.append("occlusion")
    metrics = {}
    for k in all_metric:
        metrics[k] = {}
        for vi in vis_intervals:
            metrics[k][vi] = [[] for _ in range(len(dep_thresh) + 1)]

    gt_matched = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    gt_missed = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    redundant_det = np.zeros(len(dep_thresh) + 1)
    if eval_mode == "let_iou":
        gt_al = np.zeros((len(vis_intervals), len(dep_thresh) + 1))

    num_gt = 0
    det_tp_mask = []
    det_gt_mask = []
    det_tp_pred_loc = []
    all_scores = []
    if eval_mode == "let_iou":
        det_tp_let_al_mask = []

    for timestamp in annotation["timestamps"]:
        det_bbox3d, det_scores, det_locs, det_yaw = [], [], [], []
        gt_bbox3d, gt_locs, gt_yaw, gt_ignore, gt_visible = [], [], [], [], []
        det_bbox3d_full, gt_bbox3d_full = [], []
        # fetch 3d ground truth box
        for gt in all_gts[timestamp]:
            dim = gt["vcs_dim_"]
            yaw = gt["vcs_rot_z_"]
            loc = gt["vcs_loc_"]
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            gt_bbox3d_full.append(
                [loc[0], loc[1], loc[2], dim[2], dim[1], dim[0], -yaw]
            )
            gt_bbox3d.append(bbox3d)
            gt_locs.append(gt["vcs_loc_"])
            gt_yaw.append(gt["vcs_rot_z_"])
            gt_visible.append(gt["vcs_visible_"])
            if enable_ignore:
                gt_ignore.append(gt["vcs_ignore_"])

        for det in all_dets[timestamp]:
            dim = det["bev3d_dim"]
            yaw = det["bev3d_rot"]
            loc = det["bev3d_ct"]
            # [x, y, l, w, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            det_bbox3d_full.append(
                [
                    loc[0],
                    loc[1],
                    det["bev3d_loc_z"],
                    dim[2],
                    dim[1],
                    dim[0],
                    -yaw,
                ]
            )
            det_bbox3d.append(bbox3d)
            assert det["bev3d_score"] >= 0
            det_scores.append(det["bev3d_score"])
            det_locs.append(det["bev3d_ct"])
            det_yaw.append(det["bev3d_rot"])

        det_locs = np.array(det_locs)
        gt_locs = np.array(gt_locs)
        gt_visible = np.array(gt_visible)
        det_bbox3d_full = np.array(det_bbox3d_full)
        gt_bbox3d_full = np.array(gt_bbox3d_full)
        if len(all_gts[timestamp]) == 0:
            if det_locs.any():
                pred_dep_thresh_inds = np.sum(
                    abs(det_locs[:, 0:1]) > dep_thresh, axis=-1
                )
                pred_dep_inds, pred_cnts = np.unique(
                    pred_dep_thresh_inds, return_counts=True
                )
                redundant_det[pred_dep_inds] += pred_cnts
            continue
        else:
            timestamp_count += 1

        if len(all_dets[timestamp]) == 0:
            for vis_idx, interval in enumerate(vis_intervals):
                vis_lthr, vis_rthr = [
                    float(_.translate({ord(i): None for i in "()"}))
                    for _ in interval.split(",")
                ]
                visib_gt_ind = (gt_visible >= vis_lthr) * (
                    gt_visible < vis_rthr
                )

                if enable_ignore:
                    valid_gt_ind = np.array(gt_ignore) == 0
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind * valid_gt_ind][:, 0:1])
                        > dep_thresh,
                        axis=-1,
                    )
                else:
                    gt_dep_thresh_inds = np.sum(
                        abs(gt_locs[visib_gt_ind][:, 0:1]) > dep_thresh,
                        axis=-1,
                    )

                gt_dep_inds, gt_cnts = np.unique(
                    gt_dep_thresh_inds, return_counts=True
                )
                gt_missed[vis_idx][gt_dep_inds] += gt_cnts
            # bev3d ap metric bug fix
            gt_det_loc = gt_locs.copy()
            if enable_ignore:
                gt_det_loc = gt_det_loc[np.invert(gt_ignore)]
            num_gt += len(gt_bbox3d) - sum(gt_ignore)
            det_gt_mask += gt_det_loc.tolist()
            continue

        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        if eval_mode == "bev_iou":
            (matched_dict, redundant, det_ignored_mask) = rotate_iou_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                iou_threshold,
                gt_ignore,
            )
        elif eval_mode == "let_iou":
            (matched_dict, redundant, det_ignored_mask) = let_iou_matching(
                det_bbox3d,
                gt_bbox3d,
                det_scores,
                iou_threshold,
                gt_ignore,
                let_iou_param,
            )
        elif eval_mode == "ct_rot_size":
            (matched_dict, redundant, det_ignored_mask) = ct_rot_size_matching(
                det_bbox3d_full,
                gt_bbox3d_full,
                det_scores,
                ct_rot_size_threshold,
                gt_ignore,
            )
        else:
            raise NotImplementedError
        redundant_det_mask = np.zeros(det_locs.shape[0], dtype=bool)
        for ind in redundant:
            redundant_det_mask[ind] = 1
        redundant_det_dep = np.sum(
            abs(det_locs[redundant_det_mask, 0:1]) > dep_thresh, axis=-1
        )
        redundant_det_dep_inds, redundant_det_dep_cnts = np.unique(
            redundant_det_dep, return_counts=True
        )
        redundant_det[redundant_det_dep_inds] += redundant_det_dep_cnts
        all_scores += det_scores[np.invert(det_ignored_mask)].tolist()
        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0
        tp = tp[np.invert(det_ignored_mask)]
        tp_det_loc = det_locs.copy()
        tp_det_loc = tp_det_loc[np.invert(det_ignored_mask)]
        gt_det_loc = gt_locs.copy()
        if enable_ignore:
            gt_det_loc = gt_det_loc[np.invert(gt_ignore)]
        num_gt += len(gt_bbox3d) - sum(gt_ignore)
        if eval_mode == "let_iou":
            al = matched_dict["let_al_det"]
            al = al[np.invert(det_ignored_mask)]
            det_tp_let_al_mask += al.tolist()

        det_tp_mask += tp.tolist()
        det_gt_mask += gt_det_loc.tolist()
        det_tp_pred_loc += tp_det_loc.tolist()
        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))

        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]
        for vis_idx, interval in enumerate(vis_intervals):
            vis_lthr, vis_rthr = [
                float(_.translate({ord(i): None for i in "()"}))
                for _ in interval.split(",")
            ]
            gt_visible_mask = (gt_visible >= vis_lthr) * (
                gt_visible < vis_rthr
            )

            if enable_ignore:
                gt_missed_mask = np.invert(mask) * np.invert(
                    np.array(gt_ignore) == 1
                )
            else:
                gt_missed_mask = np.invert(mask)
            gt_missed_dep_thresh_inds = np.sum(
                abs(gt_locs[gt_visible_mask * gt_missed_mask, 0:1])
                > dep_thresh,
                axis=-1,
            )
            gt_missed_dep_inds, gt_missed_cnts = np.unique(
                gt_missed_dep_thresh_inds, return_counts=True
            )
            gt_missed[vis_idx][gt_missed_dep_inds] += gt_missed_cnts

            if np.sum(mask) == 0:
                continue

            pred_dim = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_dim"]
                    for assign in det_assigns
                ]
            )
            pred_loc = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_ct"]
                    for assign in det_assigns
                ]
            )
            pred_yaw_rad = np.array(
                [
                    all_dets[timestamp][assign]["bev3d_rot"]
                    for assign in det_assigns
                ]
            )
            pred_yaw = np.rad2deg(pred_yaw_rad) % 360.0
            if eval_occlusion:
                pred_occlusion = np.array(
                    [
                        all_dets[timestamp][assign]["bev3d_occlusion_id"]
                        for assign in det_assigns
                    ]
                )

            gt_dim = np.array(
                [all_gts[timestamp][ind]["vcs_dim_"] for ind in inds]
            )
            gt_loc = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"] for ind in inds]
            )
            gt_depth = np.array(
                [all_gts[timestamp][ind]["vcs_loc_"][0] for ind in inds]
            )
            gt_yaw_rad = np.array(
                [all_gts[timestamp][ind]["vcs_rot_z_"] for ind in inds]
            )
            gt_vis = np.array(
                [all_gts[timestamp][ind]["vcs_visible_"] for ind in inds]
            )
            gt_yaw = np.rad2deg(gt_yaw_rad) % 360.0
            gt_vis_mask = (gt_vis >= vis_lthr) * (gt_vis < vis_rthr)
            if eval_occlusion:
                gt_occlusion = np.array(
                    [all_gts[timestamp][ind]["vcs_occlusion_"] for ind in inds]
                )

            dep_thresh_inds = np.sum(
                abs(gt_depth[gt_vis_mask][:, np.newaxis]) > dep_thresh, axis=-1
            )
            dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
            gt_matched[vis_idx][dep_inds.tolist()] += cnts

            if eval_mode == "let_iou":
                let_al_gt = matched_dict["let_al_gt"][mask]
                for dep_ind in dep_inds.tolist():
                    gt_al[vis_idx][dep_ind] += let_al_gt[
                        dep_thresh_inds == dep_ind
                    ].sum()

            dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
            dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
            dxy = (dx ** 2 + dy ** 2) ** 0.5
            dw = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
            dl = np.abs(pred_dim[:, 2] - gt_dim[:, 2])
            dh = np.abs(pred_dim[:, 0] - gt_dim[:, 0])

            dxp = dx / np.abs(gt_loc[:, 0])
            dyp = dy / np.abs(gt_loc[:, 1])
            dxyp = dxy / np.abs(gt_loc[:, 0] ** 2 + gt_loc[:, 1] ** 2) ** 0.5
            dxy_10p_error = dxyp <= 0.1
            dwp = dw / np.abs(gt_dim[:, 1])
            dlp = dl / np.abs(gt_dim[:, 2])
            dhp = dh / np.abs(gt_dim[:, 0])
            abs_rot = np.abs(gt_yaw - pred_yaw)
            drot = np.minimum(abs_rot, 360.0 - abs_rot)
            if eval_occlusion:
                if (gt_occlusion == occlusion_ignore_id).all():
                    occlusion = None
                else:
                    occlusion = (pred_occlusion == gt_occlusion).astype(
                        np.float32
                    )

            res = {
                "dx": dx,
                "dy": dy,
                "dxy": dxy,
                "dw": dw,
                "dl": dl,
                "dh": dh,
                "dxp": dxp,
                "dyp": dyp,
                "dxyp": dxyp,
                "dwp": dwp,
                "dlp": dlp,
                "dhp": dhp,
                "drot": drot,
                "dxy_10p_error": dxy_10p_error,
            }
            if eval_occlusion:
                res["occlusion"] = occlusion
            dep_thresh_inds = np.sum(
                abs(gt_depth[:, np.newaxis]) > dep_thresh, axis=-1
            )
            for key, val in metrics.items():
                if (
                    eval_occlusion
                    and key == "occlusion"
                    and res["occlusion"] is None
                ):
                    continue
                for dep_ind in dep_inds:
                    dep_mask = dep_thresh_inds == dep_ind
                    val[interval][dep_ind] += res[key][
                        dep_mask * gt_vis_mask
                    ].tolist()

    result_aps = {
        "det_tp_mask": det_tp_mask,
        "det_gt_mask": det_gt_mask,
        "det_tp_pred_loc": det_tp_pred_loc,
        "all_scores": all_scores,
        "num_gt": float(num_gt),
    }
    metrics["counts"] = {
        "gt_matched": gt_matched,
        "gt_missed": gt_missed,
        "redundant_det": redundant_det,
        "timestamp_count": timestamp_count,
        "total_timestamp_count": total_timestamp_count,
        "result_aps": result_aps,
    }
    if eval_mode == "let_iou":
        metrics["counts"]["result_aps"][
            "det_tp_let_al_mask"
        ] = det_tp_let_al_mask
        metrics["counts"]["gt_al"] = gt_al
    return metrics


def calap(recall, prec):
    """Calculate ap metric.

    Args:
        recall (np.ndarray): recalls for ap.
        prec (np.ndarray): precisions for ap.

    Returns:
        float: ap metric.
    """
    mrec = [0] + list(recall.flatten()) + [1]
    mpre = [0] + list(prec.flatten()) + [0]
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0
    for i in range(len(mpre) - 1):
        if mpre[i + 1] > 0:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def draw_pr_curves(eval_files, eval_names, draw_detection_rate=False):
    """Draw compare p-r curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names
        draw_detection_rate: whether draw detection rate curv.

    Returns
        b_io: buffer, buffer object contains a image
    """

    if draw_detection_rate:
        fig, (ax1, ax2) = pylab.subplots(1, 2, figsize=(15, 7))
        min_threshold = 0
        max_threshold = 1
    else:
        fig, ax1 = pylab.subplots(1, 1, figsize=(7, 7))
    line_color = {
        "0": "b",
        "1": "g",
        "2": "r",
        "3": "c",
        "4": "m",
        "5": "y",
        "6": "k",
        "7": "sandybrown",
        "8": "coral",
        "all": "fuchsia",
    }
    for eval_file, _ in zip(eval_files, eval_names):  # noqa
        results = json.load(eval_file)
        for cid, res in results.items():
            recall = res["recall"]
            precision = res["precision"]
            conf = res["conf"]
            detection_rate = res.get("detection_rate", None)

            ax1.plot(
                recall,
                precision,
                line_color[cid],
                label="num_gt:{}, class:{}".format(
                    results[str(cid)]["num_gt"], cid
                ),
            )
            if detection_rate:
                ax2.plot(conf, precision, label=f"class {cid}: precision")
                ax2.plot(conf, recall, label=f"class {cid}: recall")
                ax2.plot(
                    conf, detection_rate, label=f"class {cid}: detection_rate"
                )
                min_threshold = min(conf + [min_threshold])
                max_threshold = max(conf + [max_threshold])

    ax1.grid()
    ax1.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xticks(np.arange(0.0, 1, 0.05))
    ax1.set_yticks(np.arange(0.0, 1, 0.05))
    ax1.set_title("recall vs precision")

    if draw_detection_rate:
        ax2.grid(True)
        ax2.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
        ax2.set_xlabel("threshold")
        ax2.set_ylabel("recall & precision & detection_rate")
        ax2.set_xlim([min_threshold, max_threshold])
        ax2.set_ylim([0, 1])
        ax2.set_xticks(
            np.arange(
                min_threshold,
                max_threshold + 0.00001,
                (max_threshold - min_threshold) / 20.0,
            )
        )
        ax2.set_yticks(np.arange(0.0, 1.0, 0.05))
        ax2.set_title("thr vs recall & precision & detection_rate")

    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        pylab.xticks(rotation=45)

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    pylab.close(fig)
    return b_io


def draw_curves(
    eval_files, eval_names, output_file, draw_detection_rate=False
):
    """Draw compare curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names
        output_file: array-like, a list of output file
        draw_detection_rate: whether draw detection_rate curv.
    """
    b_io = draw_pr_curves(
        [open(eval_file, "r") for eval_file in eval_files],
        eval_names,
        draw_detection_rate,
    )
    with open(output_file, "wb") as f:
        f.write(b_io.read())


def ct_rot_matching(
    det_boxes: np.ndarray,
    det_locs: np.ndarray,
    gt_boxes: np.ndarray,
    gt_locs: np.ndarray,
    det_scores: np.ndarray,
    gt_ignore_mask=None,
) -> Mapping:
    """Match GT and pred boxes by center location on vcs and rotation.

    Args:
        det_boxes (np.ndarray): the predict boxes, shape:[N, 5].
        det_locs (np.ndarray): the predict location, shape:[N, 2].
        gt_boxes (np.ndarray): the GT boxes, shape: [K,5].
        gt_locs (np.ndarray): the gt location, shape: [K,3].
        det_scores (np.ndarray): the pred score, shape: [N,].
        gt_ignore_mask (Sequence, optional): the gt igonre list.

    Returns:
        Mapping: results based on location center and rotation matching.
    """
    det_ct, gt_ct = det_locs, gt_locs
    assert len(det_boxes) == len(det_scores)
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int64) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int64)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5
        dist_err = dxy / gt_dist

        det_yaw = np.rad2deg(det_boxes[dt_idx, -1])
        gt_yaw = np.rad2deg(gt_boxes[:, -1])
        abs_rot = np.abs(gt_yaw - det_yaw) % 180
        drot = np.minimum(abs_rot, 180.0 - abs_rot)
        gt_mask = ((gt_dist <= 50) | (dist_err <= 0.2)) & valid_gt

        dxy[gt_mask == 0] = 100
        min_ind = np.argmin(dxy)
        min_dxy = dxy[min_ind]
        if gt_dist[min_ind] < 30:
            dxy_thresh = 3.0
        elif gt_dist[min_ind] < 60:
            dxy_thresh = 5.0
        else:
            dxy_thresh = 8.0

        if min_dxy < dxy_thresh and drot[min_ind] < 20:
            if gt_ignore_mask and gt_ignore_mask[min_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[min_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[min_ind] = min_dxy
                valid_gt[min_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


def ct_rot_size_matching(
    det_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    det_scores: np.ndarray,
    ct_rot_size_threshold: Mapping,
    gt_ignore_mask=None,
) -> Mapping:
    """Match GT and pred boxes by center location on vcs, rotation and size.

    Args:
        det_boxes (np.ndarray): the predict boxes, shape:[N, 5].
        gt_boxes (np.ndarray): the GT boxes, shape: [K,5].
        det_scores (np.ndarray): the pred score, shape: [N,].
        ct_rot_size_threshold (dict): threshold, e.g. :
            {
                1000: {
                    "ct": 2,
                    "rot": 360,
                    "size": 0.2,
                }
            }, 1000 means the distance.
        gt_ignore_mask (Sequence, optional): the gt igonre list.

    Returns:
        Mapping: results based on location center and rotation matching.
    """
    det_ct, gt_ct = det_boxes[:, :3], gt_boxes[:, :3]
    assert len(det_boxes) == len(det_scores)
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int64) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int64)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5

        det_yaw = np.rad2deg(det_boxes[dt_idx, -1])
        gt_yaw = np.rad2deg(gt_boxes[:, -1])
        abs_rot = np.abs(gt_yaw - det_yaw) % 180
        drot = np.minimum(abs_rot, 180.0 - abs_rot)

        det_size = det_boxes[dt_idx, 3:6]
        gt_size = gt_boxes[:, 3:6]

        size_error = np.abs(det_size - gt_size)
        r_size_error = np.mean(size_error / gt_size, axis=1)

        gt_mask = valid_gt * (det_assign < 0)

        dxy[gt_mask == 0] = 100
        min_ind = np.argmin(dxy)
        min_dxy = dxy[min_ind]
        dist_range = sorted(ct_rot_size_threshold.keys())
        dxy_thresh, rot_thresh, size_thresh = 0, 0, 0
        for range_i in dist_range:
            if gt_dist[min_ind] < range_i:
                dxy_thresh = ct_rot_size_threshold[range_i]["ct"]
                rot_thresh = ct_rot_size_threshold[range_i]["rot"]
                size_thresh = ct_rot_size_threshold[range_i]["size"]
                break
        if (
            min_dxy < dxy_thresh
            and drot[min_ind] < rot_thresh
            and r_size_error[min_ind] < size_thresh
        ):
            if gt_ignore_mask and gt_ignore_mask[min_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[min_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[min_ind] = min_dxy
                valid_gt[min_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


def rotate_iou_matching(
    det_boxes: np.ndarray,
    det_locs: np.ndarray,
    gt_boxes: np.ndarray,
    gt_locs: np.ndarray,
    det_scores: np.ndarray,
    iou_threshold=0.2,
    gt_ignore_mask=None,
) -> Mapping:
    """Calculate the iou between GT and pred rotation boxes on vcs.

    Args:
        det_boxes (np.ndarray): the predict boxes, shape:[N, 5].
        det_locs (np.ndarray): the predict location, shape:[N, 2].
        gt_boxes (np.ndarray): the GT boxes, shape: [K,5].
        gt_locs (np.ndarray): the gt location, shape: [K,3].
        det_scores (np.ndarray): the pred score, shape: [N,].
        iou_threshold (float, optional): Defaults to 0.2.
        gt_ignore_mask (Sequence, optional): the gt igonre list.

    Returns:
        Mapping: results based on iou matching.
    """
    det_ct, gt_ct = det_locs, gt_locs
    assert len(det_boxes) == len(det_scores)
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int64) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int64)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    pairwise_iou = rotate_iou(det_boxes, gt_boxes)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    for dt_idx in det_sorted_idx:
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5
        dist_err = dxy / gt_dist
        gt_mask = ((gt_dist <= 50) | (dist_err <= 0.2)) & valid_gt
        det_iou = pairwise_iou[dt_idx] * gt_mask.astype(float)
        max_ind = np.argmax(det_iou)
        max_iou = det_iou[max_ind]
        if max_iou > iou_threshold:
            if gt_ignore_mask and gt_ignore_mask[max_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[max_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[max_ind] = max_iou
                valid_gt[max_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


def let_iou_matching(
    det_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    det_scores: np.ndarray,
    iou_threshold=0.2,
    gt_ignore_mask=None,
    let_parms: Mapping = None,
):
    """Calculate the iou between GT and pred rotation boxes on vcs.

    Args:
        det_boxes (np.ndarray): the predict boxes, shape:[N, 5].
        gt_boxes (np.ndarray): the GT boxes, shape: [K,5].
        det_scores (np.ndarray): the pred score, shape: [N,].
        iou_threshold (float, optional): Defaults to 0.2.
        gt_ignore_mask (Sequence, optional): the gt igonre list.

    Returns:
        Mapping: results based on iou matching.
    """
    assert len(det_boxes) == len(det_scores)
    let_rotate_ious, al = let_iou_2d(
        det_boxes,
        gt_boxes,
        **let_parms,
    )
    cost_metrix = let_rotate_ious * al
    det_sorted_idx = np.argsort(det_scores)[::-1]

    valid_gt = np.ones(gt_boxes.shape[0], dtype=bool)
    det_ignored_mask = np.zeros(shape=det_boxes.shape[0], dtype=bool)
    matched_det = np.zeros(shape=det_boxes.shape[0], dtype=np.int64)
    det_assign = np.zeros(shape=gt_boxes.shape[0], dtype=np.int64) - 1
    overlaps = np.zeros(shape=gt_boxes.shape[0])
    let_al_det = np.zeros(shape=det_boxes.shape[0])
    let_al_gt = np.zeros(shape=gt_boxes.shape[0])

    for det_idx in det_sorted_idx:
        det_cost = cost_metrix[det_idx] * valid_gt.astype(float)
        max_ind = np.argmax(det_cost)
        max_let_iou = let_rotate_ious[det_idx, max_ind]
        max_let_al = al[det_idx, max_ind]
        if max_let_iou > iou_threshold and det_cost[max_ind] > 0:
            if gt_ignore_mask and gt_ignore_mask[max_ind]:
                det_ignored_mask[det_idx] = True
                matched_det[det_idx] = -1
            else:
                det_assign[max_ind] = det_idx
                matched_det[det_idx] = 1
                overlaps[max_ind] = max_let_iou
                let_al_det[det_idx] = max_let_al
                let_al_gt[max_ind] = max_let_al
                valid_gt[max_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {
            "overlaps": overlaps,
            "det_assign": det_assign,
            "let_al_det": let_al_det,
            "let_al_gt": let_al_gt,
        },
        redundant,
        det_ignored_mask,
    )


def ct_matching(
    det_locs: np.ndarray,
    gt_locs: np.ndarray,
    det_scores: np.ndarray,
    ct_match_mode: Optional[str] = "percentage",
    gt_ignore_mask: Optional[Sequence[int]] = None,
    thresholds: Optional[Union[dict, float]] = None,
) -> Mapping:
    """Match GT and pred boxes by center location on vcs.

    The hyper-parameters refer to https://jira.hobot.cc:8443/browse/MSD-6559.
    Args:
        det_locs: the predict location, shape:[N, 2].
        gt_locs: the gt location, shape: [K,3].
        det_scores: the pred score, shape: [N,].
        ct_match_mode: match mode, default "percentage", or
            "distance": according to center location
        gt_ignore_mask: the gt igonre list.
        thresholds: the mapping of gt distance(meter) and matching
            threshold(percent of gt distance).
    Returns:
        Mapping: results based on location center matching.
    """
    det_ct, gt_ct = det_locs, gt_locs
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int64) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int64)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    if ct_match_mode != "distance":
        assert isinstance(thresholds, dict)
        thresholds = sorted(thresholds.items(), key=lambda x: x[0])

    for dt_idx in det_sorted_idx:
        if not np.any(valid_gt):
            # all the gt instances have been matched.
            break
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5

        dxy[valid_gt == 0] = np.inf
        min_ind = np.argmin(dxy)
        min_dxy = dxy[min_ind]
        if ct_match_mode == "percentage":
            for critical_dist, thresh_percent in thresholds:
                if gt_dist[min_ind] < critical_dist:
                    dxy_thresh = gt_dist[min_ind] * thresh_percent
                    break
                dxy_thresh = gt_dist[min_ind] * thresh_percent
        elif ct_match_mode == "distance":
            dxy_thresh = thresholds
        else:
            raise NotImplementedError(
                f"please confirm {ct_match_mode} is percentage or distance"
            )

        if min_dxy < dxy_thresh:
            if gt_ignore_mask and gt_ignore_mask[min_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[min_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[min_ind] = min_dxy
                valid_gt[min_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


class NpEncoder(json.JSONEncoder):
    """Json encoder for numpy array."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def bev3d_bbox_tag_eval(
    det_res: Mapping,
    annotation: Mapping,
    dep_thresh: Optional[Sequence[str]],
    score_threshold: float,
    iou_threshold: float,
    gt_max_depth: float,
    eval_vcs_range: Optional[Sequence[float]],
    enable_ignore: bool,
    vis_intervals: Optional[Sequence[str]],
    ego_ignore_range: Optional[Sequence[float]],
    eval_occlusion: bool,
    occlusion_ignore_id: int = -99,
    eval_mode: str = "bev_iou",
    let_iou_param: Optional[Mapping[str, float]] = None,
    eval_cls: Optional[bool] = True,
    target_cid: Optional[str] = "all",
    base_taggers: Optional[Dict] = None,
    taggers: Optional[Dict] = None,
    ct_rot_size_threshold: Optional[Mapping] = None,
    eval_velo: Optional[bool] = False,
    gt_select_keys: Optional[Mapping[str, str]] = None,
    pred_select_keys: Optional[Mapping[str, str]] = None,
) -> Mapping:
    """Eval the metric between GT and pred boxes of bev3d.

    Args:
        det_res (Dict): the predict 3d boxes info.
        annotation (Dict): the ground truth 3d boxes info.
        dep_thresh (tuple of int, default: None): Depth range to
            validation.
        score_threshold (float): Threshold for score.
        iou_threshold (float): Threshold for IoU.
        gt_max_depth (float): Max depth for gts.
        eval_vcs_range (tuple of float): Max vcs
            (bottom, right, top, left) for gts & preds.
        enable_ignore (bool): Whether to use ignore_mask.
        ego_ignore_range (tuple of float): Ego range
            (bottom, right, top, left) to be ignored,(-0.6, -0.5, 2.0, 0.5)
            recommended based on the minimum tire diameter, wheelbase and track
        vis_intervals (tuple of str): Piecewise interval of visibility.
        eval_occlusion (bool): whether need to evaluate occlusion attribute
            prediction.
        eval_mode (str): which box matching scheme to use. only support
            bev_iou and let_iou.
        let_iou_param (Dict): Enable when eval_mode = let_iou. Contains various
            parameters related to let iou. Including "p_t", "min_t", "max_t".
        eval_cls (bool): eval category precision.
        base_taggers: The base taggers for combination.
        taggers (Dict): tagger infos.
        ct_rot_size_threshold: threshold for match by center/rot/size.
    Returns:
        (Dict): Dict contains the results.
    """
    assert eval_mode in ["bev_iou", "let_iou", "ct_rot_size"]

    timestamp_count = 0
    total_timestamp_count = len(annotation["timestamps"])

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    if not gt_select_keys:
        gt_select_keys = {}
    if not pred_select_keys:
        pred_select_keys = {}
    if not taggers:
        taggers = {}
    if not base_taggers:
        base_taggers = {}
    if not ct_rot_size_threshold:
        ct_rot_size_threshold = {
            1000: {
                "ct": 2,
                "rot": 360,
                "size": 0.2,
            }
        }
    for det in det_res:
        if det["bev3d_score"] < score_threshold:
            continue
        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < det["bev3d_ct"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < det["bev3d_ct"][1] < eval_vcs_range[-1]
            ):
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0]
                <= det["bev3d_ct"][0]
                <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= det["bev3d_ct"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_dets[(det["timestamp"], det["image_idx"])].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_loc_"][0])  # vcs: abs(x)=depth

        if eval_vcs_range is not None:
            if not (
                eval_vcs_range[0] < gt["vcs_loc_"][0] < eval_vcs_range[2]
                and eval_vcs_range[1] < gt["vcs_loc_"][1] < eval_vcs_range[-1]
            ):
                continue
        else:
            if gt_depth > gt_max_depth:
                continue
        if ego_ignore_range is not None:
            if (
                ego_ignore_range[0] <= gt["vcs_loc_"][0] <= ego_ignore_range[2]
                and ego_ignore_range[1]
                <= gt["vcs_loc_"][1]
                <= ego_ignore_range[-1]
            ):
                continue
        all_gts[(gt["timestamp"], gt["image_idx"])].append(gt)

    metrics = {}
    detail_matrics = {}
    for vi in vis_intervals:
        detail_matrics[vi] = {
            "gt": {
                "value": [defaultdict(list) for _ in annotation["timestamps"]],
                "depth_wise": [
                    [np.zeros(0) for _ in range(len(dep_thresh) + 1)]
                    for _ in annotation["timestamps"]
                ],
            },
            "pred": {
                "value": [defaultdict(list) for _ in annotation["timestamps"]],
                "depth_wise": [
                    [np.zeros(0) for _ in range(len(dep_thresh) + 1)]
                    for _ in annotation["timestamps"]
                ],
            },
        }
        for tag_name in taggers:
            detail_matrics[vi]["pred"][tag_name] = [
                np.zeros(0) for _ in annotation["timestamps"]
            ]
            detail_matrics[vi]["gt"][tag_name] = [
                np.zeros(0) for _ in annotation["timestamps"]
            ]

    gt_matched = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    gt_missed = np.zeros((len(vis_intervals), len(dep_thresh) + 1))
    redundant_det = np.zeros(len(dep_thresh) + 1)
    if eval_mode == "let_iou":
        gt_al = np.zeros((len(vis_intervals), len(dep_thresh) + 1))

    num_gt = 0
    det_tp_mask = []
    det_gt_mask = []
    det_tp_pred_loc = []
    all_scores = []
    if eval_mode == "let_iou":
        det_tp_let_al_mask = []

    for img_ind, timestamp in enumerate(annotation["timestamps"]):
        det_bbox3d, det_scores, det_locs, det_yaw, det_occ = [], [], [], [], []
        gt_bbox3d, gt_locs, gt_yaw, gt_ignore, gt_visible, gt_occ = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with_gt = []
        det_bbox3d_full, gt_bbox3d_full = [], []
        det_velo, gt_velo = [], []
        gt_tags, det_tags = {}, {}
        base_gt_tags, base_det_tags = {}, {}
        gt_cids, det_cids = [], []
        gt_selected_values = defaultdict(list)
        pred_selected_values = defaultdict(list)
        # fetch 3d ground truth box
        for gt in all_gts[(timestamp, img_ind)]:
            dim = gt["vcs_dim_"]
            yaw = gt["vcs_rot_z_"]
            loc = gt["vcs_loc_"]
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            gt_bbox3d_full.append(
                [loc[0], loc[1], loc[2], dim[2], dim[1], dim[0], -yaw]
            )
            if eval_velo:
                gt_velo.append(gt["vcs_velocities"])
            for key, value in gt_select_keys.items():
                gt_selected_values[value].append(gt[key])
            gt_bbox3d.append(bbox3d)
            gt_locs.append(gt["vcs_loc_"])
            gt_yaw.append(gt["vcs_rot_z_"])
            gt_visible.append(gt["vcs_visible_"])
            gt_cids.append(gt.get("vcs_cls_", -99))
            gt_occ.append(gt.get("vcs_occlusion_", -99))
            if enable_ignore:
                gt_ignore.append(gt["vcs_ignore_"])
            for tag_name in base_taggers:
                if tag_name not in base_gt_tags:
                    base_gt_tags[tag_name] = []
                base_gt_tags[tag_name].append(
                    obj_tagger(
                        gt.get(base_taggers[tag_name]["gt_key"], None),
                        base_taggers[tag_name]["mapper"],
                        base_taggers[tag_name]["tagger_fn"],
                        base_taggers[tag_name].get("transforms", None),
                    )
                )

        for det in all_dets[(timestamp, img_ind)]:
            dim = det["bev3d_dim"]
            yaw = det["bev3d_rot"]
            loc = det["bev3d_ct"]
            # [x, y, l, w, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[2], dim[1], -yaw]
            det_bbox3d_full.append(
                [
                    loc[0],
                    loc[1],
                    det["bev3d_loc_z"],
                    dim[2],
                    dim[1],
                    dim[0],
                    -yaw,
                ]
            )
            det_bbox3d.append(bbox3d)
            if eval_velo:
                det_velo.append(det["bev3d_velocities"])
            for key, value in pred_select_keys.items():
                pred_selected_values[value].append(det[key])
            with_gt.append(det.get("with_gt", True))
            det_cids.append(det.get("bev3d_cls_id", -99))
            det_occ.append(det.get("bev3d_occlusion_id", -99))
            assert det["bev3d_score"] >= 0
            det_scores.append(det["bev3d_score"])
            det_locs.append(det["bev3d_ct"])
            det_yaw.append(det["bev3d_rot"])
            for tag_name in base_taggers:
                if tag_name not in base_det_tags:
                    base_det_tags[tag_name] = []
                base_det_tags[tag_name].append(
                    obj_tagger(
                        det.get(base_taggers[tag_name]["pred_key"], None),
                        base_taggers[tag_name]["mapper"],
                        base_taggers[tag_name]["tagger_fn"],
                        base_taggers[tag_name].get("transforms", None),
                    )
                )

        for tag_name in base_taggers:
            if tag_name in base_det_tags:
                base_det_tags[tag_name] = np.array(base_det_tags[tag_name])
            if tag_name in base_gt_tags:
                base_gt_tags[tag_name] = np.array(base_gt_tags[tag_name])
        for tag_name in taggers:
            base_tag_names = taggers[tag_name]["base_tags"]
            combiner = taggers[tag_name].get("combiner", "and_tag_combiner")
            det_combined_tag = combine_tags(
                base_tag_names, base_det_tags, combiner
            )
            gt_combined_tag = combine_tags(
                base_tag_names, base_gt_tags, combiner
            )
            if det_combined_tag is not None:
                det_tags[tag_name] = det_combined_tag
            if gt_combined_tag is not None:
                gt_tags[tag_name] = gt_combined_tag

        det_locs = np.array(det_locs)
        det_scores_all = np.array(det_scores)
        det_cids = np.array(det_cids)
        det_cid_mask = (
            det_cids == target_cid
            if target_cid != "all"
            else np.ones_like(det_cids) == 1
        )
        with_gt = np.array(with_gt)
        det_bbox3d_full = np.array(det_bbox3d_full)
        gt_locs = np.array(gt_locs)
        gt_visible = np.array(gt_visible)
        gt_cids = np.array(gt_cids)
        gt_cid_mask = (
            gt_cids == target_cid
            if target_cid != "all"
            else np.ones_like(gt_cids) == 1
        )
        gt_bbox3d_full = np.array(gt_bbox3d_full)
        for key, value in pred_selected_values.items():
            pred_selected_values[key] = np.array(value)
        for key, value in gt_selected_values.items():
            gt_selected_values[key] = np.array(value)
        if len(all_gts[(timestamp, img_ind)]) == 0:
            if det_locs.any():
                pred_dep_thresh_inds = np.sum(
                    abs(det_locs[:, 0:1]) > dep_thresh, axis=-1
                )
                pred_dep_inds, pred_cnts = np.unique(
                    pred_dep_thresh_inds, return_counts=True
                )
                redundant_det[pred_dep_inds] += pred_cnts

                for interval in detail_matrics:
                    for key, value in pred_selected_values.items():
                        detail_matrics[interval]["pred"]["value"][img_ind][
                            key
                        ].append(value)
                    for dep_ind in range(len(dep_thresh) + 1):
                        dep_mask = pred_dep_thresh_inds == dep_ind
                        pr_mask = (dep_mask * det_cid_mask).astype(np.int8) - 1
                        pr_mask[~with_gt] = -1
                        detail_matrics[interval]["pred"]["depth_wise"][
                            img_ind
                        ][dep_ind] = pr_mask
                    for tag_name in taggers:
                        pr_mask = (
                            (det_tags[tag_name] >= 0) * det_cid_mask
                        ).astype(np.int8) - 1
                        pr_mask[~with_gt] = -1
                        detail_matrics[interval]["pred"][tag_name][
                            img_ind
                        ] = pr_mask
                # bev3d ap metric bug fix
                vaild_det_cid_mask = det_cid_mask[with_gt]
                all_scores += det_scores_all[vaild_det_cid_mask].tolist()
                det_tp_pred_loc += det_locs[vaild_det_cid_mask].tolist()
                det_tp_mask += [0] * len(det_scores_all[vaild_det_cid_mask])
                if eval_mode == "let_iou":
                    det_tp_let_al_mask += [0] * len(
                        det_scores_all[vaild_det_cid_mask]
                    )
            continue
        else:
            timestamp_count += 1

        if len(all_dets[(timestamp, img_ind)]) == 0:
            for vis_idx, interval in enumerate(vis_intervals):
                vis_lthr, vis_rthr = [
                    float(_.translate({ord(i): None for i in "()"}))
                    for _ in interval.split(",")
                ]
                visib_gt_ind = (gt_visible >= vis_lthr) * (
                    gt_visible < vis_rthr
                )

                if enable_ignore:
                    valid_gt_ind = (np.array(gt_ignore) == 0) * visib_gt_ind
                else:
                    valid_gt_ind = visib_gt_ind
                gt_dep_thresh_inds = np.sum(
                    abs(gt_locs[:, 0:1]) > dep_thresh,
                    axis=-1,
                )
                gt_dep_inds, gt_cnts = np.unique(
                    gt_dep_thresh_inds[valid_gt_ind], return_counts=True
                )
                gt_missed[vis_idx][gt_dep_inds] += gt_cnts
                for key, value in gt_selected_values.items():
                    detail_matrics[interval]["gt"]["value"][img_ind][
                        key
                    ].append(value)
                for dep_ind in range(len(dep_thresh) + 1):
                    dep_mask = gt_dep_thresh_inds == dep_ind
                    pr_mask = (dep_mask * gt_cid_mask * valid_gt_ind).astype(
                        np.int8
                    ) - 1
                    detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                        dep_ind
                    ] = pr_mask
                for tag_name in taggers:
                    pr_mask = (
                        (gt_tags[tag_name] >= 0) * gt_cid_mask * valid_gt_ind
                    ).astype(np.int8) - 1
                    detail_matrics[interval]["gt"][tag_name][img_ind] = pr_mask

            # bev3d ap metric bug fix
            gt_det_loc = gt_locs.copy()
            if enable_ignore:
                gt_det_loc = gt_det_loc[np.invert(gt_ignore) * gt_cid_mask]
            else:
                gt_det_loc = gt_det_loc[gt_cid_mask]
            num_gt += len(gt_det_loc)
            det_gt_mask += gt_det_loc.tolist()

            continue
        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        if eval_mode == "bev_iou":
            (matched_dict, redundant, det_ignored_mask) = rotate_iou_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                iou_threshold,
                gt_ignore,
            )
        elif eval_mode == "let_iou":
            (matched_dict, redundant, det_ignored_mask) = let_iou_matching(
                det_bbox3d,
                gt_bbox3d,
                det_scores,
                iou_threshold,
                gt_ignore,
                let_iou_param,
            )
        elif eval_mode == "ct_rot_size":
            (matched_dict, redundant, det_ignored_mask) = ct_rot_size_matching(
                det_bbox3d_full,
                gt_bbox3d_full,
                det_scores,
                ct_rot_size_threshold,
                gt_ignore,
            )
        else:
            raise NotImplementedError
        redundant_det_mask = np.zeros(det_locs.shape[0], dtype=bool)
        for ind in redundant:
            redundant_det_mask[ind] = 1
        redundant_det_dep = np.sum(
            abs(det_locs[redundant_det_mask, 0:1]) > dep_thresh, axis=-1
        )
        redundant_det_dep_inds, redundant_det_dep_cnts = np.unique(
            redundant_det_dep, return_counts=True
        )
        redundant_det[redundant_det_dep_inds] += redundant_det_dep_cnts
        # save fp
        for interval in vis_intervals:
            for dep_ind in range(len(dep_thresh) + 1):
                redundant_dep_mask = redundant_det_dep == dep_ind
                pr_mask = redundant_dep_mask * det_cid_mask[redundant_det_mask]

                pr_mask = (pr_mask).astype(np.int8) - 1  # mask = 0 for FP
                detail_matrics[interval]["pred"]["depth_wise"][img_ind][
                    dep_ind
                ] = pr_mask

                pr_mask = (det_ignored_mask[det_ignored_mask]).astype(
                    np.int8
                ) * 0 - 1  # mask = -1 for ignore
                detail_matrics[interval]["pred"]["depth_wise"][img_ind][
                    dep_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["pred"]["depth_wise"][
                            img_ind
                        ][dep_ind],
                        pr_mask,
                    ],
                    axis=0,
                )

            for tag_name in taggers:
                pr_mask = (
                    det_tags[tag_name][redundant_det_mask] >= 0
                ) * det_cid_mask[redundant_det_mask]
                pr_mask = (pr_mask).astype(np.int8) - 1  # mask = 0 for FP
                detail_matrics[interval]["pred"][tag_name][img_ind] = pr_mask
                pr_mask = (det_ignored_mask[det_ignored_mask]).astype(
                    np.int8
                ) * 0 - 1  # mask = -1 for ignore
                detail_matrics[interval]["pred"][tag_name][
                    img_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["pred"][tag_name][img_ind],
                        pr_mask,
                    ],
                    axis=0,
                )
            for key, value in pred_selected_values.items():
                detail_matrics[interval]["pred"]["value"][img_ind][key].extend(
                    [
                        value[redundant_det_mask],
                        value[det_ignored_mask],
                    ]
                )

        all_scores += det_scores[
            np.invert(det_ignored_mask) * det_cid_mask
        ].tolist()
        tp = np.ones(len(det_scores), dtype=bool)
        tp[redundant] = 0
        tp = tp[np.invert(det_ignored_mask) * det_cid_mask]
        tp_det_loc = det_locs.copy()
        tp_det_loc = tp_det_loc[np.invert(det_ignored_mask) * det_cid_mask]
        gt_det_loc = gt_locs.copy()
        if enable_ignore:
            gt_det_loc = gt_det_loc[np.invert(gt_ignore) * gt_cid_mask]
        else:
            gt_det_loc = gt_det_loc[gt_cid_mask]
        num_gt += len(gt_det_loc)
        if eval_mode == "let_iou":
            al = matched_dict["let_al_det"]
            al = al[np.invert(det_ignored_mask) * det_cid_mask]
            det_tp_let_al_mask += al.tolist()
        det_tp_mask += tp.tolist()
        det_gt_mask += gt_det_loc.tolist()
        det_tp_pred_loc += tp_det_loc.tolist()
        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))
        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]
        for vis_idx, interval in enumerate(vis_intervals):
            vis_lthr, vis_rthr = [
                float(_.translate({ord(i): None for i in "()"}))
                for _ in interval.split(",")
            ]
            gt_visible_mask = (gt_visible >= vis_lthr) * (
                gt_visible < vis_rthr
            )

            if enable_ignore:
                gt_missed_mask = np.invert(mask) * np.invert(
                    np.array(gt_ignore) == 1
                )
                gt_ignore_mask = (np.array(gt_ignore) == 1) * np.invert(
                    gt_visible_mask
                )
            else:
                gt_missed_mask = np.invert(mask)
                gt_ignore_mask = np.invert(gt_visible_mask)
            gt_missed_dep_thresh_inds = np.sum(
                abs(gt_locs[gt_visible_mask * gt_missed_mask, 0:1])
                > dep_thresh,
                axis=-1,
            )
            gt_missed_dep_inds, gt_missed_cnts = np.unique(
                gt_missed_dep_thresh_inds, return_counts=True
            )
            gt_missed[vis_idx][gt_missed_dep_inds] += gt_missed_cnts
            # save depth-wise ignore
            for dep_ind in range(len(dep_thresh) + 1):
                missed_dep_mask = gt_missed_dep_thresh_inds == dep_ind
                pr_mask = (
                    missed_dep_mask
                    * gt_cid_mask[gt_visible_mask * gt_missed_mask]
                )
                pr_mask = (pr_mask).astype(np.int8) - 1  # mask = 0 for FN
                detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                    dep_ind
                ] = pr_mask

                pr_mask = (
                    gt_ignore_mask[gt_ignore_mask].astype(np.int8) * 0
                    - 1  # mask = -1 for ignore
                )
                detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                    dep_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                            dep_ind
                        ],
                        pr_mask,
                    ],
                    axis=0,
                )
            # save tag-wise ignore
            for tag_name in taggers:
                pr_mask = (
                    gt_tags[tag_name][gt_visible_mask * gt_missed_mask] >= 0
                ) * gt_cid_mask[gt_visible_mask * gt_missed_mask]
                pr_mask = (pr_mask).astype(np.int8) - 1  # mask = 0 for FN
                detail_matrics[interval]["gt"][tag_name][img_ind] = pr_mask

                pr_mask = (
                    gt_ignore_mask[gt_ignore_mask].astype(np.int8) * 0
                    - 1  # mask = -1 for ignore
                )
                detail_matrics[interval]["gt"][tag_name][
                    img_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["gt"][tag_name][img_ind],
                        pr_mask,
                    ],
                    axis=0,
                )

            for key, value in gt_selected_values.items():
                detail_matrics[interval]["gt"]["value"][img_ind][key].extend(
                    [
                        value[gt_visible_mask * gt_missed_mask],
                        value[gt_ignore_mask],
                    ]
                )

            if np.sum(mask) == 0:
                continue

            for key, value in pred_select_keys.items():
                detail_matrics[interval]["pred"]["value"][img_ind][
                    value
                ].append(
                    np.array(
                        [
                            all_dets[(timestamp, img_ind)][assign][key]
                            for assign in det_assigns
                        ]
                    )
                )
            gt_depth = np.array(
                [
                    all_gts[(timestamp, img_ind)][ind]["vcs_loc_"][0]
                    for ind in inds
                ]
            )
            gt_vis = np.array(
                [
                    all_gts[(timestamp, img_ind)][ind]["vcs_visible_"]
                    for ind in inds
                ]
            )
            matched_gt_cid_mask = np.array([gt_cid_mask[ind] for ind in inds])
            gt_vis_mask = (gt_vis >= vis_lthr) * (gt_vis < vis_rthr)
            for key, value in gt_select_keys.items():
                detail_matrics[interval]["gt"]["value"][img_ind][value].append(
                    np.array(
                        [
                            all_gts[(timestamp, img_ind)][ind][key]
                            for ind in inds
                        ]
                    )
                )
            dep_thresh_inds = np.sum(
                abs(gt_depth[gt_vis_mask][:, np.newaxis]) > dep_thresh, axis=-1
            )
            dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
            gt_matched[vis_idx][dep_inds.tolist()] += cnts

            if eval_mode == "let_iou":
                let_al_gt = matched_dict["let_al_gt"][mask]
                for dep_ind in dep_inds.tolist():
                    gt_al[vis_idx][dep_ind] += let_al_gt[
                        dep_thresh_inds == dep_ind
                    ].sum()

            dep_thresh_inds = np.sum(
                abs(gt_depth[:, np.newaxis]) > dep_thresh, axis=-1
            )

            for dep_ind in range(len(dep_thresh) + 1):
                dep_mask = dep_thresh_inds == dep_ind
                pr_mask = dep_mask * gt_vis_mask * matched_gt_cid_mask

                pr_mask = (pr_mask).astype(np.int8) * 2 - 1
                detail_matrics[interval]["pred"]["depth_wise"][img_ind][
                    dep_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["pred"]["depth_wise"][
                            img_ind
                        ][dep_ind],
                        pr_mask,
                    ],
                    axis=0,
                )
                detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                    dep_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["gt"]["depth_wise"][img_ind][
                            dep_ind
                        ],
                        pr_mask,
                    ],
                    axis=0,
                )

            for tag_name in taggers:
                pr_mask = (
                    (gt_tags[tag_name][inds] >= 0)
                    * gt_vis_mask
                    * matched_gt_cid_mask
                )
                pr_mask = (pr_mask).astype(np.int8) * 2 - 1
                detail_matrics[interval]["pred"][tag_name][
                    img_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["pred"][tag_name][img_ind],
                        pr_mask,
                    ],
                    axis=0,
                )
                detail_matrics[interval]["gt"][tag_name][
                    img_ind
                ] = np.concatenate(
                    [
                        detail_matrics[interval]["gt"][tag_name][img_ind],
                        pr_mask,
                    ],
                    axis=0,
                )

    for img_ind, _ in enumerate(annotation["timestamps"]):
        for interval in vis_intervals:
            for value in gt_select_keys.values():
                if value in detail_matrics[interval]["gt"]["value"][img_ind]:
                    detail_matrics[interval]["gt"]["value"][img_ind][
                        value
                    ] = np.concatenate(
                        detail_matrics[interval]["gt"]["value"][img_ind][value]
                    )
            for value in pred_select_keys.values():
                if value in detail_matrics[interval]["pred"]["value"][img_ind]:
                    detail_matrics[interval]["pred"]["value"][img_ind][
                        value
                    ] = np.concatenate(
                        detail_matrics[interval]["pred"]["value"][img_ind][
                            value
                        ]
                    )
    result_aps = {
        "det_tp_mask": det_tp_mask,
        "det_gt_mask": det_gt_mask,
        "det_tp_pred_loc": det_tp_pred_loc,
        "all_scores": all_scores,
        "num_gt": float(num_gt),
    }
    metrics["counts"] = {
        "gt_matched": gt_matched,
        "gt_missed": gt_missed,
        "redundant_det": redundant_det,
        "timestamp_count": timestamp_count,
        "total_timestamp_count": total_timestamp_count,
        "result_aps": result_aps,
    }
    metrics["detail"] = detail_matrics
    if eval_mode == "let_iou":
        metrics["counts"]["result_aps"][
            "det_tp_let_al_mask"
        ] = det_tp_let_al_mask
        metrics["counts"]["gt_al"] = gt_al
    return metrics
