# Copyright (c) OpenMMLab. All rights reserved.

import pickle
import time
from functools import partial
from multiprocessing import Pool
from os import path as osp
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from shapely.geometry import LineString

from .tpfp import custom_tpfp_gen


def average_precision(
    recalls: np.ndarray, precisions: np.ndarray, mode: str = "area"
) -> Union[float, np.ndarray]:
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls: Array of recalls.
        precisions: Array of precisions.
        mode: Method to calculate average precision.
            Options are 'area' or '11points'.
            'area' means calculating the area under precision-recall curve,
            '11points' means calculating the average precision of recalls
            at [0, 0.1, ..., 1].

    Returns:
        Calculated average precision.
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1]
            )
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported'
        )
    if no_scale:
        ap = ap[0]
    return ap


def get_cls_results(
    gen_results: Dict,
    annotations: Dict,
    num_sample: int = 100,
    num_pred_pts_per_instance: int = 30,
    eval_use_same_gt_sample_num_flag: bool = False,
    class_id: int = 0,
    fix_interval: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get detection results and ground truth information of a certain class.

    Args:
        gen_results: List of generated results, same as `eval_map()`.
        annotations: List of annotation dictionaries, same as `eval_map()`.
        num_sample: Number of samples. Default is 100.
        num_pred_pts_per_instance: Number of predicted points per instance.
            Default is 30.
        eval_use_same_gt_sample_num_flag: Flag to use the same number of
            ground truth samples for evaluation. Default is False.
        class_id: ID of a specific class. Default is 0.
        fix_interval: Flag to fix the interval. Default is False.

    Returns:
        Detected bounding boxes, ground truth bounding boxes.
    """
    # if len(gen_results) == 0 or

    cls_gens, cls_scores = [], []
    for res in gen_results["vectors"]:
        if res["type"] == class_id:
            if len(res["pts"]) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res["pts"])
            else:
                line = res["pts"]
                line = LineString(line)

                if fix_interval:
                    distances = list(np.arange(1.0, line.length, 1.0))
                    distances = (
                        [
                            0,
                        ]
                        + distances
                        + [
                            line.length,
                        ]
                    )
                    sampled_points = np.array(
                        [
                            list(line.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                else:
                    distances = np.linspace(0, line.length, num_sample)
                    sampled_points = np.array(
                        [
                            list(line.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)

            cls_gens.append(sampled_points)
            cls_scores.append(res["confidence_level"])
    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res, -1)
        cls_scores = np.array(cls_scores)[:, np.newaxis]
        cls_gens = np.concatenate([cls_gens, cls_scores], axis=-1)
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0, num_pred_pts_per_instance * 2 + 1))
        else:
            cls_gens = np.zeros((0, num_sample * 2 + 1))
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')

    cls_gts = []
    for ann in annotations["vectors"]:
        if ann["type"] == class_id:
            # line = ann['pts'] +  np.array((1,1)) # for hdmapnet
            line = ann["pts"]
            # line = ann['pts'].cumsum(0)
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)

            cls_gts.append(sampled_points)
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts, -1)
    else:
        cls_gts = np.zeros((0, num_sample * 2))
    return cls_gens, cls_gts
    # ones = np.ones((num_gts,1))
    # tmp_cls_gens = np.concatenate([cls_gts,ones],axis=-1)
    # return tmp_cls_gens, cls_gts


def format_res_gt_by_classes(
    result_path: str,
    gen_results: List[List],
    annotations: List[Dict],
    cls_names: List[str] = None,
    num_pred_pts_per_instance: int = 30,
    eval_use_same_gt_sample_num_flag: bool = False,
    pc_range: Tuple[float, float, float, float, float, float] = (
        -15.0,
        -30.0,
        -5.0,
        15.0,
        30.0,
        3.0,
    ),
    nproc: int = 8,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Format detection results and ground truth information by classes.

    Args:
        result_path: Path to the results.
        gen_results: List of generated results.
        annotations: List of annotation dictionaries.
        cls_names: List of class names. Default is None.
        num_pred_pts_per_instance: Number of predicted points per instance.
            Default is 30.
        eval_use_same_gt_sample_num_flag: Flag to use the same number of
            ground truth samples for evaluation. Default is False.
        pc_range: Range of the point cloud.
            Default is (-15.0, -30.0, -5.0, 15.0, 30.0, 3.0).
        nproc: Number of processes for multiprocessing. Default is 8.

    Returns:
        Formatted detection results and ground truth information by classes.
    """
    assert cls_names is not None
    start_time = time.time()
    num_fixed_sample_pts = 100
    fix_interval = False
    print("results path: {}".format(result_path))

    output_dir = osp.join(*osp.split(result_path)[:-1])
    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)
    cls_gens, cls_gts = {}, {}
    print("Formatting ...")
    formatting_file = "cls_formatted.pkl"
    formatting_file = osp.join(output_dir, formatting_file)

    for i, clsname in enumerate(cls_names):

        gengts = pool.starmap(
            partial(
                get_cls_results,
                num_sample=num_fixed_sample_pts,
                num_pred_pts_per_instance=num_pred_pts_per_instance,
                eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,  # noqa E501
                class_id=i,
                fix_interval=fix_interval,
            ),
            zip(gen_results, annotations),
        )

        gens, gts = tuple(zip(*gengts))

        cls_gens[clsname] = gens
        cls_gts[clsname] = gts

    with open(formatting_file, "wb") as ff:
        pickle.dump([cls_gens, cls_gts], ff)
    print(
        "Cls data formatting done in {:.2f}s!! with {}".format(
            float(time.time() - start_time), formatting_file
        )
    )
    pool.close()
    return cls_gens, cls_gts


def eval_map(
    gen_results: List[List],
    annotations: List[Dict],
    cls_gens: Dict[str, List[np.ndarray]],
    cls_gts: Dict[str, List[np.ndarray]],
    threshold: float = 0.5,
    cls_names: List[str] = None,
    logger: Any = None,
    tpfp_fn: Any = None,
    pc_range: Tuple[float, float, float, float, float, float] = (
        -15.0,
        -30.0,
        -5.0,
        15.0,
        30.0,
        3.0,
    ),
    metric: Any = None,
    num_pred_pts_per_instance: int = 30,
    nproc: int = 8,
) -> Tuple[float, List[Dict[str, Union[int, np.ndarray]]]]:
    """Evaluate mean average precision (mAP) for object detection.

    Args:
        gen_results: List of generated results.
        annotations: List of annotation dictionaries.
        cls_gens: Dict containing lists of generated bounding boxes per class.
        cls_gts: Dict containing lists of gt bounding boxes per class.
        threshold: Threshold value for detection. Default is 0.5.
        cls_names: List of class names. Default is None.
        logger: Logger object for logging. Default is None.
        tpfp_fn: Function for computing true positives and false positives.
            Default is None.
        pc_range: Range of the point cloud.
            Default is (-15.0, -30.0, -5.0, 15.0, 30.0, 3.0).
        metric: Evaluation metric. Default is None.
        num_pred_pts_per_instance: Number of predicted points per instance.
            Default is 30.
        nproc: Number of processes for multiprocessing. Default is 8.

    Returns:
        Mean average precision (mAP) and evaluation results per class.
    """
    start_time = time.time()
    pool = Pool(nproc)

    eval_results = []

    for _, clsname in enumerate(cls_names):

        # get gt and det bboxes of this class
        cls_gen = cls_gens[clsname]
        cls_gt = cls_gts[clsname]
        # choose proper function according to datasets to compute tp and fp
        # XXX
        # func_name = cls2func[clsname]
        # tpfp_fn = tpfp_fn_dict[tpfp_fn_name]
        tpfp_fn = custom_tpfp_gen
        # Trick for serialized
        # only top-level function can be serized
        # somehow use partitial the return function is defined
        # at the top level.

        # tpfp=tpfp_fn(cls_gen[i],cls_gt[i],threshold=threshold,metric=metric)
        # TODO this is a hack
        tpfp_fn = partial(tpfp_fn, threshold=threshold, metric=metric)
        args = []
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(tpfp_fn, zip(cls_gen, cls_gt, *args))
        tp, fp = tuple(zip(*tpfp))

        num_gts = 0
        for _, bbox in enumerate(cls_gt):
            num_gts += bbox.shape[0]

        # sort all det bboxes by score, also sort tp and fp
        cls_gen = np.vstack(cls_gen)
        num_dets = cls_gen.shape[0]
        sort_inds = np.argsort(-cls_gen[:, -1])  # descending, high score front
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]

        # calculate recall and precision with tp and fp
        # num_det*num_res
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # calculate AP
        # if dataset != 'voc07' else '11points'
        mode = "area"
        ap = average_precision(recalls, precisions, mode)
        eval_results.append(
            {
                "num_gts": num_gts,
                "num_dets": num_dets,
                "recall": recalls,
                "precision": precisions,
                "ap": ap,
            }
        )
        print(
            "cls:{} done in {:.2f}s!!".format(
                clsname, float(time.time() - start_time)
            )
        )
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0

    return mean_ap, eval_results
