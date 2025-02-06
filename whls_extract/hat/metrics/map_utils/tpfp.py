from typing import Tuple

import numpy as np

from .tpfp_chamfer import custom_polyline_score


def custom_tpfp_gen(
    gen_lines: np.ndarray,
    gt_lines: np.ndarray,
    threshold: float = 0.5,
    metric: str = "chamfer",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate true positives and false positives for detected lines.

    Args:
        gen_lines: Detected lines of this image.
        gt_lines: Ground truth lines of this image.
        threshold: Threshold value for considering a detection
            as a true positive. Default is 0.5.
        metric: Metric to use for evaluation. Default is 'chamfer'.

    Returns:
        tuple: A tuple containing true positives (tp) and false positives (fp).
    """
    if metric == "chamfer":
        if threshold > 0:
            threshold = -threshold
    # else:
    #     raise NotImplementedError

    # import pdb;pdb.set_trace()
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_gens == 0:
        return tp, fp

    gen_scores = gen_lines[:, -1]  # n
    # distance matrix: n x m

    matrix = custom_polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, 2),
        gt_lines.reshape(num_gts, -1, 2),
        linewidth=2.0,
        metric=metric,
    )
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp
