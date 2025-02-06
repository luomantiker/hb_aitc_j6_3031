# from ..chamfer_dist import ChamferDistance
from typing import List

import numpy as np
from scipy.spatial import distance
from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString
from shapely.strtree import STRtree


def custom_polyline_score(
    pred_lines: List[np.ndarray],
    gt_lines: List[np.ndarray],
    linewidth: float = 1.0,
    metric: str = "chamfer",
) -> np.ndarray:
    """Calculate custom polyline score for predicted and ground truth lines.

    Args:
        pred_lines: List of predicted lines.
            Each line represented as an array of shape (npts, 2).
        gt_lines: List of ground truth lines.
            Each line represented as an array of shape (npts, 2).
        linewidth: Width of each line. Default is 1.0.
        metric: Metric to use for scoring.
            Options are 'chamfer' or 'iou'. Default is 'chamfer'.

    Returns:
        Matrix containing scores for each predicted and ground truth line pair.
    """
    if metric == "iou":
        linewidth = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = [
        LineString(i).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for i in pred_lines
    ]
    gt_lines_shapely = [
        LineString(i).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for i in gt_lines
    ]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict(  # noqa C402
        (id(pt), i) for i, pt in enumerate(pred_lines_shapely)
    )

    if metric == "chamfer":
        iou_matrix = np.full((num_preds, num_gts), -100.0)
    elif metric == "iou":
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    else:
        raise NotImplementedError

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                if metric == "chamfer":
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], "euclidean"
                    )
                    # import pdb;pdb.set_trace()
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()

                    iou_matrix[pred_id, i] = -(valid_ba + valid_ab) / 2
                elif metric == "iou":
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, i] = inter / union

    return iou_matrix
