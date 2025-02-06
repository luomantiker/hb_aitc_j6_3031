import copy
import math
from typing import Optional

import numpy as np

EPSILON = 1e-8


def rbbox_to_corners(corners: np.ndarray, rbbox: np.ndarray) -> np.ndarray:
    """Generate clockwise corners and rotate it clockwise.

    Args:
        corners: shape=[8], the 4 corner point of
            rotate bbox, [x0, y0, x1, y1, x2, y2, x3, y3].
        rbbox: shape=[5], the rotated bbox's
            info (x, y, l, w, yaw).
    """

    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.zeros((4,), dtype=np.float64)
    corners_y = np.zeros((4,), dtype=np.float64)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = (
            -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y
        )


def line_segment_intersection(
    pts1: np.ndarray, pts2: np.ndarray, i: int, j: int, temp_pts: np.ndarray
) -> bool:
    """Find intersection of 2 lines defined by their end points.

    Args:
        pts1: shape=[8], the 4 point of rotate_bbox1.
        pts2: shape=[8], the 4 point of rotate_bbox2.
        i: the index of pts1, from [0,3].
        j: the index of pts2, from [0,3].
        temp_pts: shape=[2], the tmp intersection point.

    Returns:
        bool: whether the two line have the intersected point.
    """

    point_A = np.zeros((2,), dtype=np.float64)
    point_B = np.zeros((2,), dtype=np.float64)
    point_C = np.zeros((2,), dtype=np.float64)
    point_D = np.zeros((2,), dtype=np.float64)
    point_A[0] = pts1[2 * i]
    point_A[1] = pts1[2 * i + 1]
    point_B[0] = pts1[2 * ((i + 1) % 4)]
    point_B[1] = pts1[2 * ((i + 1) % 4) + 1]
    point_C[0] = pts2[2 * j]
    point_C[1] = pts2[2 * j + 1]
    point_D[0] = pts2[2 * ((j + 1) % 4)]
    point_D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = point_B[0] - point_A[0]
    BA1 = point_B[1] - point_A[1]
    DA0 = point_D[0] - point_A[0]
    CA0 = point_C[0] - point_A[0]
    DA1 = point_D[1] - point_A[1]
    CA1 = point_C[1] - point_A[1]

    acd = DA1 * CA0 > CA1 * DA0
    bcd = (point_D[1] - point_B[1]) * (point_C[0] - point_B[0]) > (
        point_C[1] - point_B[1]
    ) * (point_D[0] - point_B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = point_D[0] - point_C[0]
            DC1 = point_D[1] - point_C[1]
            ABBA = point_A[0] * point_B[1] - point_B[0] * point_A[1]
            CDDC = point_C[0] * point_D[1] - point_D[0] * point_C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def point_in_quadrilateral(
    pt_x: float, pt_y: float, corners: np.ndarray
) -> bool:
    """Check whether a point lies in a rectangle defined by corners.

    Args:
        pt_x: point's x coordinate value
        pt_y: point's y coordinate value
        corners: 4 points of a rectangle.

    Returns:
        bool: the point in rectangle or not.
    """

    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return (
        abab + EPSILON >= abap
        and abap + EPSILON >= 0
        and adad + EPSILON >= adap
        and adap + EPSILON >= 0
    )


def quadrilateral_intersection(
    pts1: np.ndarray, pts2: np.ndarray, int_pts: np.ndarray
) -> int:
    """Find intersection points pf two boxes.

    Args:
        pts1: shape=[8], the 4 points of box1.
        pts2: shape=[8], the 4 points of box2.
        int_pts: shape=[16], the intersected
            point between box1 and box2.

    Returns:
        the number of intersection points.
    """

    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros((2,), dtype=np.float64)
    pts1_reshaped = pts1.reshape([4, 2])
    pts2_reshaped = pts2.reshape([4, 2])
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                if (
                    np.abs(pts1_reshaped - temp_pts).mean(axis=1) > EPSILON
                ).all() and (
                    np.abs(pts2_reshaped - temp_pts).mean(axis=1) > EPSILON
                ).all():  # skip the repeated points
                    int_pts[num_of_inter * 2] = temp_pts[0]
                    int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                    num_of_inter += 1
    return num_of_inter


def triangle_area(
    point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray
) -> float:
    """Calculate the triangle area.

    Args:
        point_a: point 1 in trangle.
        point_b: point 2 in trangle.
        point_c: point 3 in trangle.

    Returns:
        triangle_area.
    """
    return (
        (point_a[0] - point_c[0]) * (point_b[1] - point_c[1])
        - (point_a[1] - point_c[1]) * (point_b[0] - point_c[0])
    ) / 2.0


def polygon_area(polygon_pts: np.ndarray, num_intersection: int) -> float:
    """Calculate the polygon area by calculating triangle areas.

    Args:
        polygon_pts: shape=[16], the intersection points.
        num_intersection: the intersection points number.

    Returns:
        Intersection area.
    """

    area_val = 0.0
    for i in range(num_intersection - 2):
        area_val += abs(
            triangle_area(
                polygon_pts[:2],
                polygon_pts[2 * i + 2 : 2 * i + 4],
                polygon_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val


def get_inter_area_between_rot_box(
    rbbox1: np.ndarray, rbbox2: np.ndarray
) -> float:
    """Compute intersection of two rotated boxes.

    Args:
        rbox1: shape=[5], Rotated 2d box.
        rbox2: shape=[5], Rotated 2d box.
    Returns:
        Intersection area between two rotated boxes.
    """
    corners1 = np.zeros((8,), dtype=np.float64)
    corners2 = np.zeros((8,), dtype=np.float64)
    intersection_corners = np.zeros((16,), dtype=np.float64)
    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)
    num_intersection = quadrilateral_intersection(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return polygon_area(intersection_corners, num_intersection)


def sort_vertex_in_convex_polygon(int_pts: np.ndarray, num_of_inter: int):
    """Sort convex_polygon's vertices in clockwise order.

    Args:
        int_pts: shape=[16], the intersection points.
        num_of_inter: number of intersection points.
    """

    if num_of_inter > 0:
        center = np.zeros((2,), dtype=np.float64)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.zeros((2,), dtype=np.float64)
        vs = np.zeros((16,), dtype=np.float64)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d + EPSILON
            v[1] = v[1] / d + EPSILON
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


def rotate_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
) -> np.ndarray:
    """Compute rotated ious between GT and pred bboxes.

    The iou func is mainly modify by the mmdetection3d:
    https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/kitti_utils/rotate_iou.py, # noqa
    which removing the numba and cuda component, only using numpy.
    For more detail, please refer to this official link, if want to
    check the rotate_iou by your own sample, please using the unit_tests
    function: test_metric_3dv.py.

    another Rotate_iou implementation: https://github.com/lilanxiao/Rotated_IoU/blob/master/utiles.py # noqa

    Note:
        When `criterion` in `[0,1]`, there are some differences in the
        calculation compared to mmdetection3d. In mmdetection3d,
        `criterion=0` indicate `area_inter/area_gt_box`,
        `criterion=1` indicate `area_inter/area_pred_box`.

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion: Indicate different type of iou.
            -1 indicate `area_inter/(area_pred_box + area_gt_box - area_inter)`
            0 indicate `area_inter/area_pred_box`
            1 indicate `area_inter/area_gt_box`

    Returns:
        ious between pred and gt boxes.
    """
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    rotate_ious = np.zeros((num_pred, num_gt), dtype=np.float64)
    for idx_pred in range(num_pred):
        for idx_gt in range(num_gt):
            area_pred = pred_boxes[idx_pred][2] * pred_boxes[idx_pred][3]
            area_gt = gt_boxes[idx_gt][2] * gt_boxes[idx_gt][3]
            area_inter = get_inter_area_between_rot_box(
                pred_boxes[idx_pred], gt_boxes[idx_gt]
            )
            if criterion == -1:
                tmp_iou = area_inter / (area_pred + area_gt - area_inter)
            elif criterion == 0:
                tmp_iou = area_inter / area_pred
            elif criterion == 1:
                tmp_iou = area_inter / area_gt
            else:
                tmp_iou = area_inter
            rotate_ious[idx_pred][idx_gt] = tmp_iou

    return rotate_ious


def box_match_with_ct_nms(
    gt_bboxes: np.ndarray,
    pred_bboxes: np.ndarray,
    scale_l: float = 1.0,
    scale_w: float = 1.0,
    use_yaw_filter: bool = False,
    yaw_threshold: float = 3.15,
    use_mutual_ctnms: bool = False,
) -> np.ndarray:
    r"""Do bbox match by use ct-nms.

    For more detail, please refer to this link:
    https://horizonrobotics.feishu.cn/wiki/WkkmwjgudiNvbJkECCwcycJ5nlh

    Args:
        gt_boxes: GT vcs 2d boxes on bev.
            shape:[num_gt, 5], num_gt is the number of boxes.
            [x, y, l, w, yaw]
            _w_                 yaw (x-axis to box direction, Counterclockwise is positive) # noqa
            \  \                \  | (x-axis)
             \  \  l             \ |
              \__\    (y-axis)____\| (vcs Coordinate System)
        pred_boxes: Predict vcs 2d boxes on bev.
            shape:[num_pred, 5], num_pred is the number of boxes.
            [x, y, l, w, yaw]
        scale_l: Matching threshold in the l direction. If ct_dist_l <
            box_dim_l * scale_l, represents matching in the l direction.
        scale_w: Matching threshold in the w direction. If ct_dist_w <
            box_dim_w * scale_w, represents matching in the w direction.
        use_yaw_filter: Whether use yaw when match two boxes. If
            use_yaw_filter = True, only when the error in yaw between
            two boxes is less than a certain threshold can be matched.
        yaw_threshold: The threshold of yaw error.
        use_mutual_ctnms: If use_mutual_ctnms = False, only compare
            the ct distance with the gt bbox dim. Else, compare the ct
            distance with both gt bbox dim and pred dim.

    Returns:
        array of shape [num_gt, num_pred] and bool dtype .
        return[i][j] express whether the i-th gt bbox and the j-th pred
        bbox is matched.
    """
    num_gt = gt_bboxes.shape[0]
    num_pred = pred_bboxes.shape[0]
    gt_bboxes = np.repeat(gt_bboxes, num_pred, axis=0)
    pred_bboxes = np.tile(pred_bboxes, [num_gt, 1])

    gt_ct = gt_bboxes[:, :2]
    gt_lw = gt_bboxes[:, 2:4]
    gt_yaw = gt_bboxes[:, -1:]
    gt_cos_yaw = np.cos(gt_yaw)
    gt_sin_yaw = np.sin(gt_yaw)

    pred_ct = pred_bboxes[:, :2]
    pred_lw = pred_bboxes[:, 2:4]
    pred_yaw = pred_bboxes[:, -1:]
    pred_cos_yaw = np.cos(pred_yaw)
    pred_sin_yaw = np.sin(pred_yaw)

    gt_unit_vector_l = np.hstack([gt_cos_yaw, gt_sin_yaw])
    gt_unit_vector_w = np.hstack([-gt_sin_yaw, gt_cos_yaw])
    gt_ct_dist_l = ((pred_ct - gt_ct) * gt_unit_vector_l).sum(axis=-1)[
        ..., None
    ]
    gt_ct_dist_w = ((pred_ct - gt_ct) * gt_unit_vector_w).sum(axis=-1)[
        ..., None
    ]
    gt_thresh_l = gt_lw[:, :1] * scale_l
    gt_thresh_w = gt_lw[:, 1:2] * scale_w
    gt_ct_dist = np.abs(np.hstack([gt_ct_dist_l, gt_ct_dist_w]))
    gt_thresh = np.abs(np.hstack([gt_thresh_l, gt_thresh_w]))

    if use_mutual_ctnms:
        pred_unit_vector_l = np.hstack([pred_cos_yaw, pred_sin_yaw])
        pred_unit_vector_w = np.hstack([-pred_sin_yaw, pred_cos_yaw])

        pred_ct_dist_l = ((gt_ct - pred_ct) * pred_unit_vector_l).sum(axis=-1)[
            ..., None
        ]
        pred_ct_dist_w = ((gt_ct - pred_ct) * pred_unit_vector_w).sum(axis=-1)[
            ..., None
        ]
        pred_thresh_l = pred_lw[:, :1] * scale_l
        pred_thresh_w = pred_lw[:, 1:2] * scale_w
        pred_ct_dist = np.abs(np.hstack([pred_ct_dist_l, pred_ct_dist_w]))
        pred_thresh = np.abs(np.hstack([pred_thresh_l, pred_thresh_w]))

    match = (gt_ct_dist < gt_thresh).all(axis=-1)
    if use_mutual_ctnms:
        match = np.logical_or(match, (pred_ct_dist < pred_thresh).all(axis=-1))
    if use_yaw_filter:
        yaw_error = np.abs((gt_yaw - pred_yaw)) % (2 * np.pi)
        yaw_error_standard = yaw_error > np.pi
        yaw_error[yaw_error_standard] = (
            2 * np.pi - yaw_error[yaw_error_standard]
        )

        match = np.logical_and(match, yaw_error.squeeze() < yaw_threshold)
    match = match.reshape(num_gt, num_pred)

    return match


def let_nms(
    gt_bboxes,
    pred_bboxes,
    p_t: float,
    min_t: float,
    max_t: float,
    radius: float,
    e_loc_threshold: float,
    angle_threshold: float = 3.15,
    area_threshold: float = 1e-6,
    allow_yaw_opposite: bool = False,
    ct_nms_param: dict = None,
) -> np.ndarray:
    """Do nms between GT and pred bboxes by using the way of let-iou.

    This nms function is written based on the following paper:
    `LET-3D-AP: Longitudinal Error Tolerant 3D Average Precision for
    Camera-Only 3D Detection`. For more detail, please refer to this link:
    https://horizonrobotics.feishu.cn/wiki/wikcnNHqbpiTzAPPBVwiEujsMke#

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        p_t: The percentage of longitudinal error tolerance.
        min_t: The minimum tolerance distance for longitudinal error.
        max_t: The maximum tolerance distance for longitudinal error.
        radius: The radius range with the gt center point as the
            center is used to suppress the FP.
        e_loc_threshold: The threshold of location error.
        angle_threshold: The threshold of angle error, defaults is 3.15,
            which is rounding up pi to 2 decimal places, means no limit
            on angle.
        area_threshold: The threshold of area ratio, which means
            area of small box / area of large box should be large than
            area_threshold, so area_threshold should less than 1,
            defaults is 1e-6.
        allow_yaw_opposite: Whether the yaw angle is allowed to differ
            by 180 degrees, more details refer to this link:
            https://horizonrobotics.feishu.cn/wiki/HHULwkOwIiisCTkyRepcSrusnFf#
        ct_nms_param: If ct_nms_param is None, use the original center
            point distance to match repeating box. Else, use ct nms to
            replace the eloc. Detailed parameters can be found in function
            ct_match.

    Returns:
        keep or not of pred boxes.
    """
    num_pred = pred_bboxes.shape[0]
    num_gt = gt_bboxes.shape[0]
    left = np.ones((num_gt, num_pred))
    for idx_gt in range(num_gt):
        for idx_pred in range(num_pred):
            pred_bbox_center = pred_bboxes[idx_pred][:2][::-1]
            gt_bbox_center = gt_bboxes[idx_gt][:2][::-1]
            # eloc
            e_loc = pred_bbox_center - gt_bbox_center
            g_norm = np.linalg.norm(gt_bbox_center)
            p_norm = np.linalg.norm(pred_bbox_center)
            u_g = gt_bbox_center / g_norm
            u_p = pred_bbox_center / p_norm
            # elon
            e_lon = (e_loc.dot(u_g)) * u_g
            e_lon_norm = np.linalg.norm(e_lon)
            # al
            tl = max(p_t * g_norm, min_t)
            tl = min(tl, max_t)
            al = 1 - min(e_lon_norm / tl, 1.0)
            # eloc norm
            e_loc_norm = np.linalg.norm(e_loc)
            # ct_nms
            if ct_nms_param:
                ct_nms_match = box_match_with_ct_nms(
                    gt_bboxes=gt_bboxes[idx_gt : idx_gt + 1, :],
                    pred_bboxes=pred_bboxes[idx_pred : idx_pred + 1, :],
                    **ct_nms_param,
                )
                if ct_nms_match.all():
                    left[idx_gt][idx_pred] = 0
                    continue
            else:
                if e_loc_norm < radius:
                    left[idx_gt][idx_pred] = 0
                    continue
            if al > 0.0:
                # p aligned
                pred_aligned_center = (gt_bbox_center.dot(u_p)) * u_p
                e_loc_aligned = pred_aligned_center - gt_bbox_center
                e_loc_aligned_norm = np.linalg.norm(e_loc_aligned)
                # angle error
                pred_angle = pred_bboxes[idx_pred][-1]
                gt_angle = gt_bboxes[idx_gt][-1]
                error_angle = (pred_angle - gt_angle) % (np.pi * 2)
                error_angle = (
                    error_angle
                    if error_angle <= np.pi
                    else np.pi * 2 - error_angle
                )
                if allow_yaw_opposite:
                    error_angle = min(
                        abs(abs(pred_angle - gt_angle) - np.pi)
                        % np.pi,  # noqa
                        error_angle,
                    )
                # area ratio
                pred_area = pred_bboxes[idx_pred][2] * pred_bboxes[idx_pred][3]
                gt_area = gt_bboxes[idx_gt][2] * gt_bboxes[idx_gt][3]
                area_ratio = pred_area / (gt_area + 1e-8)
                if (
                    e_loc_aligned_norm < e_loc_threshold
                    and error_angle < angle_threshold
                    and area_ratio > area_threshold
                    and area_ratio < 1 / area_threshold
                ):
                    left[idx_gt][idx_pred] = 0
    return left


def let_iou_2d(
    pred_bbox: np.ndarray,
    gt_bbox: np.ndarray,
    p_t: float,
    max_t: float,
    min_t: float,
):
    """Compute 2d rotate iou between pred_bbox and gt_bbox by using 2d let iou.

    For more detail, please refer to this link:
    https://horizonrobotics.feishu.cn/wiki/wikcnNHqbpiTzAPPBVwiEujsMke#

    Args:
        pred_bbox: shape:[N, 5] (x, y, l, w, -yaw)
        gt_bbox: shape:[M, 5] (x, y, l, w, -yaw)
        p_t: Superparameter in let-nms. Maximum tolerance percentage, when
            the distance between gt and the camera is x,
            the maximum tolerance distance is p_t * x.
        max_t: Superparameter in let-nms. Maximum tolerance of distant targets.
        min_t: Superparameter in let-nms. Minimum tolerance of nearby targets.
    """
    num_pred = pred_bbox.shape[0]
    num_gt = gt_bbox.shape[0]
    pred_bboxes_repeat = np.repeat(pred_bbox, num_gt, axis=0)
    gt_bboxes_repeat = np.tile(gt_bbox, [num_pred, 1])
    pred_bbox_center = pred_bboxes_repeat[:, :2]
    gt_bbox_center = gt_bboxes_repeat[:, :2]
    e_loc = pred_bbox_center - gt_bbox_center  # e_loc: position error
    pred_norm = np.linalg.norm(pred_bbox_center, axis=1, keepdims=True)
    gt_norm = np.linalg.norm(gt_bbox_center, axis=1, keepdims=True)
    u_p = pred_bbox_center / pred_norm
    u_g = gt_bbox_center / gt_norm
    e_lon = (e_loc * u_g).sum(
        axis=1, keepdims=True
    ) * u_g  # e_lon: longitudinal position error
    e_lon_norm = np.linalg.norm(e_lon, axis=1, keepdims=True)
    tl = np.clip(p_t * gt_norm, min_t, max_t)
    al = 1 - np.clip(e_lon_norm / tl, None, 1.0)  # al: longitudinal affinity.
    pred_aligned_bbox_repeat = copy.deepcopy(pred_bboxes_repeat)
    al_align_index = (al > 0.0).squeeze(axis=1)
    # Move the prediction box along the line connecting the camera and pred,
    # until it is closest to the center point of the GT.
    pred_aligned_bbox_repeat[:, :2][al_align_index] = (
        gt_bbox_center[al_align_index] * u_p[al_align_index]
    ).sum(axis=1, keepdims=True) * u_p[al_align_index]
    let_ious = np.array(
        [
            rotate_iou(
                pred_aligned_bbox_repeat[i : i + 1, :],
                gt_bboxes_repeat[i : i + 1, :],
            ).squeeze()
            for i in range(gt_bboxes_repeat.shape[0])
        ]
    )
    let_ious = let_ious.reshape(num_pred, num_gt)
    al = al.reshape(num_pred, num_gt)
    return let_ious, al
