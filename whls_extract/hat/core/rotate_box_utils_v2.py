# noqa D205, D208, D400
"""
    Note: This script can only be imported locally where it is used,
    and cannot be imported globally in the script, otherwise it may
    cause OOM problems.
    TODO(mengyang.duan): fix this.
"""

import logging
import math
import warnings
from typing import Optional

import numba
import numpy as np
from numba import cuda

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

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

EPSILON = 1e-8
numba_cuda_available = False


def numba_jit(*args, **kwargs):  # noqa D205, D209, D400, D401
    """Wrapper of numba jit, will automatically choose to use
    `numba.cuda.jit` or `numba.jit` according to the machine environment."""

    wrapper = numba.jit(nopython=True)

    def wrap_func(func):
        return wrapper(func)

    return wrap_func


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba_jit(
    "(float32[:], float32[:], float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def trangle_area(a, b, c):
    return (
        (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
    ) / 2.0


@numba_jit("(float32[:], int32)", device=numba_cuda_available, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val


@numba_jit("(float32[:], int32)", device=numba_cuda_available, inline=True)
def sort_vertex_in_convex_polygon_numba(int_pts, num_of_inter):
    if num_of_inter > 0:
        if numba_cuda_available:
            center = cuda.local.array((2,), dtype=numba.float32)
        else:
            center = np.zeros((2,), dtype=np.float32)

        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter

        if numba_cuda_available:
            v = cuda.local.array((2,), dtype=numba.float32)
            vs = cuda.local.array((16,), dtype=numba.float32)
        else:
            v = np.zeros((2,), dtype=np.float32)
            vs = np.zeros((16,), dtype=np.float32)

        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            d = max(d, EPSILON)
            v[0] = v[0] / d
            v[1] = v[1] / d
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


@numba_jit(
    "(float32[:], float32[:], int32, int32, float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def line_segment_intersection_numba(pts1, pts2, i, j, temp_pts):
    if numba_cuda_available:
        A = cuda.local.array((2,), dtype=numba.float32)
        B = cuda.local.array((2,), dtype=numba.float32)
        C = cuda.local.array((2,), dtype=numba.float32)
        D = cuda.local.array((2,), dtype=numba.float32)
    else:
        A = np.zeros((2,), dtype=np.float32)
        B = np.zeros((2,), dtype=np.float32)
        C = np.zeros((2,), dtype=np.float32)
        D = np.zeros((2,), dtype=np.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@numba_jit(
    "(float32, float32, float32[:])", device=numba_cuda_available, inline=True
)
def point_in_quadrilateral_numba(pt_x, pt_y, corners):
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

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba_jit(
    "(float32[:], float32[:], float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def quadrilateral_intersection_numba(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral_numba(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral_numba(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    if numba_cuda_available:
        temp_pts = cuda.local.array((2,), dtype=numba.float32)
    else:
        temp_pts = np.zeros((2,), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection_numba(
                pts1, pts2, i, j, temp_pts
            )
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@numba_jit(
    "(float32[:], float32[:])", device=numba_cuda_available, inline=True
)
def rbbox_to_corners_numba(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]

    if numba_cuda_available:
        corners_x = cuda.local.array((4,), dtype=numba.float32)
        corners_y = cuda.local.array((4,), dtype=numba.float32)
    else:
        corners_x = np.zeros((4,), dtype=np.float32)
        corners_y = np.zeros((4,), dtype=np.float32)
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


@numba_jit(
    "(float32[:], float32[:])", device=numba_cuda_available, inline=True
)
def inter(rbbox1, rbbox2):
    if numba_cuda_available:
        corners1 = cuda.local.array((8,), dtype=numba.float32)
        corners2 = cuda.local.array((8,), dtype=numba.float32)
        intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    else:
        corners1 = np.zeros((8,), dtype=np.float32)
        corners2 = np.zeros((8,), dtype=np.float32)
        intersection_corners = np.zeros((16,), dtype=np.float32)

    rbbox_to_corners_numba(corners1, rbbox1)
    rbbox_to_corners_numba(corners2, rbbox2)

    num_intersection = quadrilateral_intersection_numba(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon_numba(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


@numba_jit(
    "(float32[:], float32[:], int32)", device=numba_cuda_available, inline=True
)
def dev_rotate_iou_eval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        union = area1 + area2 - area_inter
        union = max(union, EPSILON)
        return area_inter / union
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


def rotate_iou_cpu(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
) -> np.ndarray:
    """Rotated box iou running in numba cpu.

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
    rotate_ious = np.zeros((num_pred, num_gt), dtype=np.float32)
    for idx_pred in range(num_pred):
        for idx_gt in range(num_gt):
            iou = dev_rotate_iou_eval(
                pred_boxes[idx_pred], gt_boxes[idx_gt], criterion=criterion
            )
            rotate_ious[idx_pred][idx_gt] = iou

    return rotate_ious


@numba_jit(
    "(int64, int64, float32[:], float32[:], float32[:], int32)", fastmath=False
)  # noqa E501
def rotate_iou_kernel_eval(
    N,
    K,
    dev_boxes,
    dev_query_boxes,
    dev_iou,
    criterion=-1,
):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if tx < col_size:
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if tx < row_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = (
                row_start * threadsPerBlock * K
                + col_start * threadsPerBlock
                + tx * K
                + i
            )
            dev_iou[offset] = dev_rotate_iou_eval(
                block_boxes[tx * 5 : tx * 5 + 5],  # pred_box
                block_qboxes[i * 5 : i * 5 + 5],  # gt_box
                criterion,
            )


def rotate_iou_gpu(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
    device_id: Optional[int] = 0,
) -> np.ndarray:
    """Rotated box iou running in numba gpu.

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion: Indicate different type of iou.
            -1 indicate `area_inter/(area_pred_box + area_gt_box - area_inter)`
            0 indicate `area_inter/area_pred_box`
            1 indicate `area_inter/area_gt_box`
        device_id: GPU device id.

    Returns:
        ious between pred and gt boxes.
    """
    # box_dtype = boxes.dtype
    pred_boxes = pred_boxes.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)
    N = pred_boxes.shape[0]
    K = gt_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(pred_boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(gt_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion
        )
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(pred_boxes.dtype)


def rotate_iou_v2(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
    device_id: Optional[int] = 0,
) -> np.ndarray:
    """Compute rotated ious between GT and pred bboxes by numba.

    The iou func is mainly modify by the mmdetection3d:
    https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/kitti_utils/rotate_iou.py, # noqa

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
        device_id: GPU device id, used in `rotate_iou_gpu`

    Returns:
        ious between pred and gt boxes.
    """
    if numba_cuda_available:
        iou = rotate_iou_gpu(
            pred_boxes,
            gt_boxes,
            criterion=criterion,
            device_id=device_id,
        )
    else:
        iou = rotate_iou_cpu(
            pred_boxes,
            gt_boxes,
            criterion=criterion,
        )

    return iou
