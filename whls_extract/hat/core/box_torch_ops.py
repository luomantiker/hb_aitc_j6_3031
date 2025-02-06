import math
from typing import Tuple

import horizon_plugin_pytorch as horizon
import torch


def rotate_nms_pcdet(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    pre_maxsize: int = None,
    post_max_size: int = None,
) -> torch.Tensor:
    """Perform bbox nms for point cloud detection.

    Args:
        boxes : detected boxes of 7 dimensions.
        scores : detected box scores.
        thresh : nms IoU threshold.
        pre_maxsize : Box score threshold before applying nms.
            Defaults to None.
        post_max_size : Box score threshold after appying nms.
            Defaults to None.

    Returns:
        selcted box indices.
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = horizon.nms3d(boxes, keep, thresh)
    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected


def boxes_iou3d_gpu_horizon(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """Compute 3d boxes bev_iou and 3d_iou.

    Args:
        boxes_a: (N, 7) [x, y, z, h, w, l, ry].
        boxes_b: (M, 7) [x, y, z, h, w, l, ry].
        need_bev: need result bev_iou or not.
    Returns:
        iou_3d: (N, M)
        iou_bev:(N, M)
    """

    iou_bev = horizon.box3d_iou_bev(
        boxes_a.contiguous(), boxes_b.contiguous()
    ).cpu()
    iou3d = horizon.box3d_iou(boxes_a.contiguous(), boxes_b.contiguous()).cpu()

    return iou3d, iou_bev


def boxes_aligned_iou3d_gpu(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    box_mode: str = "wlh",
    rect: bool = False,
) -> Tuple[torch.Tensor]:
    """Compute 3d_boxes bev_iou and 3d_iou with aligned algorithm.

    Args:
        boxes_a: (N, 7) [x, y, z, w, l, h, ry].
        boxes_b: (N, 7) [x, y, z, w, l, h, ry].
        rect: True/False means boxes in camera/velodyne coord system.
        Notice: (x, y, z) are real center.
    Returns:
        iou_3d: (N, M)
        iou_bev:(N, M)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    w_index, l_index, h_index = (
        box_mode.index("w") + 3,
        box_mode.index("l") + 3,
        box_mode.index("h") + 3,
    )

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], 1))
    ).zero_()  # (N, 1)
    overlaps_bev = horizon.box3d_overlap_bev(
        boxes_a.contiguous(), boxes_b.contiguous()
    )

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(-1, 1)  # (N, 1)
    iou_bev = overlaps_bev / torch.clamp(
        area_a + area_b - overlaps_bev, min=1e-7
    )  # [N, 1]

    half_h_a = boxes_a[:, h_index] / 2.0
    half_h_b = boxes_b[:, h_index] / 2.0
    boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(
        -1, 1
    )  # z - h/2, (N, 1)
    boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(
        -1, 1
    )  # z + h/2, (N, 1)
    boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(-1, 1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)  # (N, 1)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)  # (N, 1)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)  # (N, 1)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(
        -1, 1
    )  # (N, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(
        -1, 1
    )  # (N, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d, iou_bev


def _torch_string_type_to_class(ttype):
    type_map = {
        "torch.HalfTensor": torch.HalfTensor,
        "torch.FloatTensor": torch.FloatTensor,
        "torch.DoubleTensor": torch.DoubleTensor,
        "torch.IntTensor": torch.IntTensor,
        "torch.LongTensor": torch.LongTensor,
        "torch.ByteTensor": torch.ByteTensor,
        "torch.cuda.HalfTensor": torch.cuda.HalfTensor,
        "torch.cuda.FloatTensor": torch.cuda.FloatTensor,
        "torch.cuda.DoubleTensor": torch.cuda.DoubleTensor,
        "torch.cuda.IntTensor": torch.cuda.IntTensor,
        "torch.cuda.LongTensor": torch.cuda.LongTensor,
        "torch.cuda.ByteTensor": torch.cuda.ByteTensor,
    }
    return type_map[ttype]


def get_tensor_class(tensor):
    return _torch_string_type_to_class(tensor.type())


def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = math.sin(angle)
    rot_cos = math.cos(angle)
    point_type = get_tensor_class(points)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                point_type([rot_cos, 0, -rot_sin]),
                point_type([0, 1, 0]),
                point_type([rot_sin, 0, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack(
            [
                point_type([rot_cos, -rot_sin, 0]),
                point_type([rot_sin, rot_cos, 0]),
                point_type([0, 0, 1]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                point_type([1, 0, 0]),
                point_type([0, rot_cos, -rot_sin]),
                point_type([0, rot_sin, rot_cos]),
            ]
        )
    else:
        raise ValueError("axis should in range")
    return points @ rot_mat_T


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = [
        torch.min(boxes_corner[:, :, i], dim=1)[0] for i in range(ndim)
    ]
    standup_boxes.extend(
        torch.max(boxes_corner[:, :, i], dim=1)[0] for i in range(ndim)
    )
    return torch.stack(standup_boxes, dim=1)
