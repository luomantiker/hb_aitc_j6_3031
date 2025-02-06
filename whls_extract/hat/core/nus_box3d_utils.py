import copy
import logging
from typing import List, Tuple

import numpy as np
import torch

from hat.utils.package_helper import require_packages

logger = logging.getLogger(__name__)

try:
    from nuscenes.eval.common.utils import Quaternion
except ImportError:
    Quaternion = None


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid.

    Args:
        x: The tensor to do the
            inverse.
        eps: EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def bbox3d_nus_transform(bboxes: torch.Tensor) -> torch.Tensor:
    """Transform 3d box to NuScenes format data. \
        Transform bbox3d from format (cx, cy, cz, w, l, h, yaw) \
        to format (cx, cy, log_w, log_l, cz, log_h, yaw_sin, yaw_cos).

    Args:
        bboxes: Shape (N, 7 or 9), 7 in (cx, cy, cz, w, l,
            h, yaw) format or 9 in (cx, cy, cz, w, l,
            h, yaw, velocity_x, velocity_y) format.

    Returns:
        normalized_bbox with nus format: Shape (N, 8 or 10).
            8 in (cx, cy, log_w, log_l, cz, log_h, yaw_sin,
            yaw_cos) format and 10 in (cx, cy, log_w, log_l,
            cz, log_h, yaw_sin, yaw_cos, velocity_x,
            velocity_y) format.
    """

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w_ = bboxes[..., 3:4].log()
    l_ = bboxes[..., 4:5].log()
    h_ = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w_, l_, cz, h_, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w_, l_, cz, h_, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def inverse_bbox3d_nus_transform(
    normalized_bboxes: torch.Tensor,
) -> torch.Tensor:
    """Inverse transform 3d box from NuScenes format data. \
        Transform bbox3d from format (cx, cy, log_w, log_l, \
        cz, log_h, yaw_sin, yaw_cos) to format (cx, cy, cz, \
        w, l, h, yaw).

    Args:
        normalized_bbox: Shape (N, 8 or 10).
            8 in (cx, cy, log_w, log_l, cz, log_h, yaw_sin,
            yaw_cos) format and 10 in (cx, cy, log_w, log_l,
            cz, log_h, yaw_sin, yaw_cos, velocity_x,
            velocity_y) format.

    Returns:
        denormalized_bbox: Shape (N, 7 or 9), 7 in (cx, cy,
            cz, w, l, h, yaw) format and 9 in (cx, cy, cz, w,
            l, h, yaw, velocity_x, velocity_y) format.
    """

    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w_ = normalized_bboxes[..., 2:3]
    l_ = normalized_bboxes[..., 3:4]
    h_ = normalized_bboxes[..., 5:6]

    w_ = w_.exp()
    l_ = l_.exp()
    h_ = h_.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w_, l_, h_, rot, vx, vy], dim=-1
        )
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w_, l_, h_, rot], dim=-1)
    return denormalized_bboxes


def get_min_max_coords(
    bev_size: Tuple[float],
) -> Tuple[float, float, float, float]:
    """Get min and max coords.

    Args:
        bev_size: Bev size.
    """

    min_x = -bev_size[1] + bev_size[2] / 2
    max_x = bev_size[1] - bev_size[2] / 2
    min_y = -bev_size[0] + bev_size[2] / 2
    max_y = bev_size[0] - bev_size[2] / 2
    return min_x, max_x, min_y, max_y


def adjust_coords(coords: torch.Tensor, grid_size: Tuple[int]) -> torch.Tensor:
    """Adjust coords for hnn grid_sample.

    Args:
        coords: Coords for grid_sample.
        grid_size: Grid size.
    """

    W = grid_size[0]
    H = grid_size[1]

    bev_x = (torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)).float()
    bev_y = (torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)).float()

    bev_coords = torch.stack([bev_x, bev_y], axis=-1).to(device=coords.device)
    coords = coords - bev_coords
    return coords


@require_packages(
    "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
)
def bbox_to_corner(
    bboxes: torch.Tensor, score_thresh: float = 0.0
) -> Tuple[List[np.array], List[float], List[float]]:
    """Get 3dbbox corner.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        score_thresh: Theshold for filtering bbox with low score.
    """

    decoded_bbox = []
    scores = []
    cat_ids = []
    for bbox in bboxes:
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()

        score = bbox[9]
        if score < score_thresh:
            continue
        w, l, h = bbox[3:6]
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        # Rotate
        yaw = bbox[6]
        rot = Quaternion(axis=[0, 0, 1], radians=yaw)

        corners = np.dot(rot.rotation_matrix, corners)

        # Translate
        x, y, z = bbox[:3]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners = np.transpose(corners, (1, 0))
        decoded_bbox.append(corners)
        if len(bbox) == 11:
            scores.append(bbox[9])
            cat_ids.append(bbox[10])
        else:
            scores.append(1.0)
            cat_ids.append(bbox[9])

    return decoded_bbox, scores, cat_ids


def bbox_bev2ego(bboxes: torch.Tensor, bev_size: Tuple[float]) -> torch.Tensor:
    """Convert bev coordinate to ego coordinate.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        bev_size: Bev size.
    """

    ego_bboxes = []
    min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
    for bbox in bboxes:
        bbox = copy.deepcopy(bbox)
        bbox[:6] *= bev_size[2]
        bbox[0] -= max_x
        bbox[1] -= max_y
        bbox[7:9] *= bev_size[2]
        ego_bboxes.append(bbox)
    return ego_bboxes


def bbox_ego2bev(bboxes: torch.Tensor, bev_size: Tuple[float]) -> torch.Tensor:
    """Convert ego coordinate to bev coordinate.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        bev_size: Bev size.
    """

    bev_bboxes = []
    min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
    for bbox in bboxes:
        bbox = copy.deepcopy(bbox)
        bbox[0] += max_x
        bbox[1] += max_y
        bbox[:6] /= bev_size[2]
        bbox[7:9] /= bev_size[2]
        bev_bboxes.append(bbox)
    return bev_bboxes


def bbox_ego2img(
    bboxes: torch.Tensor,
    ego2img: np.array,
    img_size: Tuple[int],
    score_thresh: float = 0.0,
):
    """Convert ego coordinate to img coordinate.

    Args:
        bboxes: Meta info for bbox. Shape as (n, 11) or (n, 10).
        ego2img: Homography for ego to image coordinate.
        img_size: Image size.
        score_thresh: Theshold for filtering bbox with low score.
    """

    def _is_visible(corners_2d, corners_3d, imsize):
        visible = np.logical_and(
            corners_2d[:, 0] > 0, corners_2d[:, 0] < imsize[1]
        )
        visible = np.logical_and(visible, corners_2d[:, 1] < imsize[0])
        visible = np.logical_and(visible, corners_2d[:, 1] > 0)
        visible = np.logical_and(visible, corners_3d[:, 2] > 1)
        return any(visible)

    decoded_corners_3d, _, _ = bbox_to_corner(bboxes, score_thresh)
    img_bboxes = []
    for corners_3d in decoded_corners_3d:
        nbr_points = corners_3d.shape[0]
        corners_3d = np.transpose(corners_3d, (1, 0))
        corners_3d = np.concatenate((corners_3d, np.ones((1, nbr_points))))
        corners_3d = np.dot(ego2img, corners_3d)
        corners_2d = corners_3d[:3, :]
        corners_2d = corners_2d / corners_2d[2:3, :].repeat(3, 0).reshape(
            3, nbr_points
        )
        corners_3d = np.transpose(corners_3d, (1, 0))
        corners_2d = np.transpose(corners_2d[:2, :], (1, 0))
        if not _is_visible(corners_2d, corners_3d, (img_size)):
            continue
        corners_2d[..., 0] = np.clip(corners_2d[..., 0], 0, img_size[1] - 1)
        corners_2d[..., 1] = np.clip(corners_2d[..., 1], 0, img_size[0] - 1)
        img_bboxes.append(corners_2d)
    return img_bboxes
