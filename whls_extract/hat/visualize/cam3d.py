import copy
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from hat.core.box3d_utils import points_cam2img
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.registry import OBJECT_REGISTRY


def show_proj_det_result_meshlab(
    data: Dict,
    result: List,
    out_dir: str,
    score_thr=0.0,
):
    """Show result of projecting 3D bbox to 2D image by meshlab."""

    assert "img" in data.keys(), "image data is not provided for visualization"

    img_filename = data["filename"]
    file_name = os.path.split(img_filename)[-1].split(".")[0]

    img = data["img"]
    pred_bboxes = result[0]["bboxes"].tensor.cpu().numpy()
    pred_scores = result[0]["scores"].cpu().numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    if "cam2img" not in data:
        raise NotImplementedError("camera intrinsic matrix is not provided")

    show_bboxes = CameraInstance3DBoxes(
        pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5)
    )

    show_multi_modality_result(
        img,
        show_bboxes,
        data["cam2img"][0],
        file_name,
        out_dir,
    )
    return file_name


def show_multi_modality_result(
    img: np.ndarray,
    pred_bboxes: CameraInstance3DBoxes,
    proj_mat: np.ndarray,
    filename: str,
    out_dir: str = None,
    pred_bbox_color: Tuple[int] = (241, 101, 72),
):
    """Convert multi-modality detection results into 2D results.

    img: The numpy array of image in cv2 fashion.
    pred_bboxes: Predicted boxes.
    proj_mat: The projection matrix.
        according to the camera intrinsic parameters.
    out_dir: Path of output directory.
    filename: Filename of the current frame.
    pred_bbox_color: Color of bbox lines.
        The tuple of color should be in BGR order.
    """
    draw_bbox = draw_camera_bbox3d_on_img

    pred_img = draw_bbox(pred_bboxes, img, proj_mat, color=pred_bbox_color)
    if out_dir is not None:
        print(out_dir, filename)
        result_path = os.path.join(out_dir, filename)
        os.makedirs(result_path, exist_ok=True)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(result_path, "img.png"))
        pred_img_pil = Image.fromarray(pred_img)
        pred_img_pil.save(os.path.join(result_path, "pred.png"))
    else:
        plt.imshow(pred_img)


def draw_camera_bbox3d_on_img(
    bboxes3d: CameraInstance3DBoxes,
    raw_img: np.ndarray,
    cam2img: dict,
    color: Tuple[int] = (0, 255, 0),
    thickness: int = 1,
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d: 3d bbox in camera coordinate system to visualize.
        raw_img: The numpy array of image.
        cam2img: Camera intrinsic matrix.
        color: The color to draw bboxes.
        thickness: The thickness of bboxes.
    """
    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert cam2img.shape == torch.Size([3, 3]) or cam2img.shape == torch.Size(
        [4, 4]
    )
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def plot_rect3d_on_img(
    img: np.ndarray,
    num_rects: int,
    rect_corners: np.array,
    color: Tuple[int] = (0, 255, 0),
    thickness: int = 1,
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img: The numpy array of image.
        num_rects: Number of 3D rectangulars.
        rect_corners: Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color: The color to draw bboxes.
        thickness: The thickness of bboxes.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in line_indices:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

    return img.astype(np.uint8)


@OBJECT_REGISTRY.register
class Cam3dViz(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, data, results, save_path, score_thr):
        file_name = show_proj_det_result_meshlab(
            data=data,
            result=results,
            out_dir=save_path,
            score_thr=score_thr,
        )
        return file_name
