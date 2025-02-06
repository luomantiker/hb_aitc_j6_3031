# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping, Optional, Sequence, Tuple

import cv2
import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn

from hat.core.box3d_utils import points_cam2img, points_img2cam
from hat.core.cam_box3d import CameraInstance3DBoxes
from hat.core.nus_box3d_utils import get_min_max_coords
from hat.registry import OBJECT_REGISTRY
from .grid_mask import GridMask

__all__ = [
    "MultiViewsImgResize",
    "MultiViewsImgCrop",
    "MultiViewsImgRotate",
    "MultiViewsImgFlip",
    "MultiViewsImgTransformWrapper",
    "MultiViewsGridMask",
    "BevFeatureFlip",
    "BevFeatureRotate",
    "MultiViewsSpiltImgTransformWrapper",
    "MultiViewsPhotoMetricDistortion",
]

PIL_INTERP_CODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
}


@OBJECT_REGISTRY.register
class MultiViewsSpiltImgTransformWrapper(object):
    """Wrapper split img transform for image inputs.

    Args:
       trnsforms: List of image transforms.
    """

    def __init__(self, transforms: Sequence[nn.Module], numsplit: int = 3):

        self.transforms = transforms
        self.numsplit = numsplit

    def __call__(self, data: Mapping):
        c, h, w = data["img"].shape
        assert c % self.numsplit == 0
        div_n = c // self.numsplit
        data["img"] = data["img"].reshape(div_n, self.numsplit, h, w)
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        return "MultiViewsSpiltImgTransformWrapper"


@OBJECT_REGISTRY.register
class MultiViewsImgTransformWrapper(object):
    """Wrapper img transform for image inputs.

    Args:
       trnsforms: List of image transforms.
    """

    def __init__(self, transforms: Sequence[nn.Module]):
        self.transforms = transforms

    def __call__(self, data: Mapping):
        for i, img in enumerate(data["img"]):
            new_data = {
                "img": img,
                "layout": data["layout"],
                "color_space": data["color_space"],
            }
            for transform in self.transforms:
                new_data = transform(new_data)
            data["img"][i] = new_data["img"]
        return data

    def __repr__(self):
        return "MultiViewsImgTransformWrapper"


@OBJECT_REGISTRY.register
class MultiViewsImgResize(object):
    """Resize PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        scales: Scale for random choosen.
        interpolation:Desired interpolation. Default is 'nearest'.
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scales: Optional[Tuple[float, float]] = None,
        interpolation: str = "bilinear",
    ):
        self.size = size
        self.scales = scales
        assert interpolation in PIL_INTERP_CODES
        self.interpolation = interpolation

    def _resize(self, data: Image.Image, size):
        return F.resize(data, size, PIL_INTERP_CODES[self.interpolation])

    def _resize_mat(self, mat, scale):
        view = np.eye(4)
        view[0, 0] = scale[1]
        view[1, 1] = scale[0]
        mat = view @ mat
        return mat

    def get_aug(self):
        if self.scales is not None:
            resize = np.random.uniform(*self.scales)
            return "resize", resize
        else:
            return "resize", 1.0

    def __call__(self, data: Mapping):
        if self.size:
            if not isinstance(self.size[0], Sequence):
                self.size = [self.size]
            if len(self.size) == 1:
                sizes = self.size * len(data["img"])
            elif len(self.size) == len(data["img"]):
                sizes = self.size
            else:
                raise ValueError("Size should equal to img num or 1")
        elif self.scales:
            if "scenes_aug" in data and "resize" in data["scenes_aug"]:
                resize = data["scenes_aug"]["resize"]
            else:
                _, resize = self.get_aug()
            sizes = []
            for img in data["img"]:
                W, H = img.size
                resize_dims = (int(H * resize), int(W * resize))
                sizes.append(resize_dims)
        else:
            raise ValueError("Size or scale should be set")
        for i, img in enumerate(data["img"]):
            scale = [sizes[i][0] / img.size[1], sizes[i][1] / img.size[0]]
            data["img"][i] = self._resize(img, sizes[i])
            if "ego2img" in data:
                mat = self._resize_mat(data["ego2img"][i], scale)
                data["ego2img"][i] = mat
            if "lidar2img" in data:
                mat = self._resize_mat(data["lidar2img"][i], scale)
                data["lidar2img"][i] = mat
            if "corner2ds" in data and len(data["corner2ds"][i]) > 0:
                data["corner2ds"][i][..., [0, 2]] = (
                    data["corner2ds"][i][..., [0, 2]] * scale[1]
                )
                data["corner2ds"][i][..., [1, 3]] = (
                    data["corner2ds"][i][..., [1, 3]] * scale[0]
                )

            if "center2ds" in data and len(data["center2ds"][i]) > 0:
                data["center2ds"][i][..., 0] = (
                    data["center2ds"][i][..., 0] * scale[1]
                )
                data["center2ds"][i][..., 1] = (
                    data["center2ds"][i][..., 1] * scale[0]
                )
        if "camera_intrinsic" in data:
            data["camera_intrinsic"][:, 0] = (
                data["camera_intrinsic"][:, 0] * scale[1]
            )
            data["camera_intrinsic"][:, 1] = (
                data["camera_intrinsic"][:, 1] * scale[0]
            )
        return data

    def __repr__(self):
        return "MultiViewsImgResize"


@OBJECT_REGISTRY.register
class MultiViewsImgCrop(object):
    """Crop PIL Images to the given size and modify intrinsics.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        random: Whether choosing min x randomly.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        random: bool = False,
    ):
        self.size = size if isinstance(size[0], Sequence) else [size]
        self.random = random

    def _crop(self, data: Image.Image, top, left, height, width):
        return F.crop(data, top, left, height, width)

    def _crop_mat(self, mat, left, top):
        view = np.eye(4)
        view[0, 2] = -left
        view[1, 2] = -top
        mat = view @ mat
        return mat

    def __call__(self, data: Mapping):
        if len(self.size) == 1:
            sizes = self.size * len(data["img"])
        elif len(self.size) == len(data["img"]):
            sizes = self.size
        else:
            raise ValueError("Size should equal to img num or 1")

        for i, img in enumerate(data["img"]):
            size = sizes[i]
            top = img.size[1] - size[0]
            if self.random:
                left = int(np.random.uniform(0, max(0, img.size[0] - size[1])))
            else:
                left = (img.size[0] - size[1]) / 2
            data["img"][i] = self._crop(img, top, left, size[0], size[1])
            if "ego2img" in data:
                mat = self._crop_mat(data["ego2img"][i], left, top)
                data["ego2img"][i] = mat
            if "lidar2img" in data:
                mat = self._crop_mat(data["lidar2img"][i], left, top)
                data["lidar2img"][i] = mat
            if "corner2ds" in data and len(data["corner2ds"][i]) > 0:
                data["corner2ds"][i][..., 0] = (
                    data["corner2ds"][i][..., 0] - left
                )
                data["corner2ds"][i][..., 1] = (
                    data["corner2ds"][i][..., 1] - top
                )
                data["corner2ds"][i][..., 2] = (
                    data["corner2ds"][i][..., 2] - left
                )
                data["corner2ds"][i][..., 3] = (
                    data["corner2ds"][i][..., 3] - top
                )

            if "center2ds" in data and len(data["center2ds"][i]) > 0:
                data["center2ds"][i][..., 0] = (
                    data["center2ds"][i][..., 0] - left
                )
                data["center2ds"][i][..., 1] = (
                    data["center2ds"][i][..., 1] - top
                )
            if "camera_intrinsic" in data:
                data["camera_intrinsic"][i][0, 2] = (
                    data["camera_intrinsic"][i][0, 2] - left
                )
                data["camera_intrinsic"][i][1, 2] = (
                    data["camera_intrinsic"][i][1, 2] - top
                )

        return data

    def __repr__(self):
        return "MultiViewsImgCrop"


@OBJECT_REGISTRY.register
class MultiViewsImgFlip(object):
    """Flip PIL Images  and modify intrinsics.

    Args:
        prob: Probility for flip image.
    """

    def __init__(
        self,
        prob: float = 0.5,
    ):
        self.prob = prob

    def _flip_mat(self, mat, size):
        view = np.eye(4)
        view[0, 0] = -1
        view[0, 2] = size[0] - 1
        mat = view @ mat
        return mat

    def get_aug(self):
        flip = np.random.choice([False, True], p=[1 - self.prob, self.prob])
        return "flip", flip

    def __call__(self, data: Mapping):
        if "scenes_aug" in data and "flip" in data["scenes_aug"]:
            flip = data["scenes_aug"]["flip"]
        else:
            _, flip = self.get_aug()
        for i, img in enumerate(data["img"]):
            size = img.size
            if flip:
                data["img"][i] = img.transpose(method=Image.FLIP_LEFT_RIGHT)

                if "ego2img" in data:
                    ego2img = self._flip_mat(data["ego2img"][i], size)
                    data["ego2img"][i] = ego2img
                if "lidar2img" in data:
                    lidar2img = self._flip_mat(data["lidar2img"][i], size)
                    data["lidar2img"][i] = lidar2img
                if "center2ds" in data and len(data["center2ds"][i]) > 0:
                    data["center2ds"][i][..., 0] = (size[0] - 1) - data[
                        "center2ds"
                    ][i][..., 0]
                if "mono_3d_bboxes" in data:
                    centers = data["center2ds"][i]
                    if len(centers) > 0:
                        depth = data["depths"][i].reshape(-1, 1)
                        bbox_center = torch.tensor(
                            np.concatenate([centers, depth], axis=-1)
                        )

                        bbox_center = points_img2cam(
                            bbox_center, data["camera_intrinsic"][i]
                        ).float()
                        data["mono_3d_bboxes"][i].flip()
                        bboxes_3d = data["mono_3d_bboxes"][i].tensor.cpu()
                        bboxes_3d[:, :3] = bbox_center
                        bboxes_3d = CameraInstance3DBoxes(
                            bboxes_3d,
                            box_dim=bboxes_3d.shape[-1],
                            origin=(0.5, 0.5, 0.5),
                        )

                        data["mono_3d_bboxes"][i] = bboxes_3d
                        corners = bboxes_3d.corners
                        corners = (
                            points_cam2img(
                                corners,
                                torch.tensor(
                                    data["camera_intrinsic"][i]
                                ).float(),
                            )
                            .cpu()
                            .numpy()
                        )
                        if corners.shape[0] != 0:
                            data["corner2ds"][i][..., 0] = np.min(
                                corners[..., 0], axis=1
                            )
                            data["corner2ds"][i][..., 1] = np.min(
                                corners[..., 1], axis=1
                            )
                            data["corner2ds"][i][..., 2] = np.max(
                                corners[..., 0], axis=1
                            )
                            data["corner2ds"][i][..., 3] = np.max(
                                corners[..., 1], axis=1
                            )
        return data

    def __repr__(self):
        return "MultiViewsImgFlip"


@OBJECT_REGISTRY.register
class MultiViewsImgRotate(object):
    """Rotate PIL Images.

    Args:
        rot: Rotate angle.
                    print(xmin, xmax)
                    print(xmin, xmax)
    """

    def __init__(
        self,
        rot: Tuple[float, float],
    ):
        self.rot = rot

    def _get_rot(self, rot):
        return torch.Tensor(
            [
                [np.cos(rot), np.sin(rot)],
                [-np.sin(rot), np.cos(rot)],
            ]
        )

    def _rot_mat(self, mat, rot, size):
        view = np.eye(4)
        A = self._get_rot(rot / 180 * np.pi)
        B = torch.Tensor([size[0] - 1, size[1] - 1]) / 2
        view[:2, :2] = A
        view[:2, 2] = A @ -B + B
        if mat.shape[-1] == 3:
            mat = view[:3, :3] @ mat
        else:
            mat = view @ mat
        return mat

    def rotate(self, point, rot, size):
        A = self._get_rot(rot / 180 * np.pi)
        B = torch.Tensor([size[0] - 1, size[1] - 1]) / 2
        trans = A @ -B + B
        point = torch.Tensor(point)
        point = point @ A.T + trans
        return point

    def get_aug(self):
        rot = np.random.uniform(*self.rot)
        return "rot", rot

    def __call__(self, data: Mapping):
        if "scenes_aug" in data and "rot" in data["scenes_aug"]:
            rot = data["scenes_aug"]["rot"]
        else:
            _, rot = self.get_aug()
        for i, img in enumerate(data["img"]):
            data["img"][i] = img.rotate(rot)
            size = img.size
            if "ego2img" in data:
                mat = self._rot_mat(data["ego2img"][i], rot, size)
                data["ego2img"][i] = mat
            if "lidar2img" in data:
                lidar2img = self._rot_mat(data["lidar2img"][i], rot, size)
                data["lidar2img"][i] = lidar2img
            if "center2ds" in data and len(data["center2ds"][i]) > 0:
                data["center2ds"][i] = (
                    self.rotate(data["center2ds"][i], rot, size).cpu().numpy()
                )
            if "camera_intrinsic" in data:
                data["camera_intrinsic"][i] = self._rot_mat(
                    data["camera_intrinsic"][i], rot, size
                )
            if "mono_3d_bboxes" in data:
                # data["mono_3d_bboxes"][i].rotate(rot / 180 * np.pi)
                centers = data["center2ds"][i]
                if len(centers) > 0:
                    depth = data["depths"][i].reshape(-1, 1)
                    bbox_center = torch.tensor(
                        np.concatenate([centers, depth], axis=-1)
                    )

                    bbox_center = points_img2cam(
                        bbox_center, data["camera_intrinsic"][i]
                    ).float()

                    bboxes_3d = data["mono_3d_bboxes"][i].tensor
                    bboxes_3d[:, :3] = bbox_center
                    bboxes_3d = CameraInstance3DBoxes(
                        bboxes_3d,
                        box_dim=bboxes_3d.shape[-1],
                        origin=(0.5, 0.5, 0.5),
                    )

                    data["mono_3d_bboxes"][i] = bboxes_3d
                    corners = bboxes_3d.corners
                    corners = (
                        points_cam2img(
                            corners,
                            torch.tensor(data["camera_intrinsic"][i]).float(),
                        )
                        .cpu()
                        .numpy()
                    )
                    if corners.shape[0] != 0:
                        data["corner2ds"][i][..., 0] = np.min(
                            corners[..., 0], axis=1
                        )
                        data["corner2ds"][i][..., 1] = np.min(
                            corners[..., 1], axis=1
                        )
                        data["corner2ds"][i][..., 2] = np.max(
                            corners[..., 0], axis=1
                        )
                        data["corner2ds"][i][..., 3] = np.max(
                            corners[..., 1], axis=1
                        )
        return data

    def __repr__(self):
        return "MultiViewsImgRotate"


@OBJECT_REGISTRY.register
class MultiViewsGridMask(GridMask):
    """For grid masking augmentation."""

    def __init__(self, **kwargs):
        super(MultiViewsGridMask, self).__init__(**kwargs)

    def __call__(self, data):
        imgs = data["img"]

        for idx, img in enumerate(imgs):
            img = np.asarray(img)
            img = super().__call__(img)
            img = img.astype("uint8")
            img = Image.fromarray(img)
            data["img"][idx] = img
        return data


@OBJECT_REGISTRY.register
class MultiViewsPhotoMetricDistortion(object):
    """
    Apply photometric distortions to an input image.

    This class implements random adjustments to brightness, contrast,
    saturation, and hue of an image, which are commonly used for data
    augmentation in computer vision tasks.

    Args:
        brightness_delta : Maximum delta for adjusting brightness randomly.
        contrast_range : Range for adjusting contrast randomly.
            Default is (0.5, 1.5).
        saturation_range : Range for adjusting saturation randomly.
            Default is (0.5, 1.5).
        hue_delta : Maximum delta for adjusting hue randomly.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float] = (0.5, 1.5),
        saturation_range: Tuple[float] = (0.5, 1.5),
        hue_delta: int = 18,
        use_pil: bool = False,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.use_pil = use_pil

    def _do_augment(self, img):
        img = np.asarray(img).astype(dtype=np.float32)
        if np.random.randint(2):
            delta = np.random.uniform(
                -self.brightness_delta, self.brightness_delta
            )
            img += delta
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(
                    self.contrast_lower, self.contrast_upper
                )
                img *= alpha
            # convert color from BGR to HSV
        if self.use_pil:
            image = Image.fromarray(img.astype(np.uint8), "RGB")
            img = np.array(image.convert("HSV"), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        if self.use_pil:
            img = Image.fromarray(img.astype(np.uint8), "HSV")
            img = np.array(img.convert("RGB"), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(
                    self.contrast_lower, self.contrast_upper
                )
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        return img

    def __call__(self, data: Mapping):
        for i, img in enumerate(data["img"]):
            img = self._do_augment(img)
            img = img.astype("uint8")
            img = Image.fromarray(img)
            data["img"][i] = img
        return data

    def __repr__(self):
        return "MultiViewsPhotoMetricDistortion"


@OBJECT_REGISTRY.register
class BevBBoxRotation(object):
    def __init__(
        self,
        transform_matrix_key=("lidar2img",),
        global_key=("lidar2global",),
        rotation_3d_range=None,
    ):
        self.transform_matrix_key = transform_matrix_key
        self.global_key = global_key
        self.rotation_3d_range = rotation_3d_range

    def get_aug(self):
        angle = np.random.uniform(*self.rotation_3d_range)
        return "angle", angle

    def __call__(self, data):
        if "scenes_aug" in data and "angle" in data["scenes_aug"]:
            angle = data["scenes_aug"]["angle"]
        else:
            _, angle = self.get_aug()
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)
        for key in self.transform_matrix_key:
            assert key in data, f"miss key {key} in data"
            for i, mat in enumerate(data[key]):
                data[key][i] = mat @ rot_mat_inv
        for key in self.global_key:
            assert key in data, f"miss key {key} in data"
            data[key] = data[key] @ rot_mat_inv

        if (
            "lidar_bboxes_labels" in data
            and len(data["lidar_bboxes_labels"]) > 0
        ):
            data["lidar_bboxes_labels"] = self.box_rotate(
                data["lidar_bboxes_labels"], angle
            )
        if "ego_bboxes_labels" in data and len(data["ego_bboxes_labels"]) > 0:
            data["ego_bboxes_labels"] = self.box_rotate(
                data["ego_bboxes_labels"], angle
            )

        if "points" in data:
            data["points"] = self.point_rotate(data["points"], angle)
        return data

    @staticmethod
    def point_rotate(point, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        point[:, :3] = point[:, :3] @ rot_mat_T
        return point

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:9].shape[-1]
            bbox_3d[:, 7:9] = bbox_3d[:, 7:9] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@OBJECT_REGISTRY.register
class BevFeatureRotate(object):
    """Rotate feat.

    Args:
        bev_size: Size of bev view.
        rot: Rotate radian.
    """

    def __init__(
        self,
        bev_size: Tuple[float, float, float],
        rot: Tuple[float, float] = (-0.3925, 0.3925),
    ):
        self.rot = rot
        self.grid_sample = hnn.GridSample(
            mode="bilinear", padding_mode="zeros"
        )
        self.bev_size = bev_size

    def _get_rot(self, rot):
        return torch.Tensor(
            [
                [np.cos(rot), np.sin(rot)],
                [-np.sin(rot), np.cos(rot)],
            ]
        )

    def _get_coords(self, rot, feat):
        H, W = feat.shape[2:]
        view = np.eye(3)
        A = self._get_rot(-rot)
        B = torch.Tensor((W - 1, H - 1)) / 2
        view[:2, :2] = A
        view[:2, 2] = A @ -B + B
        view = torch.Tensor(view)

        x = (torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)).float()
        y = (torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)).float()
        ones = torch.ones((H, W)).float()
        coords = torch.stack([x, y, ones], dim=-1)
        coords = coords.view(1, H, W, 3)
        new_coords = torch.matmul(coords, view.T)[..., :2]
        new_coords -= coords[..., :2]
        return new_coords, view

    def _rotate_bbox(self, bbox, rot):
        min_x, max_x, min_y, max_y = get_min_max_coords(self.bev_size)
        H = max_x * 2 / self.bev_size[2]
        W = max_y * 2 / self.bev_size[2]
        view = np.eye(3)
        A = self._get_rot(rot)
        B = torch.Tensor((W, H)) / 2
        view[:2, :2] = A
        view[:2, 2] = A @ -B + B
        view = torch.Tensor(view)

        bbox = torch.Tensor(bbox)
        center = torch.cat([bbox[:2], torch.ones([1])])
        center = torch.matmul(center, view.T)
        bbox[:2] = center[:2]
        rot = bbox[6] + rot
        bbox[6] = rot
        vel = torch.cat([bbox[7:9], torch.zeros([1])])
        vel = torch.matmul(vel, view.T)
        bbox[7:9] = vel[:2]
        return bbox

    def __call__(self, feats, data: Mapping):
        batch_size = feats.shape[0]
        coords = []
        with torch.no_grad():
            for b in range(batch_size):
                rot = np.random.uniform(*self.rot)
                new_coords, view = self._get_coords(rot, feats)
                new_coords = new_coords.to(device=feats.device)
                coords.append(new_coords)
                if "bev_seg_indices" in data:
                    bev_seg_indices = data["bev_seg_indices"][b : b + 1]
                    bev_seg_indices = bev_seg_indices.unsqueeze(1)
                    new_coords, _ = self._get_coords(rot, bev_seg_indices)
                    new_coords = new_coords.to(device=feats.device)
                    bev_seg_indices = self.grid_sample(
                        bev_seg_indices.float(), new_coords
                    ).int()
                    bev_seg_indices = bev_seg_indices.squeeze()
                    data["bev_seg_indices"][b] = bev_seg_indices
                if "bev_bboxes_labels" in data:
                    for i, bbox in enumerate(data["bev_bboxes_labels"][b]):
                        data["bev_bboxes_labels"][b][i] = self._rotate_bbox(
                            bbox, rot
                        )

        coords = torch.cat(coords)
        feats = self.grid_sample(feats, coords)

        return feats, data

    def __repr__(self):
        return "NuscBevRotate"


@OBJECT_REGISTRY.register
class BevFeatureFlip(object):
    """Flip bev feature.

    Args:
        bev_size: Size of bev view.
        prob_x: Probability for horizontal.
        prob_y: Probability for vertical.
    """

    def __init__(
        self,
        prob_x: float,
        prob_y: float,
        bev_size: Tuple[float, float, float],
    ):
        self.prob_x = prob_x
        self.prob_y = prob_y
        self.bev_size = bev_size

    def _flip_bbox(self, bbox, flip_x, flip_y):
        min_x, max_x, min_y, max_y = get_min_max_coords(self.bev_size)
        H = max_x * 2 / self.bev_size[2]
        W = max_y * 2 / self.bev_size[2]
        if flip_x:
            bbox[0] = W - bbox[0]
            bbox[6] = -bbox[6]
        if flip_y:
            bbox[1] = H - bbox[1]
            bbox[6] = -bbox[6]
        return bbox

    def __call__(self, feats, data: Mapping):
        batch_size = feats.shape[0]
        new_feats = []
        for b in range(batch_size):
            flip_x = np.random.choice(
                [False, True], p=[1 - self.prob_x, self.prob_x]
            )
            dims = []
            if flip_x:
                dims.append(3)
            flip_y = np.random.choice(
                [False, True], p=[1 - self.prob_y, self.prob_y]
            )
            if flip_y:
                dims.append(2)
            feat = feats[b : b + 1]
            feat = torch.flip(feat, dims)
            with torch.no_grad():
                if "bev_seg_indices" in data:
                    bev_seg_indices = data["bev_seg_indices"][b : b + 1]
                    bev_seg_indices = bev_seg_indices.unsqueeze(1)
                    data["bev_seg_indices"][b] = torch.flip(
                        bev_seg_indices, dims
                    ).squeeze()
                if "bev_bboxes_labels" in data:
                    for i, bbox in enumerate(data["bev_bboxes_labels"][b]):
                        data["bev_bboxes_labels"][b][i] = self._flip_bbox(
                            bbox, flip_x, flip_y
                        )
            new_feats.append(feat)
        new_feats = torch.cat(new_feats)
        return new_feats, data

    def __repr__(self):
        return "BevFeatureRotate"
