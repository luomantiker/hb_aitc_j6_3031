from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ImageAugmentation",
    "BevFeatureAug",
]


@OBJECT_REGISTRY.register
class ImageAugmentation(object):
    """Augment PIL Images according to the given data_config.

    Args:
        is_train: if it is for training. default False.
        data_config: Dictionary containing data augmentation transformations,
                    such as resize, crop, flip, etc .
    """

    def __init__(
        self,
        data_config: dict,
        is_train: bool = False,
    ):
        self.is_train = is_train
        self.data_config = data_config

    def sample_augmentation(
        self,
        H: int,
        W: int,
        flip: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        """Sample augmentation.

        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resizeratio,float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: Random rotation angle,float
        """
        fH, fW = self.data_config["input_size"]
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(
                *self.data_config["resize"]
            )  # resize ratio in [fW/W − 0.06, fW/W + 0.11].
            resize_dims = (
                int(W * resize),
                int(H * resize),
            )  # size after resize
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_config["crop_h"])) * newH
                )
                - fH
            )  # s * H - H_in
            crop_w = int(
                np.random.uniform(0, max(0, newW - fW))
            )  # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config["flip"] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config["rot"])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get("resize_test", 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config["crop_h"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def img_transform(
        self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate
    ):
        """Image transform.

        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize ratio.
            resize_dims: Tuple(W, H), size after resize
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def ego2img_add_post(self, ego2img, post_tran, post_rot):
        """Update image enhancement transformation to ego2img matrix.

        Args:
            ego2img: (4, 4)
            post_tran: (3,)
            post_rot:(3,3)
        Returns:
            ego2img: (4, 4)
        """
        viewpad = torch.eye(4)
        viewpad[:3, :3] = post_rot
        viewpad[:2, 2] = post_tran[:2]
        ego2img = viewpad @ ego2img
        return ego2img

    def img_augs_transform(
        self,
        data: Dict,
        flip: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        """Img augmentation transform.

        Args:
            data:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)       N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
            ego2img:     (N_views, 4, 4)
        """
        imgs = data["img"]
        imgs_aug = []
        ego2imgs = []
        for i in range(len(imgs)):
            img = imgs[i]

            # initialize
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale
            )
            resize, resize_dims, crop, flip, rotate = img_augs

            # img:PIL.Image; post_rot:Tensor (2, 2); post_tran:Tensor (2, )
            img, post_rot2, post_tran2 = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # make augmentation matrices 3x3 and update ego2img
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            if "ego2img" in data:
                ego2img = data["ego2img"][i]
                ego2img = self.ego2img_add_post(ego2img, post_tran, post_rot)
                ego2imgs.append(ego2img)

            imgs_aug.append(img)

        return imgs_aug, ego2imgs

    def __call__(self, data):
        data["img"], data["ego2img"] = self.img_augs_transform(data)
        return data


@OBJECT_REGISTRY.register
class BevFeatureAug(object):
    """Augment bev feature.

    Args:
    bda_aug_conf: a dict including augmentation transform.
        ex. bda_aug_conf = dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            )
        rot_lim: Random rotation angle range.
        scale_lim: The range of random scaling, in [0-1].
        flip_dx_ratio: Probability for horizontal.
        flip_dy_ratio: Probability for vertical.
    """

    def __init__(
        self,
        bda_aug_conf: Dict,
        is_train: bool = True,
    ):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf["rot_lim"])
            scale_bda = np.random.uniform(*self.bda_aug_conf["scale_lim"])
            flip_dx = np.random.uniform() < self.bda_aug_conf["flip_dx_ratio"]
            flip_dy = np.random.uniform() < self.bda_aug_conf["flip_dy_ratio"]
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        scale_mat = torch.Tensor(
            [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
        )
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:  # Flip along the y-axis
            flip_mat = flip_mat @ torch.Tensor(
                [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )
        if flip_dy:  # Flip along the x-axis
            flip_mat = flip_mat @ torch.Tensor(
                [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            )
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def __call__(self, data):
        semantics = data["voxel_semantics"]
        mask_lidar = data["mask_lidar"]
        mask_camera = data["mask_camera"]
        (
            rotate_bda,
            scale_bda,
            flip_dx,
            flip_dy,
        ) = self.sample_bda_augmentation()

        bda_mat = np.zeros((4, 4))
        bda_mat[3, 3] = 1
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if "ego2img" in data:
            ego2img = data["ego2img"]
            for i in range(len(ego2img)):
                ego2img[i] = ego2img[i] @ bda_mat

        if flip_dx:
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if flip_dy:
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        data["ego2img"] = ego2img
        data["voxel_semantics"] = semantics
        data["mask_lidar"] = mask_lidar
        data["mask_camera"] = mask_camera

        return data
