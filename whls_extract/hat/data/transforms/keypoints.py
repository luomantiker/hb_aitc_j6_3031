import math
from typing import Tuple

import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = ["GenerateHeatmapTarget", "RandomPadLdmkData"]


@OBJECT_REGISTRY.register
class RandomPadLdmkData(object):
    """
    RandomPadLdmkData is a class for randomly padding landmark data.

    Args:
        size: The target size for padding.
        random: Whether to apply random padding. Defaults to True.
    """

    def __init__(self, size: Tuple[int], random: bool = True):
        self.size = size
        self.random = random

    def __call__(self, data):
        img = data["img"]
        if "layout" not in data or data["layout"] == "hwc":
            h, w, c = img.shape
        else:
            c, h, w = img.shape

        pad_h = self.size[0] - h
        pad_w = self.size[1] - w
        if self.random:
            up_pad = np.random.randint(pad_h + 1)
            left_pad = np.random.randint(pad_w + 1)
        else:
            up_pad = 0
            left_pad = 0
        if "gt_lmdk" in data:
            data["gt_ldmk"][:, 0] = data["gt_ldmk"][:, 0] + left_pad
            data["gt_ldmk"][:, 1] = data["gt_ldmk"][:, 1] + up_pad

        H, W = self.size
        if "layout" not in data or data["layout"] == "hwc":
            new_img = np.zeros([H, W, c]).astype(img.dtype)
            new_img[up_pad : up_pad + h, left_pad : left_pad + w] = img
        else:
            new_img = np.zeros([c, H, W]).astype(img.dtype)
            new_img[:, up_pad : up_pad + h, left_pad : left_pad + w] = img

        data["img"] = new_img

        return data


@OBJECT_REGISTRY.register
class AddGaussianNoise(object):
    """Generate gaussian noise on img.

    Args:
        prob: Prob to generate gaussian noise.
        mean: Mean of gaussian distribution. Defaults to 0.
        sigma: Sigma of gaussian distribution. Defaults to 2.
    """

    def __init__(
        self,
        prob: float,
        mean: float = 0,
        sigma: float = 2,
    ):
        self.prob = prob
        self.mean = mean
        self.sigma = sigma

    def __call__(self, data):
        assert "img" in data
        img = data["img"]
        if np.random.random() <= self.prob:
            noise = np.random.normal(
                self.mean, self.sigma, (img.shape[0], img.shape[1])
            )
            noise = noise[:, :, np.newaxis]
            img = img + noise
            img = np.clip(img, 0, 255)
        data["img"] = img.astype(np.uint8)
        return data


@OBJECT_REGISTRY.register
class GenerateHeatmapTarget(object):
    """GenerateHeatmapTarget is a class for generating heatmap targets.

        This class generates heatmap targets for a given number of landmarks
        using a Gaussian distribution.

    Args:
        num_ldmk: The number of landmarks.
        feat_stride: The stride of the feature map.
        heatmap_shape: The shape of the heatmap (height, width).
        sigma: The standard deviation for the Gaussian kernel.
    """

    def __init__(
        self,
        num_ldmk: int,
        feat_stride: int,
        heatmap_shape: Tuple[int],
        sigma: float,
    ):
        self.num_ldmk = num_ldmk
        self.feat_stride = feat_stride
        self.sigma = sigma
        self.heatmap_shape = heatmap_shape

    def ldmk2heatmap(self, ldmk):
        h, w = self.heatmap_shape
        heatmap = np.zeros([self.num_ldmk, h, w])
        radius = math.ceil(self.sigma * 3)
        k_size = radius * 2 + 1
        y_idx = np.arange(w)[:, np.newaxis].repeat(h, axis=1)
        x_idx = np.arange(h)[np.newaxis, :].repeat(w, axis=0)

        def gaussian2d(x, y, x0, y0, sigma):
            exp = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            return np.exp(exp)

        for i in range(self.num_ldmk):
            x0, y0 = ldmk[i][:2]
            x0 = x0 / float(self.feat_stride)
            y0 = y0 / float(self.feat_stride)

            xlow = max(math.floor(x0) - radius, 0)
            ylow = max(math.floor(y0) - radius, 0)

            kx = x_idx[ylow : ylow + k_size, xlow : xlow + k_size]
            ky = y_idx[ylow : ylow + k_size, xlow : xlow + k_size]
            kernel = gaussian2d(kx, ky, x0, y0, self.sigma)
            heatmap[i, ylow : ylow + k_size, xlow : xlow + k_size] = kernel

        return heatmap

    def __call__(self, data):
        ldmk = data["gt_ldmk"]
        ldmk_attr = data["gt_ldmk_attr"]
        heatmap = self.ldmk2heatmap(ldmk)

        heatmap_weight = np.ones_like(heatmap)
        # set different pixel weight
        heatmap_weight[heatmap == 0] = 0.1
        # ignore empty heatmap
        heatmap_weight[~np.any(heatmap, axis=(1, 2)), :, :] = 0
        # ignore invalid landmark
        heatmap_weight[np.where(ldmk_attr < 0), :, :] = 0
        data["gt_heatmap"] = heatmap
        data["gt_heatmap_weight"] = heatmap_weight
        return data
