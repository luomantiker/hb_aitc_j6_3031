# Copyright (c) Horizon Robotics, all rights reserved.
# Related operations of label making of anchor_free.
from __future__ import absolute_import, division, print_function
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from .circle_nms_jit import circle_nms


def transform_preds(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[int],
) -> np.ndarray:
    """Transform for pred's coordinates.

    Args:
        coords:(shape=[H, W]).
        center:(shape=[3]) center of transform.
        scale: transform scale.
        output_size:(shape=[H, W]).

    Returns:
        target_coords:(shape=[H, W]).
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=True)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
    center: np.ndarray,
    scale: float,
    rot: float,
    output_size: np.ndarray,
    shift: np.ndarray = None,
    inv: bool = False,
) -> np.ndarray:
    """Get Affine transform parameters.

    Args:
        center:(shape=[3]) center of transform.
        scale: transform scale.
        rot: rotate angle of transorm.
        output_size:(shape=[H, W]).
        shift:(shape=[2])center point shift value.
        inv: if need transorm inv.

    Returns:
        trans:(shape=[3, 3]).
    """
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_sat([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Merge affine transform matrix pt and t.

    Args:
        pt:(shape=[2,1]).
        t:(shape=[3,1]) center of transform.

    Returns:
        new_pt:(shape=[2, 1]).
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the point of diagonal quadrant.

    Point of diagonal quadrant.of point A based on point B.

    Args:
        a:(shape=[2]).
        b:(shape=[2]).

    Returns:
        point:(shape=[2]).
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_sat(src_point: np.ndarray, rot_rad: float):
    """Get Supplementary Angle Theorem.

    Base on origin point.

    Args:
        src_point:(shape=[2]).
        rot_rad:rot angle.

    Returns:
        src_result.
    """

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(
    img: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: np.ndarray,
    rot: float = 0,
) -> np.ndarray:
    """Crop transform for img.

    Args:
        img:(shape=[H, W]).
        center:(shape=[3]) center of transform.
        scale: transform scale.
        output_size:(shape=[H, W]).
        rot: rotate angle.

    Returns:
        dst_img:(shape=[H, W]).
    """
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img,
        trans,
        (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR,
    )

    return dst_img


def gaussian_radius(det_size: List[int], min_overlap: float = 0.5) -> float:
    """Get the gaussian radius by size and min_overlap.

    Args:
        det_size: the size of guassian.
        min_overlap: the overlap of guassian.

    Returns:
        radius: radius of gaussian.
    """

    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def simple_radius(det_size: List[int]) -> float:
    """Get radius by size.

    Args:
        det_size: the size of guassian.

    Returns:
        radius.
    """
    height, width = det_size
    return np.sqrt(height * width)


def gaussian2D(shape: List[int], sigma: float = 1) -> np.ndarray:
    """Get gaussian value by shape and sigma.

    Args:
        shape: the size of guassian.

    Returns:
        h:(shape=[shape]) gaussian value map.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(
    heatmap: np.ndarray,
    center: np.ndarray,
    radius: float,
    k: float = 1,
    return_gaussian: bool = False,
) -> np.ndarray:
    """Draw gaussian map.

    Args:
        heatmap:(shape=[H, W]).
        center:(shape=[>=2]).
        radius: the gaussian radius.
        k: Multiple of masked_gaussian. Defaults to 1.

    Returns:
        heatmap:(shape=[H, W]).
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if (
        min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    ):  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    if return_gaussian:
        valid_mask = np.zeros_like(gaussian)
        valid_mask[
            radius - top : radius + bottom, radius - left : radius + right
        ] = 1

        return heatmap, masked_gaussian, valid_mask

    return heatmap


def draw_umich_gaussian_torch(
    heatmap: torch.Tensor, center: torch.Tensor, radius: float, k: float = 1
) -> torch.Tensor:
    """Draw gaussian map in torch tensor.

    Args:
        heatmap: Heatmap to be masked.
        center: Center coord of the heatmap.
        radius: Radius of gaussian.
        k: Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
    ).to(heatmap.device, torch.float32)
    if (
        min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    ):  # TODO debug
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(
    regmap: np.ndarray,
    heatmap: np.ndarray,
    center: np.ndarray,
    value: np.ndarray,
    radius: float,
    is_offset: bool = False,
) -> np.ndarray:
    """Draw densereg map.

    Args:
        regmap:(shape=[H, W]).
        heatmap:(shape=[H, W]).
        center:(shape=[3]).
        value(shape=[H, W]).
        radius: the gaussian radius.

    Returns:
        reg:(shape=[H, W]) densereg map.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = (
        np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32)
        * value
    )
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_regmap = regmap[:, y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    masked_reg = reg[
        :, radius - top : radius + bottom, radius - left : radius + right
    ]
    if (
        min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    ):  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top : y + bottom, x - left : x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(
    heatmap: np.ndarray, center: np.ndarray, sigma: float
) -> np.ndarray:
    """Draw msra_gaussian map.

    Args:
        heatmap:(shape=[H, W]).
        center:(shape=[3]).
        sigma: the gaussian sigma.

    Returns:
        heatmap:(shape=[H, W]).
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap


def lighting_img(
    data_rng: np.ndarray,
    image: np.ndarray,
    alphastd: float,
    eigval: np.ndarray,
    eigvec: np.ndarray,
) -> np.ndarray:
    """Make image be light.

    Args:
        data_rng:(shape=[3]).
        image:(shape=[H, W]).
        eigval:(shape=[3]).
        eigvec:(shape=[3]).
    """
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_img(alpha: float, image1: np.ndarray, image2: np.ndarray):
    """Make 2 image blend.

    Args:
        image1:(shape=[H, W]).
        image2:(shape=[H, W]).
        alpha:blend alpha on image1.
    """
    image1 *= alpha
    image2 *= 1 - alpha
    image1 += image2


def saturation_img(
    data_rng: np.ndarray,
    image: np.ndarray,
    gs: np.ndarray,
    gs_mean: np.ndarray,
    var: float,
):
    """Make image saturation.

    Args:
        data_rng:(shape=[3]).
        image:(shape=[H, W]).
        var:the saturation variance.
        gs:(shape=[3]).
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_img(alpha, image, gs[:, :, None])


def brightness_img(
    data_rng: np.ndarray,
    image: np.ndarray,
    gs: np.ndarray,
    gs_mean: np.ndarray,
    var: float,
):
    """Make image brightness.

    Args:
        data_rng:(shape=[3]).
        image:(shape=[H, W]).
        var:the brightness variance.
        gs:(shape=[3]).
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_img(
    data_rng: np.ndarray,
    image: np.ndarray,
    gs: np.ndarray,
    gs_mean: np.ndarray,
    var: float,
):
    """Make image contrast.

    Args:
        data_rng:(shape=[3])
        image:(shape=[H, W]).
        var:the contrast variance.
        gs:(shape=[3]).
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_img(alpha, image, gs_mean)


def color_aug(
    data_rng: np.ndarray,
    image: np.ndarray,
    eig_val: np.ndarray,
    eig_vec: np.ndarray,
):
    """Make image augumentation.

    Args:
        data_rng:(shape=[3]).
        image:(shape=[H, W]).
        eigval:(shape=[3]).
        eigvec:(shape=[3]).
    """
    functions = [brightness_img, contrast_img, saturation_img]
    random.shuffle(functions)

    gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_img(data_rng, image, 0.1, eig_val, eig_vec)


def _gather_feat(feat: torch.Tensor, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat: torch.Tensor, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat: torch.Tensor, kernel: int = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad
    )
    keep = (hmax == heat).float()
    return heat * keep


def _circle_nms(boxes: np.ndarray, min_radius: float, post_max_size: int = 83):
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[
        :post_max_size
    ]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


def _topk(scores: torch.Tensor, K: int = 40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def bilinear_interpolate_torch(
    im: torch.Tensor, x: torch.Tensor, y: torch.Tensor
):
    """Bilinear interpolate torch.

    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = (
        torch.t((torch.t(Ia) * wa))
        + torch.t(torch.t(Ib) * wb)
        + torch.t(torch.t(Ic) * wc)
        + torch.t(torch.t(Id) * wd)
    )

    return ans
