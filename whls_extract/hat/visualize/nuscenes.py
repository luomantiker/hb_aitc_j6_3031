# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
import os
from typing import Optional, Tuple, Type

import numpy as np
from matplotlib import pyplot as plt

from hat.core.nus_box3d_utils import bbox_ego2bev, bbox_ego2img, bbox_to_corner
from hat.registry import OBJECT_REGISTRY

__all__ = ["NuscenesViz", "NuscenesMultitaskViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class NuscenesViz(object):
    """
    The visiualize method of Nuscenes result.

    Args:
        bev_size: Size of Bev size.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
        use_bce: Whether use bce as segmentation result.
    """

    def __init__(
        self,
        bev_size: Tuple[float, float, float] = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        use_bce: bool = False,
    ):
        self.bev_size = bev_size
        self.score_thresh = score_thresh
        self.is_plot = is_plot
        self.use_bce = use_bce
        self.sample_idx = 0

    def draw_bboxes(self, ax, bboxes, colors=("b", "r", "k"), linewidth=2):
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                # corner = selected_corners[:, i]
                ax.plot(
                    [prev[0], corner[0]],
                    [prev[1], corner[1]],
                    color=color,
                    linewidth=linewidth,
                )
                prev = corner

        for bbox in bboxes:
            # Draw the sides
            for i in range(4):
                ax.plot(
                    [bbox[i][0], bbox[i + 4][0]],
                    [bbox[i][1], bbox[i + 4][1]],
                    color=colors[2],
                    linewidth=linewidth,
                )
            draw_rect(bbox[:4], colors[0])
            draw_rect(bbox[4:], colors[1])

            center_bottom_forward = np.mean(bbox[2:4], axis=0)
            center_bottom = np.mean(bbox[[2, 3, 7, 6]], axis=0)
            ax.plot(
                [center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0],
                linewidth=linewidth,
            )

    def draw_cam_img(self, axes, imgs, homos, bboxes):
        for i, img in enumerate(imgs):
            name = img["name"]
            image = img["img"]
            ax = axes[i]
            ax.set_title(name)
            ax.grid(False)
            image = image.numpy().squeeze()
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            img_bbox = bbox_ego2img(
                bboxes, homos[i], image.shape[:2], self.score_thresh
            )
            self.draw_bboxes(ax, img_bbox)
            ax.imshow(image)

    def create_pascal_label_colormap(self):
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap

    def label_to_color_image(self, label):
        if label.ndim != 2:
            raise ValueError("Expect 2-D input label")

        colormap = self.create_pascal_label_colormap()
        if np.max(label) >= len(colormap):
            raise ValueError("label value too large.")

        return colormap[label]

    def draw_mask(self, ax, mask):
        mask = mask.cpu().numpy()
        mask_img = self.label_to_color_image(mask)
        ax.grid(False)

        ax.imshow(mask_img, origin="lower")

    def draw_bev_bboxes(self, ax, bboxes):
        W = int(self.bev_size[1] * 2 / self.bev_size[2])
        H = int(self.bev_size[0] * 2 / self.bev_size[2])

        self.draw_bboxes(ax, bboxes)
        ax.set_ylim(ymin=0, ymax=H)
        ax.set_xlim(xmin=0, xmax=W)
        ax.grid(False)

    def draw_ego_bboxes(self, ax, bboxes):

        self.draw_bboxes(ax, bboxes)
        ax.grid(False)

    def __call__(self, imgs, preds, meta, save_path=None):
        num_imgs = len(imgs)
        n = num_imgs

        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
            n += 1
        else:
            mask = None

        if "ego2img" in meta:
            homos = meta["ego2img"]
        else:
            homos = meta["lidar2img"]
        if len(bboxes) != 0:
            n += 1

        cols = 3
        fig, axes = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(16, 24)
        )
        axes = axes.flatten()
        self.draw_cam_img(axes[:6], imgs, homos, bboxes)

        cur_index = 6
        if len(bboxes) != 0:
            if "bev_det" in preds:
                bev_bboxes = bbox_ego2bev(bboxes, self.bev_size)
                bev_bboxes, _, _ = bbox_to_corner(bev_bboxes)
                self.draw_bev_bboxes(axes[cur_index], bev_bboxes)

            else:
                ego_bboxes, _, _ = bbox_to_corner(bboxes.cpu().numpy())
                self.draw_ego_bboxes(axes[cur_index], ego_bboxes)
            cur_index += 1
        if mask is not None:
            self.draw_mask(axes[cur_index], mask)
            cur_index += 1

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(
                    save_path, f"nusc_pred_{self.sample_idx}.png"
                )
                plt.savefig(result_path)
            else:
                plt.show()
        self.sample_idx += 1


@OBJECT_REGISTRY.register
class NuscenesMultitaskViz(NuscenesViz):
    """
    The visiualize method of Nuscenes for multitasks.

    Args:
        occ_viz: The occ visiualize module.
        bev_size: Size of Bev size.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
        use_bce: Whether use bce as segmentation result.
    """

    def __init__(
        self,
        occ_viz: Optional[Type] = None,
        bev_size: Tuple[float, float, float] = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        use_bce: bool = False,
    ):
        self.occ_viz = occ_viz
        super().__init__(bev_size, score_thresh, is_plot=True, use_bce=use_bce)

    def plot_points(self, ax, points: np.ndarray, reverse: bool = False):
        ax.set_aspect("equal")
        if reverse:
            ax.plot(
                points[:, 1],
                -points[:, 0],
                "orchid",
                alpha=0.3,
                linestyle="none",
                marker=".",
                markersize=1,
            )
        else:
            ax.plot(
                -points[:, 1],
                points[:, 0],
                "orchid",
                alpha=0.3,
                linestyle="none",
                marker=".",
                markersize=1,
            )

    def __call__(self, imgs, points, preds, meta, save_path=None):
        num_imgs = len(imgs)
        n = num_imgs + 1
        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []
        cols = 3
        homos = meta["lidar2img"]

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
            n += 1
        else:
            mask = None

        fig, axes = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(30, 30)
        )
        axes = axes.flatten()
        new_bboxes = copy.deepcopy(bboxes)
        new_bboxes[:, 3] = bboxes[:, 4]
        new_bboxes[:, 4] = bboxes[:, 3]
        bboxes = new_bboxes
        self.draw_cam_img(axes[:6], imgs, homos, bboxes)

        cur_index = 6
        if len(bboxes) != 0:
            ego_bboxes, _, _ = bbox_to_corner(bboxes.cpu().numpy())
            for box in ego_bboxes:
                x, y = copy.deepcopy(box[:, 0]), copy.deepcopy(box[:, 1])
                box[:, 1], box[:, 0] = -x, y
            axes[cur_index].set_aspect("equal")
            self.draw_ego_bboxes(axes[cur_index], ego_bboxes)
            cur_index += 1
        if mask is not None:
            self.draw_mask(axes[cur_index], mask)
            cur_index += 1

        self.plot_points(axes[cur_index], points[0], reverse=True)
        self.draw_ego_bboxes(axes[cur_index], ego_bboxes)
        cur_index += 1

        if "occ_det" in preds:
            occ_pred = preds["occ_det"]
            vis_result = self.occ_viz.vis_occ(occ_pred)
            vis_result = np.rot90(vis_result, k=2)
            axes[cur_index].imshow(vis_result.astype(np.uint8))
            cur_index += 1

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(
                    save_path, f"nusc_pred_{self.sample_idx}.png"
                )
                plt.savefig(result_path)
            else:
                plt.show()
        self.sample_idx += 1
