# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from hat.registry import OBJECT_REGISTRY

__all__ = ["NuscenesMapViz"]

logger = logging.getLogger(__name__)

# get color map: divider->orange, ped->b, boundary->r, sd map->g
colors_plt = ["orange", "b", "r", "g"]


@OBJECT_REGISTRY.register
class NuscenesMapViz(object):
    """
    The visiualize method of Nuscenes result.

    Args:
        pc_range: Size of pc_range.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
        car_img_path: Path of car img showed in bev coords.
        use_lidar: Whether to use lidar coords.
    """

    def __init__(
        self,
        pc_range: Tuple[float, float, float] = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        car_img_path: str = None,
        use_lidar: bool = True,
    ):
        self.pc_range = pc_range
        self.score_thresh = score_thresh
        self.is_plot = is_plot

        self.car_img = None
        if car_img_path is not None:
            car_img = Image.open(car_img_path)
            self.car_img = car_img

        self.gt_name = "GT_fixednum_pts_MAP"
        self.gt_sd_name = "GT_SD_pts_MAP"
        self.pred_name = "PRED_MAP_plot"
        self.use_lidar = use_lidar

    def draw_cam_img(self, axes, imgs):
        # surrounding view
        for i, img in enumerate(imgs):
            name = img["name"]
            image = img["img"]
            ax = axes[i]
            ax.set_title(name)
            ax.grid(False)
            image = image.numpy().squeeze()
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

            ax.imshow(image)

    def draw_ego_lines(self, ax, instances, labels, name):

        ax.set_xlim(self.pc_range[0], self.pc_range[3])
        ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.grid(False)
        # ax.axis('off')
        ax.set_title(name)

        lines_pts = instances
        for points, label in zip(lines_pts, labels):
            pts = points.numpy()
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])

            ax.plot(
                x,
                y,
                color=colors_plt[label],
                linewidth=1,
                alpha=0.8,
                zorder=-1,
            )
            ax.scatter(
                x, y, color=colors_plt[label], s=2, alpha=0.8, zorder=-1
            )

        if self.car_img is not None:
            ax.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])

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

    def draw_mask(self, ax, mask, name):
        mask = mask.squeeze(0).cpu().numpy().astype(np.int32)
        mask_img = self.label_to_color_image(mask)
        ax.grid(False)
        ax.set_title(name)

        ax.imshow(mask_img, origin="lower")

    def __call__(self, imgs, preds=None, gts=None, save_path=None):

        num_imgs = len(imgs)
        n = num_imgs

        if preds is not None:
            n += 1
        osm_mask = None
        gt_seg_mask = None
        if gts is not None:
            n += 1
            if "osm_mask" in gts:
                osm_mask = gts["osm_mask"]
                n += 1
            if "gt_seg_mask" in gts:
                gt_seg_mask = gts["gt_seg_mask"]
                n += 1

        cols = 3
        fig, axes = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(16, 24)
        )
        axes = axes.flatten()

        self.draw_cam_img(axes[:num_imgs], imgs)

        cur_index = num_imgs
        if gts is not None:
            gt_labels = gts["gt_labels_map"]
            gt_ins = gts["gt_instances"].fixed_num_sampled_points
            self.draw_ego_lines(
                axes[cur_index], gt_ins, gt_labels, self.gt_name
            )
            cur_index += 1

        if osm_mask is not None:
            self.draw_mask(axes[cur_index], osm_mask, "osm_mask")
            cur_index += 1
        if gt_seg_mask is not None:
            self.draw_mask(axes[cur_index], gt_seg_mask, "gt_seg_mask")
            cur_index += 1

        if preds is not None:
            pred_labels = preds[0]["labels_3d"]
            pred_ins = preds[0]["pts_3d"]
            self.draw_ego_lines(
                axes[cur_index], pred_ins, pred_labels, self.pred_name
            )
            cur_index += 1

        for i in range(cur_index, len(axes)):
            ax = axes[i]
            ax.grid(False)
            ax.axis("off")

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "nusc_map_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
