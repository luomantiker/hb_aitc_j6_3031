import logging
import math
import os
from typing import Tuple

import numpy as np

try:
    from av2.map.map_api import ArgoverseStaticMap
except Exception:
    ArgoverseStaticMap = None

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["Argoverse2Viz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Argoverse2Viz(object):
    """
    The visiualize method of Argoverse2 result.

    Args:
        num_historical_steps: Number of historical time steps.
        is_plot: Whether to plot image.

    """

    _DRIVABLE_AREA_COLOR = "#7A7A7A"
    _LANE_SEGMENT_COLOR = "#8f8787"  # "E0E0E0"

    _DEFAULT_ACTOR_COLOR = "#D3E8EF"
    _FOCAL_AGENT_COLOR = "#9896f1"  # "FF8C00"
    _AV_COLOR = "#008B8B"
    _PREDICT_AGENT_COLOR = "#f6416c"  # "4B0082"
    _GT_AGENT_COLOR = "#9896f1"  # "FF8C00"
    _LANE_CROSSWALK_COLOR = "#FFA500"

    _ESTIMATED_VEHICLE_LENGTH_M = 4.0
    _ESTIMATED_VEHICLE_WIDTH_M = 2.0
    _ESTIMATED_CYCLIST_LENGTH_M = 2.0
    _ESTIMATED_CYCLIST_WIDTH_M = 0.7

    _BOUNDING_BOX_ZORDER = 100

    @require_packages("av2")
    def __init__(self, num_historical_steps: int, is_plot: bool = True):
        self.is_plot = is_plot
        self.num_historical_steps = num_historical_steps
        self._point_types = [
            "DASH_SOLID_YELLOW",
            "DASH_SOLID_WHITE",
            "DASHED_WHITE",
            "DASHED_YELLOW",
            "DOUBLE_SOLID_YELLOW",
            "DOUBLE_SOLID_WHITE",
            "DOUBLE_DASH_YELLOW",
            "DOUBLE_DASH_WHITE",
            "SOLID_YELLOW",
            "SOLID_WHITE",
            "SOLID_DASH_WHITE",
            "SOLID_DASH_YELLOW",
            "SOLID_BLUE",
            "NONE",
            "UNKNOWN",
            "CROSSWALK",
            "CENTERLINE",
        ]
        self._static_types = {
            "static",
            "background",
            "construction",
            "riderless_bicycle",
        }
        self._agent_types = [
            "vehicle",
            "pedestrian",
            "motorcyclist",
            "cyclist",
            "bus",
            "static",
            "background",
            "construction",
            "riderless_bicycle",
            "unknown",
        ]
        self._agent_categories = [
            "TRACK_FRAGMENT",
            "UNSCORED_TRACK",
            "SCORED_TRACK",
            "FOCAL_TRACK",
        ]
        self._polygon_types = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]

    def _plot_polylines(
        self,
        polylines: list,
        style: str = "-",
        line_width: float = 1.0,
        alpha: float = 1.0,
        color: str = "r",
        arrow: bool = False,
        arrow_length: float = 15.0,
    ) -> None:
        """Plot a group of polylines with the specified config.

        Args:
            polylines: Collection of (N, 2) polylines to plot.
            style: Style of the line to plot
                (e.g. `-` for solid, `--` for dashed)
            line_width: Desired width for the plotted lines.
            alpha: Desired alpha for the plotted lines.
            color: Desired color for the plotted lines.
            arrow: Whether using arrow at the  endpoint of the lines.
            arrow_length: The length of the arrows.
        """
        for polyline in polylines:
            plt.plot(
                polyline[:, 0],
                polyline[:, 1],
                style,
                linewidth=line_width,
                color=color,
                alpha=alpha,
            )
            if arrow:
                end_point = polyline[-1]
                direction = polyline[-1] - polyline[-2]
                direction = direction / np.linalg.norm(direction)

                plt.annotate(
                    "",
                    xy=(end_point[0], end_point[1]),
                    xytext=(
                        polyline[-2, 0],
                        polyline[-2, 1],
                    ),
                    arrowprops={
                        "edgecolor": color,
                        "linewidth": line_width,  # 箭头线条宽度
                        "mutation_scale": arrow_length,
                        # "shrink": 0.05,
                        "arrowstyle": "->",
                        "alpha": alpha,
                        "fill": False,
                    },
                )

    def _plot_polygons(
        self, polygons, alpha: float = 1.0, color: str = "r"
    ) -> None:
        """Plot a group of filled polygons with the specified config.

        Args:
            polygons: Collection of polygons specified by (N,2)
                arrays of vertices.
            alpha: Desired alpha for the polygon fill.
            color: Desired color for the polygon.
        """
        for polygon in polygons:
            plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

    def _plot_actor_bounding_box(
        self,
        ax,
        cur_location,
        heading: float,
        color: str,
        bbox_size: Tuple[float, float],
    ) -> None:
        """Plot an actor bounding box centered on the actor's current location.

        Args:
            ax: Axes on which actor bounding box should be plotted.
            cur_location: Current location of the actor (2,).
            heading: Current heading of the actor (in radians).
            color: Desired color for the bounding box.
            bbox_size: Desired size for the bounding box (length, width).
        """
        (bbox_length, bbox_width) = bbox_size

        # Compute coordinate for pivot point of bounding box
        d = np.hypot(bbox_length, bbox_width)
        theta_2 = math.atan2(bbox_width, bbox_length)
        pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
        pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

        vehicle_bounding_box = Rectangle(
            (pivot_x, pivot_y),
            bbox_length,
            bbox_width,
            angle=np.degrees(heading),
            color=color,
            zorder=self._BOUNDING_BOX_ZORDER,
        )
        ax.add_patch(vehicle_bounding_box)

    def plot_raw_map(self, static_map):
        for drivable_area in static_map.vector_drivable_areas.values():
            self._plot_polygons(
                [drivable_area.xyz], alpha=0.5, color=self._DRIVABLE_AREA_COLOR
            )

        # Plot lane segments
        for lane_segment in static_map.vector_lane_segments.values():
            self._plot_polylines(
                [
                    lane_segment.left_lane_boundary.xyz,
                    lane_segment.right_lane_boundary.xyz,
                ],
                line_width=0.5,
                color=self._LANE_SEGMENT_COLOR,
            )
        show_ped_xings = True
        # Plot pedestrian crossings
        if show_ped_xings:
            for ped_xing in static_map.vector_pedestrian_crossings.values():
                self._plot_polylines(
                    [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                    alpha=1.0,
                    color=self._LANE_SEGMENT_COLOR,
                )

    def plot_packed_map(self, data):
        for i, pt in enumerate(data["map_point"]["position"]):
            pl_type = data["map_polygon"]["type"][i]
            if pl_type == self._polygon_types.index("PEDESTRIAN"):
                color = self._LANE_CROSSWALK_COLOR
            else:
                color = self._LANE_SEGMENT_COLOR
            left_s = []
            right_s = []
            center_s = []
            for j, s in enumerate(data["map_point"]["side"][i]):
                if s == 0:
                    left_s.append(j)
                elif s == 1:
                    right_s.append(j)
                else:
                    center_s.append(j)

            if color == self._LANE_CROSSWALK_COLOR:
                self._plot_polylines(
                    [pt[left_s + right_s], pt[center_s]],
                    alpha=1,
                    color=color,
                )
            else:
                self._plot_polylines(
                    [
                        pt[left_s],
                        pt[right_s],
                        # pt[center_s]
                    ],
                    line_width=0.5,
                    color=color,
                )

    def __call__(self, data, preds=None, map_path=None, save_path=None):
        assert (
            ArgoverseStaticMap is not None
        ), "Argoverse2 visualize should install av2"
        ht = self.num_historical_steps
        A = data["agent"]["num_nodes"]
        _, ax = plt.subplots(figsize=(25, 25))
        ax.set_aspect("equal", adjustable="box")
        if map_path is not None:
            static_map = ArgoverseStaticMap.from_json(map_path)
            self.plot_raw_map(static_map)
        else:
            self.plot_packed_map(data)
        av_idx = data["agent"]["av_index"]
        av_pos = data["agent"]["position"][av_idx][0]
        if preds is not None:
            preds["pred"] = preds["pred"] + av_pos[None, :2]
        for i in range(A):
            cat = data["agent"]["category"][i]
            agent_type = data["agent"]["type"][i]
            position = np.array(data["agent"]["position"][i])
            heading = np.array(data["agent"]["heading"][i])
            valid_mask = np.array(data["agent"]["valid_mask"][i])
            category = self._agent_categories[cat]
            agent_type = self._agent_types[agent_type]
            track_color = self._DEFAULT_ACTOR_COLOR

            if category == "FOCAL_TRACK":
                track_color = self._FOCAL_AGENT_COLOR
                self._plot_polylines(
                    [position[:ht, :]], color=track_color, line_width=2
                )

                self._plot_polylines(
                    [position[ht:, :]],
                    color=self._GT_AGENT_COLOR,
                    line_width=2,
                    arrow=True,
                )

                if preds is not None:
                    pred_track = preds["pred"][0].detach().cpu().numpy()
                    for i in range(6):
                        self._plot_polylines(
                            [pred_track[i, :, :]],
                            color=self._PREDICT_AGENT_COLOR,
                            line_width=1,
                            arrow=True,
                        )
                # break
            elif i == data["agent"]["av_index"]:
                track_color = self._AV_COLOR
            elif agent_type in self._static_types:
                continue
            if valid_mask[ht - 1]:
                # Plot bounding boxes for all vehicles and cyclists
                if agent_type == "vehicle":
                    self._plot_actor_bounding_box(
                        ax,
                        position[ht - 1],
                        heading[ht - 1],
                        track_color,
                        (
                            self._ESTIMATED_VEHICLE_LENGTH_M,
                            self._ESTIMATED_VEHICLE_WIDTH_M,
                        ),
                    )
                elif agent_type == "cyclist" or agent_type == "motorcyclist":
                    self._plot_actor_bounding_box(
                        ax,
                        position[ht - 1],
                        heading[ht - 1],
                        track_color,
                        (
                            self._ESTIMATED_CYCLIST_LENGTH_M,
                            self._ESTIMATED_CYCLIST_WIDTH_M,
                        ),
                    )
                else:
                    plt.plot(
                        position[ht - 1, 0],
                        position[ht - 1, 1],
                        "o",
                        color=track_color,
                        markersize=4,
                    )
        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "traj_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
