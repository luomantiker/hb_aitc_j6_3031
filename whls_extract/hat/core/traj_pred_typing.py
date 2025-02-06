# Copyright (c) Horizon Robotics. All rights reserved.
# This file defines some structures for trajectory prediction.

import pathlib
from typing import List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray
from shapely.geometry import LineString, Point, Polygon

__all__ = [
    "PathLike",
    "ArrayLike",
    "Color",
    "TrajGroupIndex",
    "SeqIndex",
    "AgentIndex",
    "AgentIndexWithStamp",
    "SeqCenter",
    "PlaneAffineCoordinateSystem",
    "BboxCoordinates",
    "VehicleSafeArea",
    "PedestrainSafeArea",
    "StaticSafeArea",
    "detect_safe_area_collision",
]

PathLike = Union[str, pathlib.Path]
ArrayLike = Union[List, Tuple, ndarray]
# RGB color varying from 0 to 255
Color = Tuple[float, float, float]


class TrajGroupIndex(NamedTuple):
    """The idenfication to slice out a group of trajectory data.

    TrajGroupIndex is a tuple of indexes which can be used to slice out a
    DataFrame which corresponds to a group of trajectory data. The data
    in the same group is within a certain time period, data in different
    groups are usually unrelated and thus we separate them with different
    TrajGroupIndex indexes.

    Args:
        map_id: the map id.
        date: the date of the data.
        data_num: The index of the current group in the data collected
            on the same `date`.
        data_version: the version of the data.
        center_car_id: the track id of the center vehicle (usually
            be the ego vehicle).
    """

    map_id: str  # noqa: E701
    date: str
    data_num: float
    data_version: float
    center_car_id: float


class SeqIndex(NamedTuple):
    """Index to slice out a sequence from a big DataFrame.

    Args:
        traj_group_index: the idenfication to slice out a
            group of trajectory data
        frame_slice: the start, end and step of the data slice.
    """

    traj_group_index: TrajGroupIndex
    # frame ids can be expressed by range object for better efficiency
    frame_slice: slice


class AgentIndex(NamedTuple):  # noqa: D205,D400
    """Index to slice out a trajectory centered on a specific obstacle
    from a big DataFrame.

    Args:
        seq_index: index to slice out a sequence from a big
            DataFrame.
        track_id: the id of the agent.
        last_context_frame: in trajectory prediction, we use the
            information of some historical frames as input. the
            `last_context_frame` is the frame id of the last historical frame.
    """

    seq_index: SeqIndex
    track_id: int
    last_context_frame: int


class AgentIndexWithStamp(NamedTuple):  # noqa: D205,D400
    """Index to slice out a trajectory centered on a specific obstacle
    from a big DataFrame.

    Args:
        seq_index: index to slice out a sequence from a big
            DataFrame.
        track_id: the id of the agent.
        last_context_frame: in trajectory prediction, we use the
            information of some historical frames as input. the
            `last_context_frame` is the frame id of the last historical frame.
        timestamp: the unix timestamp of the `last_context_frame`.
    """

    seq_index: SeqIndex
    track_id: int
    last_context_frame: int
    timestamp: int


class SeqCenter(NamedTuple):
    """An object containing sequence center information.

    Args:
        pos_x: the x-coordinate of the center vehicle (usually
            in the global coordinates). [m]
        pos_y: the y-coordinate of the center vehicle (usually
            in the global coordinates). [m]
        yaw: the yaw of the center vehicle. [Rad]
    """

    pos_x: float
    pos_y: float
    yaw: float


class PlaneAffineCoordinateSystem(NamedTuple):
    """Data structure representing a 2D affine coordinate system.

    An affine coordinate system is defined relatively.

    Args:
        translation: translation relative to the reference system.
        rotation: radian rotation relative to the reference system.
        scale: how much the unit length along an axis is
            represented in the reference system.
    """

    translation: np.ndarray = np.array([0, 0])
    rotation: float = 0
    scale: np.ndarray = np.array([1.0, 1.0])


class BboxCoordinates(NamedTuple):
    """A tuple containing bbox coordinates.

    Args:
        x0: the x-coordinate of the bounding box corner 0.
        y0: the y-coordinate of the bounding box corner 0.
        (x0, y0) to (x3, y3) should be clockwise.
    """

    x0: float
    x1: float
    x2: float
    x3: float
    y0: float
    y1: float
    y2: float
    y3: float


FILTER_FLAG = {
    "normal": 0,
    "ego_only": 1,
    "vehicle_only": 2,
    "his_frame_not_enough": 3,
    "fut_frame_not_enough": 4,
    "ped_label_unstable": 5,
    "obs_too_far": 6,
    "obs_stationary": 7,
    "speed_drifting": 8,
    "obs_yaw_bouncing": 9,
    "reach_model_limit_num": 10,
    "dense_parking_vehicle": 11,
    "vehicle_lateral_drifting": 12,
    "normal_too_close_correct_yaw": 13,
    "vehicle_fut_traj_reverse": 15,
    "newest_his_frame_too_far": 16,
}

ANCHOR_TYPE = {
    # Here, we split the obstacle motion into different types under different
    # granularities. Each classification has a specific semantics.
    "poor": {
        0: "slow_straight",
        1: "medium_straight",
        2: "fast_straight",
        3: "slow_left",
        4: "fast_left",
        5: "slow_right",
        6: "fast_right",
        7: "right_lc",
        8: "right_lc_merge",
        9: "left_lc",
        10: "left_lc_merge",
        11: "ped_right",
        12: "ped_straight",
        13: "ped_left",
    },
    "classical": {
        0: "slow_straight",
        1: "medium_straight",
        2: "fast_straight",
        3: "slow_left",
        4: "fast_left",
        5: "slow_right",
        6: "fast_right",
        7: "right_lc",
        8: "right_lc_merge",
        9: "left_lc",
        10: "left_lc_merge",
        11: "ped_right",
        12: "ped_straight",
        13: "ped_left",
    },
    "expand": {
        0: "slow_straight",
        1: "medium_straight",
        2: "fast_straight",
        3: "very_fast_straight",
        4: "flying_on_road",
        5: "slow_gentle_left",
        6: "slow_sharp_left",
        7: "medium_gentle_left",
        8: "medium_sharp_left",
        9: "fast_gentle_left",
        10: "fast_sharp_left",
        11: "slow_gentle_right",
        12: "slow_sharp_right",
        13: "medium_gentle_right",
        14: "medium_sharp_right",
        15: "fast_gentle_right",
        16: "fast_sharp_right",
        17: "slow_gentle_left_lc",
        18: "slow_sharp_left_lc",
        19: "fast_gentle_left_lc",
        20: "fast_sharp_left_lc",
        21: "slow_gentle_left_lc_merge",
        22: "slow_sharp_left_lc_merge",
        23: "fast_gentle_left_lc_merge",
        24: "fast_sharp_left_lc_merge",
        25: "slow_gentle_right_lc",
        26: "slow_sharp_right_lc",
        27: "fast_gentle_right_lc",
        28: "fast_sharp_right_lc",
        29: "slow_gentle_right_lc_merge",
        30: "slow_sharp_right_lc_merge",
        31: "fast_gentle_right_lc_merge",
        32: "fast_sharp_right_lc_merge",
        33: "ped_right",
        34: "ped_straight",
        35: "ped_left",
    },
}

ANCHOR_TYPE_BLOCKLIST = {
    # Given a specific obstacle type (e.g., vehicle, pedestrain, etc.), not all
    # motion types defined in ANCHOR_TYPE are suitable (e.g. a pedestrain
    # cannot move as fast as a vehicle). Therefore, we set this type blocklist
    # to filter out the invalid types. The values in the blocklist is just the
    # keys in ANCHOR_TYPE.
    # fmt: off
    "poor": {
        "veh": [11, 13],
        "ped": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cyc": [2, 4, 7, 9, 11, 13],
    },
    "classical": {
        "veh": [11, 13],
        "ped": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cyc": [2, 4, 7, 9, 11, 13],
    },
    "expand": {
        "veh": [33, 35],
        "ped": list(np.arange(0, 33, 1)),
        "cyc": [
            2, 3, 4, 9, 10, 15, 16, 19, 20, 23, 24, 27,
            28, 31, 32, 33, 35,
        ],
    },
    # fmt: on
}


DEFAULT_COLOR_MAP = {
    "others": [0, 0, 0],
    "roadedge": [0, 0, 255],
    "roadarrow": [47, 79, 79],
    "solid_lanes": [200, 200, 200],
    "stopline": [192, 0, 64],
    "crosswalk": [255, 127, 80],
    "sections": [0, 0, 0],
    "junctions": [0, 0, 0],
    "virtuallanes": [127, 127, 127],
    "zones": [60, 20, 255],
    "parking_slots": [221, 160, 221],
    "drivable_areas": [127, 0, 0],
}

LANE_MARK_TYPES = {
    "unknown": 0,
    "arrow_left": 1,
    "arrow_forward": 2,
    "arrow_right": 3,
    "arrow_left_and_forward": 4,
    "arrow_right_and_forward": 5,
    "arrow_left_and_right": 6,
    "arrow_u_turn": 7,
    "arrow_u_turn_and_forward": 8,
    "arrow_u_turn_and_left": 9,
    "arrow_merge_left": 10,
    "arrow_merge_right": 11,
    "crosswalk_notice": 12,
    "speed_limit_low": 13,
    "speed_limit_high": 14,
    "arrow_no_left_turn": 15,
    "arrow_no_right_turn": 16,
    "arrow_no_u_turn": 17,
    "arrow_forward_and_left_and_right": 18,
    "arrow_forward_and_u_turn_and_left": 19,
    "arrow_right_and_u_turn": 20,
    "text": 21,
    "time": 22,
    "check_following_distance": 23,
    "stop_to_give_way": 24,
    "slowdown_to_give_way": 25,
    "stop_mark": 26,
    "nets": 27,
    "no_mark": 28,
}


class VehicleSafeArea:
    """Safe area of vehicle obstacles.

    Safe Area is a "defensive" area for obstacles, which means if no
    other obstacles invade this area, the obstacle will be safe enough.
    We define the safe area for moving vehicles as expanded trapezoidal
    areas around a basic trajectory.

                   . ~ ^ (expand boundary 1)
        ___  . ~ ^
       |___| - - - - - - (basic trajectory)
             ^ ~ .
                   ^ ~ . (expand boundary 2)
    """

    def __init__(
        self,
        traj: np.array,
        obs_cls: int = 0,
        expand_base: float = 4,
        expand_ratio: float = 0.2,
    ):
        """Initialize method.

        Args:
            traj ([traj_len, 2]): the basic trajectory for the safe area. [m]
                The basic trajectory can be obtained by three methods: (1)
                use rule-based expanding methods (e.g., constant velocity, CV);
                (2) use prediction results from learning-based model; (3) for
                ego vehicle, use future planning trajectories. Usually, the
                coordinates of the basic trajectory is VCS.
            obs_cls: the class of the obstacles.
            expand_base: the base width of the safe area. It is the width
                (perpendicular to the trajectory gradient) at the first
                future point.
            expand_ratio: the expanding parameters. The safe area width at
                the i-th future point is:
                [1 + (i-1) * expand_ratio] * expand_base.
        """
        self.traj = traj
        self.obs_cls = obs_cls
        self.expand_base = expand_base
        self.expand_ratio = expand_ratio
        self.exp_bound_pts, self.convex_hull = self.expand_traj_safe_area(
            traj, expand_base, expand_ratio
        )
        self.static = len(self.convex_hull) < 3
        self.global_traj = None

    @staticmethod
    def expand_traj_safe_area(
        traj: np.ndarray,
        expand_base: float = 4,
        expand_ratio: float = 0.2,
    ) -> Tuple[np.ndarray]:
        """Expand the safe area.

        Args:
            traj ([traj_len, 2]): the basic trajectory for the safe area. [m]
            expand_base: the base width of the safe area. It is the width
                (perpendicular to the trajectory gradient) at the first
                future point.
            expand_ratio: the expanding parameters. The safe area width at
                the i-th future point is:
                [1 + (i-1) * expand_ratio] * expand_base.

        Returns:
            expand_points ([traj_len * 2, 2]): the points of the two expanded
                boundaries.
            convex_hull ([num_pts, 1, 2]): the convex hull generated by
                `expand_points`.
        """
        # Calculate the points of the expanded safe area.
        expand_val = expand_base * (np.arange(0, len(traj)) * expand_ratio + 1)
        expand_radius = expand_val / 2
        traj_yaw = np.arctan2(np.diff(traj[:, 1]), np.diff(traj[:, 0]))
        traj_yaw = np.concatenate([traj_yaw, traj_yaw[-1:]], axis=0)[:, None]
        expand_yaw = np.concatenate(
            [traj_yaw + np.pi / 2, traj_yaw - np.pi / 2], axis=1
        )
        x_offset = np.cos(expand_yaw) * expand_radius[:, None]
        y_offset = np.sin(expand_yaw) * expand_radius[:, None]
        expand_points = traj[:, None, :] + np.concatenate(
            [x_offset[:, :, None], y_offset[:, :, None]], axis=-1
        )
        expand_points = expand_points.reshape(-1, 2)

        # Calculate the convex hull.
        expand_points = np.around(expand_points).astype(np.int64)
        convex_hull = cv2.convexHull(expand_points)
        return expand_points, convex_hull

    def render_traj_safe_area(
        self,
        viz_map: np.ndarray,
        resolution: float = 0.2,
        offset: tuple = (0, 0),
    ) -> np.ndarray:
        """Render the safe area for visualization.

        Args:
            viz_map ([image_h, image_w, 3]): the map for visualization.
            resolution: the resolution of the viz_map.
            offset: the coodinates of the vcs center on the viz_map. [pixel]

        Returns:
            viz_map: the `viz_map` including this safe area.
        """
        offset = np.array(offset)
        traj = self.traj / resolution + offset[None, :]
        traj = traj.astype(np.int64)
        exp_traj = self.exp_bound_pts / resolution + offset[None, :]
        exp_traj = exp_traj.astype(np.int64)
        convex_hull = self.convex_hull / resolution + offset[None, None, :]
        convex_hull = convex_hull.astype(np.int64)
        cv2.fillConvexPoly(
            viz_map,
            convex_hull,
            color=[0, 127, 0],
        )
        for i in range(len(traj) - 1):
            tmp_vec = traj[[i, i + 1], :]
            cv2.polylines(
                viz_map,
                [tmp_vec],
                isClosed=False,
                color=[255, 0, 0],
                thickness=2,
            )
        for coor in traj:
            cv2.circle(
                viz_map,
                coor,
                5,
                color=[0, 0, 255],
                thickness=-1,
            )
        for coor in exp_traj:
            cv2.circle(
                viz_map,
                coor,
                5,
                color=[0, 255, 0],
                thickness=-1,
            )
        return viz_map

    def set_img_coord_traj(self, img_coords: np.ndarray):
        """Set the basic trajectory in image coordinates.

        Args:
            img_coords: ([traj_len, 2]): the basic trajectory in the image
                coordinates. [pixel]
        """
        self.img_traj = img_coords


class PedestrainSafeArea:
    """Safe area of pedestrain obstacles.

    Safe Area is a "defensive" area for obstacles, which means if no
    other obstacles invade this area, the obstacle will be safe enough.
    We define the safe area for moving pedestrain as a circle surrounding
    the pedestrain position.
    """

    def __init__(self, point: List, radius: float, obs_cls: int = 1):
        """Initialize method.

        Args:
            point: the [x, y] coordinate of the pedestrain postion.
            radius: the radius of the pedestrain safe area.
            obs_cls: the class of the obstacles.
        """
        self.point = point
        self.radius = radius
        self.obs_cls = obs_cls

    def render_traj_safe_area(
        self,
        viz_map: np.ndarray,
        resolution: Optional[bool] = 0.2,
        offset: Optional[Tuple] = (0, 0),
    ):
        """Render the safe area for visualization.

        Args:
            viz_map ([image_h, image_w, 3]): the map for visualization.
            resolution: the resolution of the viz_map.
            offset: the coodinates of the vcs center on the viz_map. [pixel]

        Returns:
            viz_map: the `viz_map` including this safe area.
        """
        x = int(np.around(self.point[0] / resolution)) + offset[0]
        y = int(np.around(self.point[1] / resolution)) + offset[1]
        if self.radius > 0:
            cv2.circle(
                viz_map,
                [x, y],
                self.radius / resolution,
                color=[0, 127, 0],
                thickness=-1,
            )
        cv2.circle(
            viz_map,
            [x, y],
            10,
            color=[255, 255, 255],
            thickness=-1,
        )
        return viz_map


class StaticSafeArea(PedestrainSafeArea):
    """Safe area of static obstacles."""

    def __init__(self, point: List, obs_cls: int):
        """Initialize method.

        Args:
            point: the [x, y] coordinate of the pedestrain postion.
            obs_cls: the class of the obstacles.
        """
        kwargs = {"point": point, "radius": 0, "obs_cls": obs_cls}
        super(StaticSafeArea, self).__init__(**kwargs)


def detect_safe_area_collision(safe_area_1, safe_area_2):
    """Detect whether two safe areas have interaction.

    Args:
        safe_area_1: may be instance of VehicleSafeArea or PedestrainSafeArea.
        safe_area_2: may be instance of VehicleSafeArea or PedestrainSafeArea.

    Return:
        has_interaction (bool): whether to have interaction.
    """

    def collision_polygon_vs_polygon(area_1, area_2):
        """Detect collision between two polygons."""
        poly_1 = Polygon(area_1.convex_hull[:, 0, :])
        poly_2 = Polygon(area_2.convex_hull[:, 0, :])
        return poly_1.intersects(poly_2)

    def collision_polygon_vs_circle(poly, cir):
        """Detect collision between a polygon and a circle."""
        convex_hull = poly.convex_hull[:, 0, :]
        cir_center = Point(cir.point)
        if cir_center.within(Polygon(convex_hull)):
            return True
        min_dis = cir_center.distance(LineString(convex_hull))
        return min_dis <= cir.radius

    def collision_circle_vs_circle(cir_1, cir_2):
        """Detect collision between two circles."""
        x1, y1 = cir_1.point
        x2, y2 = cir_2.point
        dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        rad_sum = cir_1.radius + cir_2.radius
        return dis <= rad_sum

    if isinstance(safe_area_1, VehicleSafeArea):
        if isinstance(safe_area_2, VehicleSafeArea):
            return collision_polygon_vs_polygon(safe_area_1, safe_area_2)
        elif isinstance(safe_area_2, PedestrainSafeArea):
            return collision_polygon_vs_circle(safe_area_1, safe_area_2)
        else:
            raise TypeError(f"Undefined safe area type {type(safe_area_2)}")
    elif isinstance(safe_area_1, PedestrainSafeArea):
        if isinstance(safe_area_2, VehicleSafeArea):
            return collision_polygon_vs_circle(safe_area_2, safe_area_1)
        elif isinstance(safe_area_2, PedestrainSafeArea):
            return collision_circle_vs_circle(safe_area_1, safe_area_2)
        else:
            raise TypeError(f"Undefined safe area type {type(safe_area_2)}")
    else:
        raise TypeError(f"Undefined safe area type {type(safe_area_1)}")
