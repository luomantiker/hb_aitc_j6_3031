import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import msgpack
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from hat.utils.package_helper import require_packages
from .data_packer import Packer

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object


__all__ = [
    "Argoverse2Base",
    "Argoverse2PackedDataset",
    "Argoverse2Packer",
]


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def side_to_directed_lineseg(
    query_point: torch.Tensor,
    start_point: torch.Tensor,
    end_point: torch.Tensor,
) -> str:
    cond = (end_point[0] - start_point[0]) * (
        query_point[1] - start_point[1]
    ) - (end_point[1] - start_point[1]) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


@OBJECT_REGISTRY.register
class Argoverse2Base:
    """
    Argoverse2 dataset handler.

    Args:
        num_historical_steps: Number of historical time steps.
        num_future_step: Number of future time steps.
        split: Dataset split name, e.g., 'train', 'val', 'test'.
    """

    def __init__(
        self,
        num_historical_steps: int = 60,
        num_future_steps: int = 50,
        split: str = "val",
    ):
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.split = split
        assert split in ["train", "test", "val"]

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
        self._polygon_is_intersections = [True, False, None]
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
        self._point_sides = ["LEFT", "RIGHT", "CENTER"]
        self._polygon_to_polygon_types = [
            "NONE",
            "PRED",
            "SUCC",
            "LEFT",
            "RIGHT",
        ]

    @require_packages("av2")
    def process(self, input_dir: str):
        """
        Process data from the specified input directory.

        Args:
            input_dir: Path to the directory containing input data.

        Returns:
            pd.DataFrame: Processed data as a Pandas DataFrame.
        """
        assert (
            ArgoverseStaticMap is not None
        ), "av2 should be installed for argoverse2 data process."
        raw_file_name = os.path.basename(input_dir)
        map_dir = Path(input_dir)
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)

        df = pd.read_parquet(
            os.path.join(
                input_dir,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["agent"] = self.get_agent_features(df)
        data.update(self.get_map_features(map_api, centerlines))

        return data, map_path

    def _decode(self, sample: dict, input_dim: int):

        for k in [
            "valid_mask",
            "predict_mask",
            "type",
            "category",
            "position",
            "heading",
            "velocity",
        ]:
            sample["agent"][k] = torch.tensor(sample["agent"][k]).clone()

        for k in [
            "position",
            "orientation",
            "height",
            "type",
            "is_intersection",
        ]:
            sample["map_polygon"][k] = torch.tensor(
                sample["map_polygon"][k]
            ).clone()
        for k in [
            "position",
            "orientation",
            "magnitude",
            "height",
            "type",
            "side",
        ]:
            sample["map_point"][k] = [
                torch.tensor(p) for p in sample["map_point"][k]
            ]
        sample["map_point_to_map_polygon"]["edge_index"] = (
            torch.tensor(sample["map_point_to_map_polygon"]["edge_index"])
            .long()
            .clone()
        )
        sample["map_polygon_to_map_polygon"]["edge_index"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["edge_index"]
        ).clone()
        sample["map_polygon_to_map_polygon"]["type"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["type"]
        ).clone()

        sample["map_polygon"]["position"] = sample["map_polygon"]["position"][
            :, :input_dim
        ]
        pl_num = sample["map_polygon"]["num_nodes"]
        pl2pl_type_mat = torch.zeros([pl_num, pl_num])
        for i in range(
            sample["map_polygon_to_map_polygon"]["edge_index"].shape[1]
        ):
            src, dst = sample["map_polygon_to_map_polygon"]["edge_index"][:, i]
            pl2pl_type_mat[src][dst] = sample["map_polygon_to_map_polygon"][
                "type"
            ][i]
        sample["map_polygon_to_map_polygon"]["type_mat"] = pl2pl_type_mat
        return sample

    @require_packages("av2")
    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.predict_unseen_agents:
            historical_df = df[df["timestep"] < self.num_historical_steps]
            agent_ids = list(historical_df["track_id"].unique())
            df = df[df["track_id"].isin(agent_ids)]
        else:
            agent_ids = list(df["track_id"].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index("AV")

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(
            num_agents, self.num_steps, dtype=torch.bool
        )
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(
            num_agents, self.num_steps, self.dim, dtype=torch.float
        )

        for track_id, track_df in df.groupby("track_id"):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df["timestep"].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[
                agent_idx, self.num_historical_steps - 1
            ]
            predict_mask[agent_idx, agent_steps] = True
            if (
                self.vector_repr
            ):  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1 : self.num_historical_steps] = (
                    valid_mask[agent_idx, : self.num_historical_steps - 1]
                    & valid_mask[agent_idx, 1 : self.num_historical_steps]
                )
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, : self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps :] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(
                track_df["object_type"].values[0]
            )
            agent_category[agent_idx] = track_df["object_category"].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["position_x"].values,
                        track_df["position_y"].values,
                    ],
                    axis=-1,
                )
            ).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(
                track_df["heading"].values
            ).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(
                np.stack(
                    [
                        track_df["velocity_x"].values,
                        track_df["velocity_y"].values,
                    ],
                    axis=-1,
                )
            ).float()

        if self.split == "test":
            predict_mask[
                current_valid_mask
                | (agent_category == 2)
                | (agent_category == 3),
                self.num_historical_steps :,
            ] = True

        return {
            "num_nodes": num_agents,
            "av_index": av_idx,
            "valid_mask": valid_mask.numpy().tolist(),
            "predict_mask": predict_mask.numpy().tolist(),
            "id": agent_id,
            "type": agent_type.numpy().tolist(),
            "category": agent_category.numpy().tolist(),
            "position": position.numpy().tolist(),
            "heading": heading.numpy().tolist(),
            "velocity": velocity.numpy().tolist(),
        }

    @require_packages("av2")
    def get_map_features(
        self, map_api: ArgoverseStaticMap, centerlines: Mapping[str, Polyline]
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(
            num_polygons, self.dim, dtype=torch.float
        )
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(
                centerlines[lane_segment.id].xyz
            ).float()
            polygon_position[lane_segment_idx] = centerline[0, : self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(
                centerline[1, 1] - centerline[0, 1],
                centerline[1, 0] - centerline[0, 0],
            )
            polygon_height[lane_segment_idx] = (
                centerline[1, 2] - centerline[0, 2]
            )
            polygon_type[lane_segment_idx] = self._polygon_types.index(
                lane_segment.lane_type.value
            )
            polygon_is_intersection[
                lane_segment_idx
            ] = self._polygon_is_intersections.index(
                lane_segment.is_intersection
            )

            left_boundary = torch.from_numpy(
                lane_segment.left_lane_boundary.xyz
            ).float()
            right_boundary = torch.from_numpy(
                lane_segment.right_lane_boundary.xyz
            ).float()
            point_position[lane_segment_idx] = torch.cat(
                [
                    left_boundary[:-1, : self.dim],
                    right_boundary[:-1, : self.dim],
                    centerline[:-1, : self.dim],
                ],
                dim=0,
            )
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat(
                [
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                    torch.atan2(center_vectors[:, 1], center_vectors[:, 0]),
                ],
                dim=0,
            )
            point_magnitude[lane_segment_idx] = torch.norm(
                torch.cat(
                    [
                        left_vectors[:, :2],
                        right_vectors[:, :2],
                        center_vectors[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_height[lane_segment_idx] = torch.cat(
                [
                    left_vectors[:, 2],
                    right_vectors[:, 2],
                    center_vectors[:, 2],
                ],
                dim=0,
            )
            left_type = self._point_types.index(
                lane_segment.left_mark_type.value
            )
            right_type = self._point_types.index(
                lane_segment.right_mark_type.value
            )
            center_type = self._point_types.index("CENTERLINE")
            point_type[lane_segment_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),), left_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(right_vectors),), right_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_side[lane_segment_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[: self.dim]
            polygon_position[
                crosswalk_idx + len(cross_walk_ids)
            ] = end_position[: self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2(
                (end_position - start_position)[1],
                (end_position - start_position)[0],
            )
            polygon_orientation[
                crosswalk_idx + len(cross_walk_ids)
            ] = torch.atan2(
                (start_position - end_position)[1],
                (start_position - end_position)[0],
            )
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = (
                start_position[2] - end_position[2]
            )
            polygon_type[crosswalk_idx] = self._polygon_types.index(
                "PEDESTRIAN"
            )
            polygon_type[
                crosswalk_idx + len(cross_walk_ids)
            ] = self._polygon_types.index("PEDESTRIAN")
            polygon_is_intersection[
                crosswalk_idx
            ] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[
                crosswalk_idx + len(cross_walk_ids)
            ] = self._polygon_is_intersections.index(None)

            if (
                side_to_directed_lineseg(
                    (edge1[0] + edge1[-1]) / 2, start_position, end_position
                )
                == "LEFT"
            ):
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = (
                math.ceil(
                    torch.norm(
                        end_position - start_position, p=2, dim=-1
                    ).item()
                    / 2.0
                )
                + 1
            )
            centerline = torch.from_numpy(
                compute_midpoint_line(
                    left_ln_boundary=left_boundary.numpy(),
                    right_ln_boundary=right_boundary.numpy(),
                    num_interp_pts=int(num_centerline_points),
                )[0]
            ).float()

            point_position[crosswalk_idx] = torch.cat(
                [
                    left_boundary[:-1, : self.dim],
                    right_boundary[:-1, : self.dim],
                    centerline[:-1, : self.dim],
                ],
                dim=0,
            )
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    right_boundary.flip(dims=[0])[:-1, : self.dim],
                    left_boundary.flip(dims=[0])[:-1, : self.dim],
                    centerline.flip(dims=[0])[:-1, : self.dim],
                ],
                dim=0,
            )
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                    torch.atan2(center_vectors[:, 1], center_vectors[:, 0]),
                ],
                dim=0,
            )
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.atan2(
                        -right_vectors.flip(dims=[0])[:, 1],
                        -right_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -left_vectors.flip(dims=[0])[:, 1],
                        -left_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -center_vectors.flip(dims=[0])[:, 1],
                        -center_vectors.flip(dims=[0])[:, 0],
                    ),
                ],
                dim=0,
            )
            point_magnitude[crosswalk_idx] = torch.norm(
                torch.cat(
                    [
                        left_vectors[:, :2],
                        right_vectors[:, :2],
                        center_vectors[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat(
                    [
                        -right_vectors.flip(dims=[0])[:, :2],
                        -left_vectors.flip(dims=[0])[:, :2],
                        -center_vectors.flip(dims=[0])[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_height[crosswalk_idx] = torch.cat(
                [
                    left_vectors[:, 2],
                    right_vectors[:, 2],
                    center_vectors[:, 2],
                ],
                dim=0,
            )
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    -right_vectors.flip(dims=[0])[:, 2],
                    -left_vectors.flip(dims=[0])[:, 2],
                    -center_vectors.flip(dims=[0])[:, 2],
                ],
                dim=0,
            )
            crosswalk_type = self._point_types.index("CROSSWALK")
            center_type = self._point_types.index("CENTERLINE")
            point_type[crosswalk_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),), crosswalk_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(right_vectors),),
                        crosswalk_type,
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),),
                        crosswalk_type,
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(left_vectors),), crosswalk_type, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(center_vectors),), center_type, dtype=torch.uint8
                    ),
                ],
                dim=0,
            )
            point_side[crosswalk_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),),
                        self._point_sides.index("LEFT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(left_vectors),),
                        self._point_sides.index("RIGHT"),
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        self._point_sides.index("CENTER"),
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

        num_points = torch.tensor(
            [point.size(0) for point in point_position], dtype=torch.long
        )
        point_to_polygon_edge_index = torch.stack(
            [
                torch.arange(num_points.sum(), dtype=torch.long),
                torch.arange(num_polygons, dtype=torch.long).repeat_interleave(
                    num_points
                ),
            ],
            dim=0,
        )
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack(
                        [
                            torch.tensor(pred_inds, dtype=torch.long),
                            torch.full(
                                (len(pred_inds),),
                                lane_segment_idx,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                )
                polygon_to_polygon_type.append(
                    torch.full(
                        (len(pred_inds),),
                        self._polygon_to_polygon_types.index("PRED"),
                        dtype=torch.uint8,
                    )
                )
            succ_inds = []
            for succ in lane_segment.successors:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack(
                        [
                            torch.tensor(succ_inds, dtype=torch.long),
                            torch.full(
                                (len(succ_inds),),
                                lane_segment_idx,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                )
                polygon_to_polygon_type.append(
                    torch.full(
                        (len(succ_inds),),
                        self._polygon_to_polygon_types.index("SUCC"),
                        dtype=torch.uint8,
                    )
                )
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(
                    polygon_ids, lane_segment.left_neighbor_id
                )
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor(
                            [[left_idx], [lane_segment_idx]], dtype=torch.long
                        )
                    )
                    polygon_to_polygon_type.append(
                        torch.tensor(
                            [self._polygon_to_polygon_types.index("LEFT")],
                            dtype=torch.uint8,
                        )
                    )
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(
                    polygon_ids, lane_segment.right_neighbor_id
                )
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor(
                            [[right_idx], [lane_segment_idx]], dtype=torch.long
                        )
                    )
                    polygon_to_polygon_type.append(
                        torch.tensor(
                            [self._polygon_to_polygon_types.index("RIGHT")],
                            dtype=torch.uint8,
                        )
                    )
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(
                polygon_to_polygon_edge_index, dim=1
            )
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor(
                [[], []], dtype=torch.long
            )
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            "map_polygon": {},
            "map_point": {},
            "map_point_to_map_polygon": {},
            "map_polygon_to_map_polygon": {},
        }
        map_data["map_polygon"][
            "num_nodes"
        ] = num_polygons  # .numpy().tolist()
        map_data["map_polygon"]["position"] = polygon_position.numpy().tolist()
        map_data["map_polygon"][
            "orientation"
        ] = polygon_orientation.numpy().tolist()
        if self.dim == 3:
            map_data["map_polygon"]["height"] = polygon_height.numpy().tolist()
        map_data["map_polygon"]["type"] = polygon_type.numpy().tolist()
        map_data["map_polygon"][
            "is_intersection"
        ] = polygon_is_intersection.numpy().tolist()
        if len(num_points) == 0:
            map_data["map_point"]["num_nodes"] = 0
            map_data["map_point"]["position"] = []
            map_data["map_point"]["orientation"] = []
            map_data["map_point"]["magnitude"] = []
            if self.dim == 3:
                map_data["map_point"]["height"] = []
            map_data["map_point"]["type"] = []
            map_data["map_point"]["side"] = []
        else:
            map_data["map_point"]["num_nodes"] = num_points.sum().item()

            map_data["map_point"]["num_points"] = num_points.numpy().tolist()
            map_data["map_point"]["position"] = [
                p.numpy().tolist() for p in point_position
            ]
            map_data["map_point"]["orientation"] = [
                p.numpy().tolist() for p in point_orientation
            ]
            map_data["map_point"]["magnitude"] = [
                p.numpy().tolist() for p in point_magnitude
            ]
            if self.dim == 3:
                map_data["map_point"]["height"] = [
                    p.numpy().tolist() for p in point_height
                ]
            map_data["map_point"]["type"] = [
                p.numpy().tolist() for p in point_type
            ]
            map_data["map_point"]["side"] = [
                p.numpy().tolist() for p in point_side
            ]

        map_data["map_point_to_map_polygon"][
            "edge_index"
        ] = point_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "edge_index"
        ] = polygon_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "type"
        ] = polygon_to_polygon_type.numpy().tolist()

        return map_data


@OBJECT_REGISTRY.register
class Argoverse2PackedDataset(Dataset):
    """Argoverse2 Dataset of packed lmdb format.

    Args:
        data_path: The path to the packed dataset.
        split:  Dataset split name, e.g., 'train', 'val', 'test'.
        transforms: List of data transformations to apply.
        pack_type: The type of packing used for the dataset. here is "lmdb"
        input_dim: input_dim for argoverse2 data decode.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps for prediction.
        pack_kwargs: Additional keyword arguments for dataset packing.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "val",
        transforms: Optional[Callable] = None,
        pack_type: str = "lmdb",
        input_dim: int = 2,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        pack_kwargs: Optional[dict] = None,
    ):
        self.data_path = data_path

        if split not in ("train", "val", "test"):
            raise ValueError(f"{split} is not a valid split")
        self.split = split
        self.transforms = transforms
        self.input_dim = input_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.kwargs = {} if pack_kwargs is None else pack_kwargs

        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.argorvse2 = Argoverse2Base(
            self.num_historical_steps,
            self.num_future_steps,
            split=split,
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_data = self.pack_file.read(self.samples[idx])
        sample = msgpack.unpackb(
            raw_data,
        )
        data = self.argorvse2._decode(sample, self.input_dim)

        if self.transforms:
            data = self.transforms(data)
        return data


class Argoverse2Packer(Packer):
    """Argoverse2 Dataset Packer.

    Args:
        src_data_dir: The directory path where the source data is located.
        target_data_dir: The path where the packed data will be stored.
        split_name: The name of the dataset split to be packed
                 optional: ('train', 'test').
        num_workers: The number of workers to use for parallel processing.
        pack_type: The type of packing to be performed.
        num_samples: The number of samples to pack.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps for prediction.
        **kwargs: Additional keyword arguments for the packing process.
    """

    @require_packages("av2")
    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str = "val",
        num_workers: int = 10,
        pack_type: str = "lmdb",
        num_samples: Optional[int] = None,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        **kwargs,
    ):
        self.data_root = src_data_dir
        self.split = split_name
        assert self.split in ["train", "test", "val"]
        self._raw_file_names = [
            name
            for name in os.listdir(os.path.join(self.data_root, self.split))
            if os.path.isdir(os.path.join(self.data_root, self.split, name))
        ]
        self.raw_dir = os.path.join(self.data_root, self.split)
        if num_samples is None:
            num_samples = len(self._raw_file_names)
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self._num_samples = {
            "train": 199908,
            "val": 24988,
            "test": 24984,
        }[self.split]
        self.argoverse2 = Argoverse2Base(
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            split=self.split,
        )
        super(Argoverse2Packer, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        raw_file_name = self._raw_file_names[idx]
        df = pd.read_parquet(
            os.path.join(
                self.raw_dir,
                raw_file_name,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        map_dir = Path(self.raw_dir) / raw_file_name
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["scenario_id"] = df["scenario_id"].values[0]
        data["city"] = df["city"].values[0]
        data["agent"] = self.argoverse2.get_agent_features(df)
        data.update(self.argoverse2.get_map_features(map_api, centerlines))
        return msgpack.packb(data)
