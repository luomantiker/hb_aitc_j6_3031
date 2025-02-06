# Copyright (c) Horizon Robotics. All rights reserved.
# Utility functions for trajectory prediction transform methods.

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pandas import DataFrame

from hat.core.traj_pred_typing import AgentIndex, SeqCenter, SeqIndex


class Affine2D:
    """2-dimensional affine transformation collection."""

    @staticmethod
    def coord_translate(
        input_x: np.ndarray,
        input_y: np.ndarray,
        translation_x: float,
        translation_y: float,
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep orientation and scale intact, translate the coordinate system's
        origin to a new point (translation_x, translation_y) in the original
        coordinate system.

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            translation_x: new origin's x coordinate.
            translation_y: new origin's y coordinate.

        Returns:
            trans_x: transformed x coordinates, an 1D array.
            trans_y: transformed y coordinates, an 1D array.
        """
        # We specify the dtype of the variables here because numpy (<1.17)
        #   does not support the matmul between np.float64 and np.object. Thus
        #   sometimes cause an error here.
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        translation_mat = np.array(
            [[1, 0, -translation_x], [0, 1, -translation_y], [0, 0, 1]],
            dtype=np.float64,
        )
        transformed_coordinates = np.matmul(translation_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def coord_rotate(
        input_x: np.ndarray, input_y: np.ndarray, rotation: float
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep origin and scale intact, rotate the coordinate system. The
        rotation is defined along x-axis counterclockwise in radian.

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            rotation: radian angle along x-axis counterclockwise.

        Returns:
            trans_x: transformed x coordinates, an 1D array.
            trans_y: transformed y coordinates, an 1D array.
        """
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        sin = np.sin
        cos = np.cos
        rotation_mat = np.array(
            [
                [cos(rotation), sin(rotation), 0],
                [-sin(rotation), cos(rotation), 0],
                [0, 0, 1],
            ],
            dtype=np.object_,
        ).astype(np.float64)
        transformed_coordinates = np.matmul(rotation_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def coord_scale(
        input_x: np.ndarray,
        input_y: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep origin and orientation intact, scale the coordinate system's
        x, y axis.

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            scale_x: the x coordinate of the new unit x vector in the
                original coordinate system.
            scale_y: the y coordinate of the new unit y vector in the
                original coordinate system.

        Returns:
            trans_x: transformed x coordinates, an 1D array.
            trans_y: transformed y coordinates, an 1D array.
        """
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        scale_mat = np.array(
            [[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]],
            dtype=np.float64,
        )

        transformed_coordinates = np.matmul(scale_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def obj_translate(
        input_x: np.ndarray,
        input_y: np.ndarray,
        translation_x: float,
        translation_y: float,
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep the coordinate system intact, translate the obj to a new
        position

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            translation_x: object's new x coordinate.
            translation_y: object's new y coordinate.

        Returns:
            trans_x: transformed x coordinates, an 1D array.
            trans_y: transformed y coordinates, an 1D array.
        """
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        translation_mat = np.array(
            [[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]],
            dtype=np.float64,
        )
        transformed_coordinates = np.matmul(translation_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def obj_rotate(
        input_x: np.ndarray, input_y: np.ndarray, rotation: float
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep the coordinate system intact, rotate the object with origin as
        the rotation center. The rotation is defined along x-axis
        counterclockwise in radian.

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            rotation: radian angle along x-axis counterclockwise.

        Returns:
            trans_x: transformed x coordinates, an 1D array.
            trans_y: transformed y coordinates, an 1D array.
        """
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        sin = np.sin
        cos = np.cos
        rotation_mat = np.array(
            [
                [cos(rotation), -sin(rotation), 0],
                [sin(rotation), cos(rotation), 0],
                [0, 0, 1],
            ],
            dtype=np.object_,
        ).astype(np.float64)
        transformed_coordinates = np.matmul(rotation_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def obj_scale(
        input_x: np.ndarray,
        input_y: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[np.ndarray]:  # noqa: D205,D400
        """Keep the coordinate system intact, scale object's coordinates along
        x and y axis.

        Args:
            input_x: input x coordinates, an 1D array.
            input_y: input y coordinates, an 1D array.
            scale_x: new_length_along_x / original_length_along_x
            scale_y: new_width_along_y / original_width_along_y

        Returns:
            transformed x coordinates of 1D array,
            transformed y coordinates of 1D array.
        """
        coordinates = np.array(
            [input_x, input_y, [1] * len(input_x)], dtype=np.float64
        )
        scale_mat = np.array(
            [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64
        )

        transformed_coordinates = np.matmul(scale_mat, coordinates)

        return transformed_coordinates[0], transformed_coordinates[1]

    @staticmethod
    def batched_pose_transform(
        batched_poses: np.ndarray,
        anchor_poses: np.ndarray,
    ) -> np.ndarray:  # noqa: D205,D400
        """Transform batched poses into the coordinate systems defined by
        anchor_poses. All poses consist of the x,y coordinates and the yaw.

        Args:
            batched_poses: [B, N, 3]
            anchor_poses: anchor poses in the same coordinate system as batched
                poses. Shape: [N, 3]

        Returns:
            Transformed poses: [B, N, 3]
        """
        if batched_poses.shape[-1] != 3:
            print(batched_poses.shape)
        assert batched_poses.shape[-1] == 3
        assert anchor_poses.shape[-1] == 3
        assert batched_poses.ndim == 3
        assert anchor_poses.ndim == 2
        assert batched_poses.shape[1] == anchor_poses.shape[0]

        N = anchor_poses.shape[0]
        origins = anchor_poses[:, :2]  # [N, 2]
        headings = anchor_poses[:, 2]
        cos, sin = np.cos(headings), np.sin(headings)
        rot_mat = np.zeros((N, 2, 2))
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        output_coords = np.matmul(
            batched_poses[:, :, :2][:, :, None, :] - origins[None, :, None, :],
            rot_mat,
        ).squeeze(
            -2
        )  # [B, N, 2]
        output_headings = batched_poses[:, :, 2] - headings  # [B, N]
        output_poses = np.concatenate(
            [output_coords, output_headings[:, :, None]], axis=-1
        )
        return output_poses

    @staticmethod
    def batched_inverse_pose_transform(
        batched_poses: np.ndarray,
        anchor_poses: np.ndarray,
    ) -> np.ndarray:  # noqa: D205,D400
        """Suppose there is a global coordinate system "gcs" and a vehicle-
        centric coordinate system "vcs". The batched_poses are in "vcs" and
        the "anchor_poses" are in "gcs". And the batched_poses were transformed
        into "gcs" with the anchor_poses as the origin and the heading. This
        function conducts the inverse operation to transform the batched_poses
        back into the global coordiantes. A practical use case is to transform
        the vectorized map elements from vcs into gcs.

        Args:
            batched_poses: [B, N, 3]
            anchor_poses: anchor poses in the same coordinate system as batched
                poses. Shape: [N, 3]

        Returns:
            Transformed poses: [B, N, 3]
        """
        assert batched_poses.shape[-1] == 3
        assert anchor_poses.shape[-1] == 3
        assert batched_poses.ndim == 3
        assert anchor_poses.ndim == 2
        assert batched_poses.shape[1] == anchor_poses.shape[0]

        N = anchor_poses.shape[0]
        origins = anchor_poses[:, :2]  # [N, 2]
        headings = anchor_poses[:, 2]
        cos, sin = np.cos(headings), np.sin(headings)
        inv_rot_mat = np.zeros((N, 2, 2))
        inv_rot_mat[:, 0, 0] = cos
        inv_rot_mat[:, 0, 1] = sin
        inv_rot_mat[:, 1, 0] = -sin
        inv_rot_mat[:, 1, 1] = cos

        output_coords = (
            np.matmul(
                batched_poses[:, :, :2][:, :, None, :],
                inv_rot_mat,
            ).squeeze(-2)
            + origins
        )  # [B, N, 2]
        output_headings = batched_poses[:, :, 2] + headings  # [B, N]
        output_headings = output_headings % (2 * np.pi)
        output_poses = np.concatenate(
            [output_coords, output_headings[:, :, None]], axis=-1
        )
        return output_poses


class TdtCoordHelper:
    """A collection of coordinate calculation (translation) functions.

    This class includes some basic coordinate translation methods and
    their "upgraded version" (to process the corresponding functions
    on trajectory prediction tdt format dataframes).

    Moreover, it also includes some coordinate calculation methods like
    getting center and bounding box corners.
    """

    @staticmethod
    def get_seq_center(
        df: DataFrame,
        seq_index: SeqIndex,
        anchor_frame: Optional[str] = "middle_seq",
        context_frames: Optional[int] = None,
        last_context_frame_id: Optional[int] = None,
        agent_index: Optional[AgentIndex] = None,
    ) -> SeqCenter:  # noqa: D205,D400
        """Get the coordinates of the ego vehicle in the middle time frame.
        The ego vehicle information should be formatted as tdt format and
        inputted as a dataframe.

        Args:
            df: a sequence DataFrame.
            seq_index: a SeqIndex that contains complete
                information about how to slice the `df` from the entire
                dataframe.
            anchor_frame: which frame to extract the center coordinates
                of the sequence. Choose from 'middle_seq', 'last_context',
                'first_seq', 'agent'. Defaults to 'middle_seq'.
            context_frames: number of context frames. Defaults to None.
            last_context_frame_id: the last context frame id. This
                parameter is used when `anchor_frame` is 'last_context'.
            agent_index: a AgentIndex that contains complete
                information to slice from dataframe. Different from seq_index,
                agent_index is centered at a obstacle (not the ego vehicle).
                Defaults to None.

        Returns:
            SeqCenter: Seqcenter from middle frame ego information.
        """
        required_columns = ["frame_id", "pos_x", "pos_y", "yaw"]
        for col in required_columns:
            assert (
                col in df.columns
            ), f"The column {col} is not found in the input dataframe."

        frame_slice = seq_index.frame_slice
        frame_ids = np.sort(df["frame_id"].unique())

        # Find the frame id for the sequence center coordinates.
        if anchor_frame == "middle_seq":
            slice_middle_idx = (
                frame_slice.start + frame_slice.stop - frame_slice.step
            )  # noqa: E501
            # Find the frame_id nearest to slice_middle_idx.
            frame_id_offset = frame_ids - slice_middle_idx
            frame_id_middle_idx = np.argmin(abs(frame_id_offset))
            anchor_frame_id = frame_ids[frame_id_middle_idx]
        elif anchor_frame == "last_context":
            if context_frames is None:
                raise Exception("Please input the number of context frames!")
            if last_context_frame_id is None:
                anchor_frame_id = frame_ids[context_frames - 1]
            else:
                anchor_frame_id = last_context_frame_id
        elif anchor_frame == "first_seq":
            anchor_frame_id = frame_ids[0]
        elif anchor_frame == "agent":
            if agent_index is None:
                raise Exception("Please input an agent_index!")
            anchor_frame_id = agent_index.last_context_frame
        else:
            raise ValueError(
                f"The input anchor frame should in ["
                f"last_context, firse_seq, middle_seq, agent], "
                f"but {anchor_frame} is given."
            )

        slice_df = df.loc[df["frame_id"] == anchor_frame_id]
        return SeqCenter(
            pos_x=slice_df["pos_x"].values[0],
            pos_y=slice_df["pos_y"].values[0],
            yaw=slice_df["yaw"].values[0],
        )

    @staticmethod
    def global_phy_to_local_img(
        global_phy_coords: np.ndarray,
        seq_center: SeqCenter,
        seq_center_img_offset: np.ndarray,
        resolution: float,
        reverse: Optional[bool] = False,
    ) -> np.ndarray:
        """Transform global physical coordinates to local image cooridnates.

        The local image coordinate system is defined as a coordinate system
        with respect to a SeqCenter and a specific resolution. The origin is
        the SeqCenter (possibly shifted with an offset), the x axis's
        orientation is parallel to the yaw of SeqCenter.

        Args:
            global_phy_coords: global physical coordinates.
            seq_center: SeqCenter of the sequence.
            seq_center_img_offset: SeqCenter coordinate in the
                local image coordinate system.
            resolution: local image cooridnate system's resolution with
                the unit meters/pixel.
            reverse: if the direction of VCS and image coordinate system
                is on the contrary (like in the BEV scenario), reverse should
                be True.

        Returns:
            local image coordinates.
        """
        x, y = (global_phy_coords[:, 0], global_phy_coords[:, 1])
        x, y = Affine2D.coord_translate(
            x, y, seq_center.pos_x, seq_center.pos_y
        )
        x, y = Affine2D.coord_rotate(x, y, seq_center.yaw)
        x, y = Affine2D.coord_scale(x, y, resolution, resolution)
        if reverse:
            img_coords = np.array([-x, -y]).T
        else:
            img_coords = np.array([x, y]).T
        img_coords = img_coords + np.array(seq_center_img_offset)
        return img_coords

    @staticmethod
    def local_img_to_global_phy(
        local_img_coords: np.ndarray,
        seq_center: SeqCenter,
        seq_center_img_offset: np.ndarray,
        resolution: float,
        reverse: Optional[bool] = False,
    ) -> np.ndarray:  # noqa: D205,D400
        """Inversely transform local image coordinates to global physical
        coordinates.

        Args:
            local_img_coords: local image coordinates.
            seq_center: SeqCenter of the sequence in the global
                coordinates.
            seq_center_img_offset: SeqCenter's local image
                coordinate.
            resolution: local image cooridnate system's resolution with
                the unit meters/pixel.
            reverse: if the direction of VCS and image coordinate system
                is on the contrary (like in the BEV scenario), reverse should
                be True.

        Returns:
            global physical coordinates.
        """
        center_as_origin_coords = local_img_coords - seq_center_img_offset
        x, y = (center_as_origin_coords[:, 0], center_as_origin_coords[:, 1])
        x, y = Affine2D.coord_scale(x, y, 1 / resolution, 1 / resolution)
        x, y = Affine2D.coord_rotate(x, y, -seq_center.yaw)
        x, y = Affine2D.coord_translate(
            x, y, -seq_center.pos_x, -seq_center.pos_y
        )
        if reverse:
            global_phy_coords = np.array([-x, -y]).T
        else:
            global_phy_coords = np.array([x, y]).T
        return global_phy_coords

    @staticmethod
    def global_phy_to_local_phy(
        global_phy_coords: np.ndarray, seq_center: SeqCenter
    ) -> np.ndarray:  # noqa: D205,D400
        """Transform from global physical coordinate system to the local
        physical coordinate system.

        The local physical coordinate system's origin lies in SeqCenter's
        position, and it's x axis is the direction of the yaw of the SeqCenter.

        Args:
            global_phy_coords: global physical coordinates.
            seq_center: SeqCenter of the sequence.

        Returns:
            local physical coordinates.
        """
        x, y = (global_phy_coords[:, 0], global_phy_coords[:, 1])
        x, y = Affine2D.coord_translate(
            x, y, seq_center.pos_x, seq_center.pos_y
        )
        x, y = Affine2D.coord_rotate(x, y, seq_center.yaw)
        local_phy_coords = np.array([x, y]).T
        return local_phy_coords

    @staticmethod
    def local_phy_to_global_phy(
        local_phy_coords: np.ndarray, seq_center: SeqCenter
    ) -> np.ndarray:  # noqa: D205,D400
        """Inversely transform local physical coordinates to global
        coordinates.

        Args:
            local_phy_coords: local physical coordinates.
            seq_center: SeqCenter of the sequence.

        Returns:
            global physical coordinates.
        """
        x, y = (local_phy_coords[:, 0], local_phy_coords[:, 1])
        x, y = Affine2D.coord_rotate(x, y, -seq_center.yaw)
        x, y = Affine2D.coord_translate(
            x, y, -seq_center.pos_x, -seq_center.pos_y
        )
        global_phy_coords = np.array([x, y]).T
        return global_phy_coords

    @staticmethod
    def global_phy_to_agent_centric_phy(
        global_phy_coords: np.ndarray,
        agent_global_phy_coords: np.ndarray,
        agent_yaw: float,
    ) -> np.ndarray:  # noqa: D205,D400
        """Transform global physical cooridnates to agent centric physical
        coordinates.

        Agent centric system's origin is agent's position, and x axis's
        orientation is parallel to the agent's heading orientation.

        Args:
            global_phy_coords: global physical coordinates.
            agent_global_phy_coords: agent's position in the
                global physical coordinate system.
            agent_yaw: agent's yaw.

        Returns:
            agent centric physical cooridantes.
        """
        x, y = (global_phy_coords[:, 0], global_phy_coords[:, 1])
        agent_x, agent_y = agent_global_phy_coords
        x, y = Affine2D.coord_translate(x, y, agent_x, agent_y)
        x, y = Affine2D.coord_rotate(x, y, agent_yaw)
        agent_centric_phy_coords = np.array([x, y]).T
        return agent_centric_phy_coords

    @staticmethod
    def agent_centric_phy_to_global_phy(
        agent_centric_phy_coords: np.ndarray,
        agent_global_phy_coords: np.ndarray,
        agent_yaw: float,
    ) -> np.ndarray:  # noqa: D205,D400
        """Inversely transform agent centric cooridnates to global physical
        coordinates.

        Args:
            agent_centric_phy_coords: agent centric physical
                coordinates.
            agent_global_phy_coords: agent's global physical
                coordinates.
            agent_yaw: agent's yaw in global physical system.

        Returns:
            global physical coordinates.
        """
        x, y = (agent_centric_phy_coords[:, 0], agent_centric_phy_coords[:, 1])
        agent_x, agent_y = agent_global_phy_coords
        x, y = Affine2D.coord_rotate(x, y, -agent_yaw)
        x, y = Affine2D.coord_translate(x, y, -agent_x, -agent_y)
        agent_centric_phy_coords = np.array([x, y]).T
        return agent_centric_phy_coords

    @staticmethod
    def global_phy_df_to_local_img_df(
        phy_df: DataFrame,
        seq_center: SeqCenter,
        seq_center_img_offset: np.ndarray,
        resolution: float,
        reverse: Optional[bool] = False,
    ) -> DataFrame:  # noqa: D205,D400
        """Transform a tdt dataframe from physical coordinate system to image
        coordinate system.

        This method takes in a DataFrame in "tdt format", and transform the
        following columns to image coordinate system: [['pos_x', 'pos_y'],
        ['ego_x0', 'ego_y0'], ['ego_x1', 'ego_y1'], ['ego_x2', 'ego_y2'],
        ['ego_x3', 'ego_y3'], ['x', 'y'], ['x0', 'y0'], ['x1', 'y1'],
        ['x2', 'y2'], ['x3', 'y3'],['width', 'length'], ['obs_width',
        'obs_length']]. The 'yaw' and 'obs_yaw' columns are also transformed.
        After translation, the class will add (or update) the columns for
        image coordinate information. The added (or updated) column names
        are 'img_xxx' suppose the original column names are 'xxx'.

        Args:
            phy_df: a tdt dataframe in physical coordinate.
            seq_center: SeqCenter of the sequence dataframe in
                the physical coordinates.
            seq_center_img_offset: image coordinates of the map
                center of the dataframe.
            resolution: query resolution of the map.
            reverse: if image is local BEV, and is drawn from the
                ego car upwards, the result need to reverse.

        Returns:
            the dataframe in image coordinate system. Original
                information are kept, new image coordinate columns are added.
        """
        # The x-y coordinate tuples to transform.
        pos_to_transform = [
            ["pos_x", "pos_y"],
            ["ego_x0", "ego_y0"],
            ["ego_x1", "ego_y1"],
            ["ego_x2", "ego_y2"],
            ["ego_x3", "ego_y3"],
            ["x", "y"],
            ["x0", "y0"],
            ["x1", "y1"],
            ["x2", "y2"],
            ["x3", "y3"],
            ["width", "length"],
            ["obs_width", "obs_length"],
        ]
        col_names = [col for item in pos_to_transform for col in item]
        assert set(col_names).issubset(
            phy_df.columns
        ), f"Input DataFrame doesn't have certain columns in {col_names}"

        out_df = phy_df.copy()
        for pos in pos_to_transform[:-2]:
            phy_car_array = phy_df[pos].values
            # If the columns are full of nan, it means those columns don't
            # have the desired data in the original dataset, give them 0s to
            # make sure that these columns have corresponding columns with
            # the 'img_' prefix even when these columns can't be transformed.
            if np.all(np.isnan(phy_car_array)):
                img_car_array = np.zeros_like(phy_car_array)
            else:
                img_car_array = TdtCoordHelper.global_phy_to_local_img(
                    phy_car_array,
                    seq_center,
                    seq_center_img_offset,
                    resolution,
                    reverse,
                )
            out_df[f"img_{pos[0]}"] = img_car_array[:, 0]
            out_df[f"img_{pos[1]}"] = img_car_array[:, 1]

        out_df["img_yaw"] = phy_df["yaw"] - seq_center.yaw
        out_df["img_obs_yaw"] = phy_df["obs_yaw"] - seq_center.yaw
        if reverse:
            out_df["img_yaw"] = out_df["img_yaw"] + np.pi
            out_df["img_obs_yaw"] = out_df["img_obs_yaw"] + np.pi
        img_ego_size = phy_df[["width", "length"]].values / resolution
        img_obs_size = phy_df[["obs_width", "obs_length"]].values / resolution
        out_df["img_width"] = img_ego_size[:, 0]
        out_df["img_length"] = img_ego_size[:, 1]
        out_df["img_obs_width"] = img_obs_size[:, 0]
        out_df["img_obs_length"] = img_obs_size[:, 1]

        return out_df

    @staticmethod
    def local_img_df_to_global_phy_df(
        img_df: DataFrame,
        seq_center: SeqCenter,
        seq_center_img_offset: np.ndarray,
        resolution: float,
        reverse: Optional[bool] = False,
    ) -> DataFrame:  # noqa: D205,D400
        """Transform a dataframe from image coordinate system to physical
        coordinate system.

        This method takes in a DataFrame in "tdt format", and transform the
        following columns to physical coordinate system: [['img_pos_x',
        'img_pos_y'], ['img_ego_x0', 'img_ego_y0'], ['img_ego_x1',
        'img_ego_y1'], ['img_ego_x2', 'img_ego_y2'], ['img_ego_x3',
        'img_ego_y3'], ['img_x', 'img_y'], ['img_x0', 'img_y0'], ['img_x1',
        'img_y1'], ['img_x2', 'img_y2'], ['img_x3', 'img_y3'],['img_width',
        'img_length'], ['img_obs_width', 'img_obs_length']]. The 'img_yaw' and
        'img_obs_yaw' columns are also transformed.
        After translation, the class will add (or update) the columns for
        physical coordinate information. The added (or updated) column names
        are 'xxx' suppose the original column names are 'img_xxx'.

        Args:
            img_df: a dataframe in image coordinate.
            seq_center: SeqCenter of the sequence dataframe in
                the physical coordinates.
            seq_center_img_offset: image coordinates of the map
                center of the dataframe.
            resolution: query resolution of the map.
            reverse: if image is local BEV, and is drawn from the
                ego car upwards, the result need to reverse.

        Returns:
            the dataframe in physical coordinate system. Original
                information are kept, new image coordinate columns are added.
        """
        # The x-y coordinate tuples to add or update.
        pos_to_update = [
            ["pos_x", "pos_y"],
            ["ego_x0", "ego_y0"],
            ["ego_x1", "ego_y1"],
            ["ego_x2", "ego_y2"],
            ["ego_x3", "ego_y3"],
            ["x", "y"],
            ["x0", "y0"],
            ["x1", "y1"],
            ["x2", "y2"],
            ["x3", "y3"],
            ["width", "length"],
            ["obs_width", "obs_length"],
        ]
        col_names = ["img_" + col for item in pos_to_update for col in item]
        assert set(col_names).issubset(
            img_df.columns
        ), f"Input DataFrame doesn't have certain columns in {col_names}"

        out_df = img_df.copy()
        for pos in pos_to_update[:-2]:
            pos_ori = ["img_" + i for i in pos]
            img_car_array = img_df[pos_ori].values
            if np.all(np.isnan(img_car_array)):
                phy_car_array = np.zeros_like(img_car_array)
            else:
                phy_car_array = TdtCoordHelper.local_img_to_global_phy(
                    img_car_array,
                    seq_center,
                    seq_center_img_offset,
                    resolution,
                    reverse,
                )
            out_df[pos[0]] = phy_car_array[:, 0]
            out_df[pos[1]] = phy_car_array[:, 1]

        out_df["yaw"] = img_df["img_yaw"] + seq_center.yaw
        out_df["obs_yaw"] = img_df["img_obs_yaw"] + seq_center.yaw
        if reverse:
            out_df["yaw"] = out_df["yaw"] - np.pi
            out_df["obs_yaw"] = out_df["obs_yaw"] - np.pi
        phy_ego_size = img_df[["img_width", "img_length"]].values * resolution
        phy_obs_size = (
            img_df[["img_obs_width", "img_obs_length"]].values * resolution
        )
        out_df["width"] = phy_ego_size[:, 0]
        out_df["length"] = phy_ego_size[:, 1]
        out_df["obs_width"] = phy_obs_size[:, 0]
        out_df["obs_length"] = phy_obs_size[:, 1]

        return out_df

    @staticmethod
    def global_phy_df_to_local_phy_df(
        phy_df: DataFrame, seq_center: SeqCenter
    ) -> DataFrame:  # noqa: D205,D400
        """Transform a dataframe from global physical coordinate system to
        local physical coordinate system.

        This method takes in a DataFrame in "tdt format", and transform the
        following columns to the local physical coordinate system:
           [['pos_x', 'pos_y'], ['ego_x0', 'ego_y0'], ['ego_x1', 'ego_y1'],
            ['ego_x2', 'ego_y2'], ['ego_x3', 'ego_y3'], ['x', 'y'],
            ['x0', 'y0'], ['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3'],
            ['width', 'length'], ['obs_width', 'obs_length']].
        The 'yaw' and 'obs_yaw' columns are also transformed.

        Args:
            phy_df: a dataframe in physical coordinate.
            seq_center: SeqCenter of the sequence dataframe.

        Returns:
            DataFrame: the dataframe with target columns transformed to
                local physical coordinate system.
        """
        # x-y coordinate tuples to transform
        pos_to_transform = [
            ["pos_x", "pos_y"],
            ["ego_x0", "ego_y0"],
            ["ego_x1", "ego_y1"],
            ["ego_x2", "ego_y2"],
            ["ego_x3", "ego_y3"],
            ["x", "y"],
            ["x0", "y0"],
            ["x1", "y1"],
            ["x2", "y2"],
            ["x3", "y3"],
            ["width", "length"],
            ["obs_width", "obs_length"],
        ]

        col_names = [col for item in pos_to_transform for col in item]
        assert set(col_names).issubset(phy_df.columns), (
            "Input DataFrame " f"doesn't have certain columns in {col_names}"
        )

        out_df = phy_df.copy()
        for pos in pos_to_transform[:-2]:
            phy_car_array = phy_df[pos].values
            # If the columns are full of nan, it means those columns don't
            # have the desired data in the original dataset, give them 0s to
            # make sure that these columns still have values.
            if np.all(np.isnan(phy_car_array)):
                local_car_array = np.zeros_like(phy_car_array)
            else:
                local_car_array = TdtCoordHelper.global_phy_to_local_phy(
                    phy_car_array, seq_center
                )
            out_df[f"{pos[0]}"] = local_car_array[:, 0]
            out_df[f"{pos[1]}"] = local_car_array[:, 1]

        out_df["yaw"] = phy_df["yaw"] - seq_center.yaw
        out_df["obs_yaw"] = phy_df["obs_yaw"] - seq_center.yaw
        out_df["width"] = phy_df["width"]
        out_df["length"] = phy_df["length"]
        out_df["obs_width"] = phy_df["obs_width"]
        out_df["obs_length"] = phy_df["obs_length"]

        return out_df

    @staticmethod
    def get_bbox_corners(
        x: float,
        y: float,
        yaw: float,
        length: float,
        width: float,
        clockwise: Optional[bool] = True,
    ) -> Tuple[np.ndarray]:
        """Get an object bounding box corners' coordinates.

        Args:
            x: object center's x coordinate.
            y: object center's y coordinate.
            yaw: object's heading orientation along x-axis
                counterclockwise
            length: object's physical length along global physical
                x-axis when object's heading points towards x-axis positive
                direction (yaw=0).
            width: object's physical width along global physical
                y-axis when object's heading points towards x-axis positive
                direction (yaw=0).
            clockwise: whether or not the corner points are
                sorted clockwise (in bird-eye view). Defaults to True.

        Returns:
            corners_x: 1D array containing [x0, x1, x2, x3]
            corners_y: 1D array containing [y0, y1, y2, y3]
        """
        # Construct an base object centered at origin with 0 rotation.
        if clockwise:
            base_corners_x = np.array([length, length, -length, -length]) / 2
            base_corners_y = np.array([width, -width, -width, width]) / 2
        else:
            base_corners_x = np.array([length, -length, -length, length]) / 2
            base_corners_y = np.array([width, width, -width, -width]) / 2
        # Use object affine transform to get object's bbox corners.
        corners_x, corners_y = Affine2D.obj_rotate(
            base_corners_x, base_corners_y, yaw
        )
        corners_x, corners_y = Affine2D.obj_translate(
            corners_x, corners_y, x, y
        )
        return corners_x, corners_y


def cross_product(
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate the cross product of two vectors.

                     p0 (x0, y0)
                      *
                     /
                    /
        *----------*
    p2 (x2, y2)   p1 (x1, y1)

    Args:
        x0: x coordinate of point 0.
        y0: y coordinate of point 0.
        x1: x coordinate of point 1.
        y1: y coordinate of point 1.
        x2: x coordinate of point 2.
        y2: y coordinate of point 2.

    Returns:
        cross product of vector p1->p2 and vector p1->p0.
    """
    # Vector p1->p2
    p1p2_x = x2 - x1
    p1p2_y = y2 - y1

    # Vector p1->p0
    p1p0_x = x0 - x1
    p1p0_y = y0 - y1

    return p1p2_x * p1p0_y - p1p2_y * p1p0_x


def is_point_in_convex_polygon(
    xp: float, yp: float, vertex_x: np.ndarray, vertex_y: np.ndarray
) -> bool:
    """Determine whether the point is inside the convex polygon.

    For details, please refer to the following example.

    Example:
                            D (xd, yd)
                            *
                         *     *
           (xe, ye) E *           * C (xc, yc)
                      *           *
                      *    *p     *
                      *           *
           (xa, ya) A ************* B (xb, yb)

        vertex_x            : [xa, xb, xc, xd, xe]
        vertex_y            : [ya, yb, yc, yd, ye]
        np.roll(vertex_x, 1): [xe, xa, xb, xc, xd]
        np.roll(vertex_y, 1): [ye, ya, yb, yc, yd]
        cross product of AE : cross_product(xp, yp, xa, ya, xe, ye)
        cross product of BA : cross_product(xp, yp, xb, yb, xa, ya)
        cross product of CB : cross_product(xp, yp, xc, yc, xb, yb)
        cross product of DC : cross_product(xp, yp, xd, yd, xc, yc)
        cross product of ED : cross_product(xp, yp, xe, ye, xd, yd)

    NOTE: The points inside the convex polygon are all on the same side of the
    vector where the convex polygon's sides are located (provided that the
    vectors where the sides are calculated are in the same direction, both
    clockwise or both counter-clockwise), which can be solved by cross product.

    Args:
        xp: x coordinate of point p.
        yp: y coordinate of point p.
        vertex_x: x coordinate of the convex polygon's
                points, shape is [N,]
        vertex_y: y coordinate of the convex polygon's
                points, shape is [N,]

    Returns:
        whether the point is inside the convex polygon.
    """
    # Calculate the cross product
    cross_product_results = []
    for x0, y0, x1, y1 in zip(
        vertex_x, vertex_y, np.roll(vertex_x, 1), np.roll(vertex_y, 1)
    ):
        cross_product_results.append(cross_product(xp, yp, x0, y0, x1, y1))

    flag_0 = np.all(np.array(cross_product_results) > 0)
    flag_1 = np.all(np.array(cross_product_results) < 0)

    if flag_0 or flag_1:
        return True
    else:
        return False


def cal_point_curvature(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate curvature based on three points.

    Args:
        x: the x coordinates, shape is [3]
        y: the y coordinates, shape is [3]

    Returns:
        kappa: the curvature.
    """
    t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
    t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

    mat = np.array([[1, -t_a, t_a ** 2], [1, 0, 0], [1, t_b, t_b ** 2]])
    inv_mat = np.linalg.inv(mat)

    a = np.matmul(inv_mat, x)
    b = np.matmul(inv_mat, y)

    kappa = (
        2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2.0 + b[1] ** 2.0) ** (1.5)
    )
    return kappa


def cal_traj_curvature(
    x_array: np.ndarray, y_array: np.ndarray, cal_rad: Optional[bool] = False
) -> float:
    """Calculate curvature or curvature radius of a trajectory.

    Args:
        x_array: the x coordinates, shape is [traj_len]
        y_array: the y coordinates, shape is [traj_len]
        cal_rad: whether or not to calculate curvature
            radius. Default to False

    Returns:
        kappa: the curvature.
    """
    assert len(x_array) == len(y_array)
    assert len(x_array) > 2
    num_pts = len(x_array) - 2
    curv_array = np.zeros([num_pts])
    for i in range(num_pts):
        k = cal_point_curvature(x_array[i : (i + 3)], y_array[i : (i + 3)])
        curv_array[i] = 1 / k if cal_rad else k
    return curv_array


def normalize_yaw(yaw: float) -> float:
    """Convert the yaw to [-pi, pi].

    Args:
        yaw: the yaw [rad].
    """
    while yaw >= np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw


def normalize_yaw_array(yaw_array: np.ndarray) -> np.ndarray:
    """Convert the yaw_array to [-pi, pi].

    Args:
        yaw: the yaw [rad].
    """
    for i in range(len(yaw_array)):
        yaw_array[i] = normalize_yaw(yaw_array[i])
    return yaw_array


def trace_and_concat_driveline(
    obs_x: float,
    obs_y: float,
    start_lane_id: str,
    navi_data: Dict,
    from_this_lane: Optional[bool] = False,
) -> Tuple[List]:
    """Trace and concat driveline base on navi information.

    The `navi_data` should contain the logical or virtual lane area of
    the current scenario. Each lane is bound with a driveline
    (represented by dense trajectory points) and have a list of its
    successor lane segments.

    This function will start from a specifical coordinates and a lane,
    then trace all the successor lane segments by DFS. Once reaches the
    edge lane segment (no successor), all the drivelines along this way
    are concatenated as one navigation path.

    Args:
        obs_x: the obstacle x coordinate. (in VCS coordinates)
        obs_y: the obstacle y coordinate. (in VCS coordinates)
        start_lane_id: the lane id of the start lane. Usually, this
            lane is the one that the obstacle locates in. But if the
            obstacle is in the intersection scenario, it may be located in
            more than one virtual lane. The "navigation path" may lead it
            to a wrong direction. In this case, a reliable start lane may
            be the last logical lane the obstacle has passed.
        navi_data: the navigation data. The key is the lane id, the
            value is a dict contains the following keys:
            1. "succ" (list): the successor lane ids.
            2. "drive_line (np.array): the drive line of this lane.
            3. "bounding_box" (np.array): the bounding box of this lane.
        from_this_lane: whether to start from this lane.

    Return:
        all_drivelines: all navigation drive lines.
        all_lane_chains: all navigation lane chains.
    """
    if start_lane_id not in navi_data:
        return None, None
    if from_this_lane:
        start_lane = [start_lane_id]
    else:
        start_lane = navi_data[start_lane_id]["succ"]
    drivelines = []
    all_drivelines = []
    all_lane_chains = []
    valid_chain = []

    def try_get_next_driveline_segment(navi_data, cur_id):
        if (cur_id in valid_chain) or (
            cur_id not in navi_data and len(drivelines) > 0
        ):
            cur_driveline = copy.deepcopy(drivelines)
            cur_driveline = np.concatenate(cur_driveline, axis=0)
            all_drivelines.append(cur_driveline)
            all_lane_chains.append(copy.deepcopy(valid_chain))
            return

        if len(navi_data[cur_id].get("drive_line")) == 0:
            return
        cur_driveline = navi_data[cur_id]["drive_line"][:, 0:2]
        if not len(cur_driveline):
            return
        if navi_data[cur_id].get("succ") is not None:
            cur_succ = navi_data[cur_id]["succ"]
        else:
            cur_succ = []
        driveline_diff = cur_driveline - np.array([obs_x, obs_y])[None, :]
        driveline_diff = np.sqrt(np.sum(driveline_diff ** 2, axis=-1))
        nearest_idx = np.argmin(driveline_diff)
        drivelines.append(cur_driveline[nearest_idx:, :])
        valid_chain.append(cur_id)

        for succ_id in cur_succ:
            try_get_next_driveline_segment(navi_data, succ_id)
        drivelines.pop()
        valid_chain.pop()

    for l_id in start_lane:
        try_get_next_driveline_segment(navi_data, l_id)
    if not len(all_drivelines):
        return None, None
    else:
        return all_drivelines, all_lane_chains


def interpolate_trajectory_from_path(
    drive_line: np.ndarray, sampled_s: np.ndarray
) -> Tuple[np.ndarray]:
    """Interpolate a trajectory based on a path.

    In this function, we sample the path and interpolate the path
    coordinates to generate trajectories. Here, we decouple the
    tangential and radial movement of a curve trajectory and tangential
    consider the tangential displacement.

    We first calculate the tangential displacement of the given path.
    Then we generate a sample target array based on obstacle kinematics.
    Finally, find the context path points of each sample target and
    interpolate the path point coordinates.

    Args:
        drive_line: the driveline, shape is [length, 2]
        sampled_s: the scalar tangential displacement of the obstacle,
            shape is [traj_len]

    Returns:
        sampled_driveline: the sampled and interpolated trajectories,
            shape is [traj_len, 2]
        sampled_mask: the mask, shape is [traj_len, 2]
    """
    traj_len = len(sampled_s)
    sampled_driveline = np.zeros([traj_len, 2])
    sampled_mask = np.zeros([traj_len])

    def find_first_larger_idx(
        sorted_data: Union[np.ndarray, List[float]], value: float, idx: int
    ) -> int:  # noqa: D205,D400
        """Find the first idx in the sorted data array that larger than a
        certain value.

        Args:
            sorted_data: the sorted data (ascending) array.
            value: the value to compare.
            idx: the start index of the traversing.

        Return:
            ret_idx (int): the first idx that larger than `value`.
        """
        len_data = len(sorted_data)
        sorted_data = np.array(sorted_data)
        if idx >= len_data:
            return None
        larger_indices = np.where(sorted_data[idx:] > value)[0]
        if len(larger_indices):
            return larger_indices[0] + idx
        else:
            return -1

    diff_s = np.sqrt(np.sum(np.diff(drive_line, axis=0) ** 2, axis=-1))
    cumsum_s = np.array([0] + list(np.cumsum(diff_s)))
    for i in range(traj_len):
        idx = find_first_larger_idx(cumsum_s, sampled_s[i], 0)
        if idx is None or idx <= 0:
            break
        idx_start, idx_end = idx - 1, idx
        sampled_driveline[i, :] = drive_line[idx_start] * (
            cumsum_s[idx_end] - sampled_s[i]
        ) + drive_line[idx_end] * (sampled_s[i] - cumsum_s[idx_start])
        sampled_driveline[i, :] /= cumsum_s[idx_end] - cumsum_s[idx_start]
        sampled_mask[i] = 1
    return sampled_driveline, sampled_mask


def cxcywh_to_x1y1x2y2(boxes: torch.Tensor) -> torch.Tensor:
    """Convert to x1y1x2y2 from cxcywh.

    Args:
        boxes: the boxes need to be
            converted, shape is [batch_size, k_value, step, 4].
            the elements in the last dimension of boxes are:
            boxes[:, :, :, 0]: x coord of the center point of bbox,
            boxes[:, :, :, 1]: y coord of the center point of bbox,
            boxes[:, :, :, 2]: width of bbox,
            boxes[:, :, :, 3]: height of bbox.

    Returns:
        new_boxes: the boxes in x1y1x2y2, shape is
            [batch_size, k_value, step, 4].
            the elements in the last dimension of new_boxes are:
            new_boxes[:, :, :, 0]: x coord of the top left point of bbox,
            new_boxes[:, :, :, 1]: y coord of the top left point of bbox,
            new_boxes[:, :, :, 2]: x coord of the bottom right point of bbox,
            new_boxes[:, :, :, 3]: y coord of the bottom right point of bbox.
    """
    assert (
        boxes.shape[3] == 4
    ), f"boxes.shape[3] must be 4, but get {boxes.shape[3]}"
    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, :, :, 0] = boxes[:, :, :, 0] - boxes[:, :, :, 2] / 2
    new_boxes[:, :, :, 1] = boxes[:, :, :, 1] - boxes[:, :, :, 3] / 2
    new_boxes[:, :, :, 2] = boxes[:, :, :, 0] + boxes[:, :, :, 2] / 2
    new_boxes[:, :, :, 3] = boxes[:, :, :, 1] + boxes[:, :, :, 3] / 2

    return new_boxes


def x1y1x2y2_to_cxcympb(boxes: torch.Tensor) -> torch.Tensor:
    """Convert to cxcympb from x1y1x2y2.

    [NOTE]:mpb is the middle point of bottom.

    Args:
        boxes: the boxes need to be converted,
            shape is [batch_size, k_value, step, 4].
            the elements in the last dimension of boxes are:
            boxes[:, :, :, 0]: x coord of the top left point of bbox,
            boxes[:, :, :, 1]: y coord of the top left point of bbox,
            boxes[:, :, :, 2]: x coord of the bottom right point of bbox,
            boxes[:, :, :, 3]: y coord of the bottom right point of bbox.

    Returns:
        new_boxes : the boxes in cxcympb,
            shape is [batch_size, k_value, step, 4].
            the elements in the last dimension of new_boxes are:
            new_boxes[:, :, :, 0]: x coord of the center point of bbox,
            new_boxes[:, :, :, 1]: y coord of the center point of bbox,
            new_boxes[:, :, :, 2]: x coord of the mpb of bbox,
            new_boxes[:, :, :, 3]: y coord of the mpb of bbox.
    """
    assert (
        boxes.shape[3] == 4
    ), f"boxes.shape[3] must be 4, but get {boxes.shape[3]}"
    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, :, :, 0] = (boxes[:, :, :, 0] + boxes[:, :, :, 2]) / 2
    new_boxes[:, :, :, 1] = (boxes[:, :, :, 1] + boxes[:, :, :, 3]) / 2
    new_boxes[:, :, :, 2] = (boxes[:, :, :, 0] + boxes[:, :, :, 2]) / 2
    new_boxes[:, :, :, 3] = boxes[:, :, :, 3]

    return new_boxes


def bbox_denormalize(
    boxes: torch.Tensor,
    normalize_type: str = "zero-one",
    w: int = 1920,
    h: int = 1080,
) -> torch.Tensor:
    """Denormalize function.

    Args:
        boxes: the boxes need to be denormalized,
            shape is [batch_size, k_value, step, 4].
            elements in boxes[:, :, :, 0:4] are the coords
            in any style, such as x1y1x2y2 or cxcywh.
        normalize_type: the normalize type used in dataset, only zero-one
            is supported now.
        w: the image width.
        h: the image height.

    Returns:
        new_boxes: the boxes denormalized,
            shape is [batch_size, k_value, step, 4].
            elements in new_bbox[:, :, :, 0:4] are the coords in any style,
            such as x1y1x2y2 or cxcywh.
    """
    assert (
        normalize_type == "zero-one"
    ), f"only zero-one denormalize supported but get {normalize_type}"
    new_bbox = torch.zeros_like(boxes)
    if normalize_type == "zero-one":
        new_bbox[:, :, :, 0] = boxes[:, :, :, 0] * w
        new_bbox[:, :, :, 1] = boxes[:, :, :, 1] * h
        new_bbox[:, :, :, 2] = boxes[:, :, :, 2] * w
        new_bbox[:, :, :, 3] = boxes[:, :, :, 3] * h
    return new_bbox
