# Copyright (c) Horizon Robotics. All rights reserved.

import os
from collections.abc import Iterable
from colorsys import hsv_to_rgb
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from hat.core.traj_pred_point_level_func import (  # noqa: E501; fmt:skip
    TruncateTrajBounce,
    TruncateTrajByLim,
)
from hat.core.traj_pred_typing import FILTER_FLAG, SeqCenter, SeqIndex
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.core.traj_pred_viz_utils import (
    PlotBehav,
    PlotExpandObs,
    PlotGT,
    PlotImage,
    PlotObjBBox,
    PlotTraj,
    trans_coord_img_to_phy,
)


class ImageRendererWrapper:
    """Wrapper of visualization frameworks."""

    # fmt:off

    # The column definition of the origin dataframes.
    # -- physical-coordinate pose related dataFrame columns.
    PHY_COLS = [
        "pos_x", "pos_y", "ego_x0", "ego_y0", "ego_x1", "ego_y1", "ego_x2",
        "ego_y2", "ego_x3", "ego_y3", "x", "y", "x0", "y0", "x1", "y1",
        "x2", "y2", "x3", "y3", "width", "length", "obs_width", "obs_length",
        "yaw", "obs_yaw"
    ]
    # -- image-coordinate pose related dataFrame columns.
    IMG_COLS = [f"img_{col}" for col in PHY_COLS]

    # The column definition of the predicted dataframes.
    PRED_DF_COLS = [
        "track_id", "frame_id", "target_frame_id", "infer_id",
        "pos_x", "pos_y", "probability", "anc_idx"]

    # The column definition of the saved dataframes.
    SAVE_DF_COLS = ["track_id", "frame_id", "timestamp"]
    for i in range(12):
        SAVE_DF_COLS += [f"future{i+1}_x", f"future{i+1}_y"]
    SAVE_DF_COLS.append("prob")

    # fmt:on

    def __init__(
        self,
        height: int,
        width: int,
        resolution: float,
        num_workers: int = 1,
        image_offset: np.array = None,
        reverse: bool = False,
        ego_track_id: int = -42,
        apply_traj_truncation: bool = False,
        prepend_last_groundtruth: bool = False,
        save_dataframe: bool = False,
        save_dataframe_path: Optional[str] = None,
        fade_eps: float = 1e-3,
        fps: float = 1,
        del_img_dir: bool = True,
        min_pred_prob: float = 0,
        use_behav_head: bool = False,
        freq_ratio: int = 1,
        first_fut_frame_idx: int = 4,
        fut_len: int = 12,
    ):
        """Initialize method.

        Args:
            height: height of context image.
            width: width of context image.
            resolution: resolution of context image.
            num_workers: render workers.
            image_offset: the offset between vcs and image coordinate system.
            reverse: if the orientation between vcs and image coordinate system
                is oppisite, recerse should be True.
            ego_track_id: the track id of the ego vehicle.
            apply_traj_truncation: whether to conduct truncation defined in
                vizlib's renderer on predicted trajectories.
            prepend_last_groundtruth: whether to connect ground truth with
                trajectories.
            save_dataframe: whether to save the dataframe of predicted
                trajectories.
            save_dataframe_path: the path to save the predicted trajectory
                dataframes.
            fade_eps: a very small constant value to avoid to divide zero.
            fps: the FPS of the output video.
            del_img_dir: whether to delete saved image directory after
                generating video.
            min_pred_prob: the minimum probability of a predicted result that
                can be shown in the video.
            use_behav_head: if set True, renderer will support showing
            predicted behaviors.
            freq_ratio: ratio of data frequency compared to 2Hz.
            first_fut_frame_idx: the first future frame index.
            fut_len: the len of future frames.
        """
        if save_dataframe:
            assert os.path.exists(save_dataframe_path), (
                f"The path to save dataframe {save_dataframe_path} "
                "is not found."
            )
        self.resolution = resolution
        self.height = height
        self.width = width
        self.offset = [0, 0]
        self.apply_traj_truncation = apply_traj_truncation
        self.image_offset = image_offset
        self.reverse = reverse
        self.ego_track_id = ego_track_id
        self.save_dataframe = save_dataframe
        self.save_dataframe_path = save_dataframe_path
        self.fade_eps = fade_eps
        self.fps = fps
        self.del_img_dir = del_img_dir
        self.min_pred_prob = min_pred_prob
        self.use_behav_head = use_behav_head
        self.freq_ratio = freq_ratio
        self.first_fut_frame_idx = first_fut_frame_idx
        self.fut_len = fut_len

        self.__num_workers = num_workers
        self.__num_colors = 500

        if self.use_behav_head:
            self.PRED_DF_COLS += ["lat_behavior", "lon_behavior"]

        # Build a track_id to color mapping, different id has different hue.
        hues = np.arange(0, 1, 1 / self.__num_colors)
        # np.random.shuffle(hues)
        self.__colors = list(map(lambda hue: hsv_to_rgb(hue, 1.0, 1.0), hues))
        self.__epsilon = 0.1
        self.__save_final_trajs_path = None
        self.__prepend_last_groundtruth = prepend_last_groundtruth

    def id_to_color(self, track_id: int) -> Tuple:
        """Map track_id to colors.

        We only have limited number of colors. Therefore for track_id >
        num_colors we take the modulo as index.

        Args:
            track_id: tracking id.

        Returns:
            three-tuple of r, g, b colors.
        """
        return self.__colors[int(track_id) % self.__num_colors]

    def __call__(
        self,
        gt_df: List[DataFrame],
        seq_center: List[SeqCenter],
        pred_trajectories: List[ndarray],
        track_ids: List[int],
        agent_classes: List[int],
        track_status: List[int],
        image: List[ndarray],
        dataset_index: List[int],
        seq_index: List[Any],
        seq_mask: List,
        seq_prob: List,
        anc_indices: List,
        video_dir: List[str],
        filename_info: Optional[List[Dict]] = None,
        track_yaw: Optional[List[Dict]] = None,
        pred_lat_behavs: List[ndarray] = None,
        pred_lon_behavs: List[ndarray] = None,
        exp_important_obs: Optional[List[tuple]] = None,
    ):
        """Call `Renderer` to generate videos.

        This function firstly transforms `pred_trajectories` to `pred_df`
        whose type is `pd.DataFrame`. Then pack all the data required by
        `Renderer` into iterator and generate videos.

        Args:
            gt_df: ground-truth dataframe.
            seq_center: sequence center coordinates.
            pred_trajectories: predicted trajectories.
            track_ids: track id list.
            agent_classes: agent class list.
            track_status: the prediction status of different obstacles.
            image: context image.
            dataset_index: the indices of the sequences in the original
                dataset.
            seq_index: sequence index.
            seq_mask: sequence mask.
            seq_prob: sequence probabilities.
            anc_indices: the indices of the used anchors in the sequence.
            video_dir: prefix to the directory where the output videos will
                be saved.
            filename_info: the information to be displayed in the file names.
            track_yaw: the selected yaw array. If it is None, the class will
                use the perception yaw.
            pred_lat_behavs: lateral behaviors array. If it is None, renderer
                will not show behavior information.
            pred_lat_behavs: longitudinal behaviors array. Ifit is None,
                renderer will not show behavior information.
            exp_important_obs: the expand important obstacles.
        """
        # Check.
        if not isinstance(gt_df, Iterable):
            raise TypeError(
                "The input ground-truth dataframe should be iterable."
            )
        if len(gt_df) == 0:
            return
        assert (
            len(gt_df)
            == len(seq_center)
            == len(pred_trajectories)
            == len(track_ids)
            == len(track_status)
            == len(image)
            == len(dataset_index)
            == len(seq_index)
        ), "Wrong shape."
        if track_yaw is not None and len(track_yaw) != 0:
            assert len(track_ids) == len(
                track_yaw
            ), "The selected yaw array has a wrong shape"
        if not (pred_lat_behavs is None or pred_lon_behavs is None):
            assert (
                len(track_ids) == len(pred_lat_behavs) == len(pred_lon_behavs)
            ), "The predicted behavs array has a wrong shape"
        if exp_important_obs is not None and len(exp_important_obs) > 0:
            assert len(track_ids) == len(
                exp_important_obs
            ), "The expanded import obstacle has a wrong shape"

        # Collect the data for video rendering.
        event_road = []
        infer_enum = []
        if self.save_dataframe:
            final_save_df = pd.DataFrame()
        for i in range(len(gt_df)):
            # Get the prediction result dataframe from the ground-truth
            # dataframe and the original prediction results.
            tmp_yaw = track_yaw[i] if (track_yaw is not None) else None
            pred_lat_behav, pred_lon_behav = None, None
            if self.use_behav_head:
                pred_lat_behav = pred_lat_behavs[i]
                pred_lon_behav = pred_lon_behavs[i]
            pred_df, save_df = self.construct_pred_df(
                gt_df[i],
                track_ids[i],
                agent_classes[i],
                pred_trajectories[i],
                seq_index[i],
                seq_mask[i],
                seq_prob[i],
                anc_indices[i],
                seq_center[i],
                self.ego_track_id,
                self.save_dataframe,
                tmp_yaw,
                pred_lat_behav,
                pred_lon_behav,
                freq_ratio=self.freq_ratio,
                fut_len=self.fut_len,
            )
            if self.save_dataframe:
                final_save_df = final_save_df.append(save_df)
            phy_x_col = pred_df.columns.get_loc("pos_x")
            phy_y_col = pred_df.columns.get_loc("pos_y")
            phy_pred_trajs = pred_df.values[:, [phy_x_col, phy_y_col]]
            pred_df.iloc[
                :, [phy_x_col, phy_y_col]
            ] = TCH.global_phy_to_local_img(
                phy_pred_trajs,
                seq_center[i],
                self.image_offset,
                self.resolution,
                reverse=self.reverse,
            )
            event_road.append(image[i])
            infer_enum.append(pred_df)

        if self.save_dataframe:
            final_save_df.to_csv(
                os.path.join(self.save_dataframe_path, "result.csv")
            )

        # Render the video.
        # -- note: we set the fixed resolution 1.0 here because we
        # already transform the dataframe to the image coordinates.
        self.render_event(
            dataset_index=dataset_index,
            seq_index=seq_index,
            event_road=event_road,
            pred_df=infer_enum,
            gt_df=gt_df,
            track_status=track_status,
            video_dir=video_dir,
            resolution=1.0,
            offset=self.offset,
            apply_traj_truncation=self.apply_traj_truncation,
            filename_info=filename_info,
            exp_important_obs=exp_important_obs,
        )

    @staticmethod
    def extract_future_trajs(
        pred_df: DataFrame, frame_id: int, track_id: int
    ) -> List[np.ndarray]:
        """Extract trajectories from prediction result DataFrame.

        This method extract all predicted trajectories for an object with
        tracking ID `track_id` forcasted at frame ID `frame_id`. There may be
        multiple random inferences performed, we extracted them all.

        `pred_df` is the prediction output. Each row represents the predicted
        coordinate for one object at one target frame id forcasted at a
        historical frame id. The prediction output `pred_df` should have the
        following columns:
            frame_id: frame index
            target_frame_id: the frame index of target frame for prediction
            track_id: tracking ID
            infer_id: ID from random inference
            pos_x: predicted x position
            pos_y: predicted y position

        Args:
            pred_df: prediction output dataframe
            frame_id: frame id
            track_id: tracking id
        Returns:
            trajs: list of predicted trajectories. Each element is a
                numpy.array representing a trajectory that is performed from
                current frame_id.
        """
        # Column assertions.
        set_musthaves = [
            "frame_id",
            "target_frame_id",
            "track_id",
            "infer_id",
            "pos_x",
            "pos_y",
        ]
        set_musthaves = set(set_musthaves)
        set_columns = set(pred_df.columns)
        assert set_musthaves.issubset(set_columns), (
            "Prediction DataFrame must have the following "
            "columns: {}".format(set_musthaves)
        )
        # Mask out the trajectory for current frame_id and track_id.
        mask = (pred_df.frame_id == frame_id) & (pred_df.track_id == track_id)
        pred_df = pred_df[mask]
        # See how many random inferences is performed.
        infer_ids = pred_df.infer_id.unique()
        # Iterate through all inferences and extract trajectory.
        trajs = []
        probs = []
        anc_indices = []
        for infer_id in infer_ids:
            # Mask out current inference.
            mask = pred_df.infer_id == infer_id
            infer_df = pred_df[mask]
            # Sort by the frame_id of prediction target (ascending order).
            infer_df = infer_df.sort_values(by=["target_frame_id"])
            # Extract trajectory coordinate.
            pos_x = infer_df.pos_x.values
            pos_y = infer_df.pos_y.values
            trajs.append(np.array([pos_x, pos_y]))
            prob = infer_df.probability.values[0]
            probs.append(prob)
            anc_idx = infer_df.anc_idx.values[0]
            anc_indices.append(anc_idx)
        return trajs, probs, anc_indices

    @staticmethod
    def extract_cur_behavs(
        pred_df: DataFrame,
        gt_behavs_df: DataFrame,
        frame_id: int,
        ego_track_id: int,
    ) -> Dict:
        """Extract behaviors from prediction result DataFrame.

        This method extract all predicted behaviors at frame ID `frame_id`.
        There may be multiple objects, we extracted them all. `pred_df` is
        the prediction output. The prediction output `pred_df` should have
        the following columns:
            frame_id: frame index
            track_id: tracking ID
            lat_behavior: lateral behavior
            lon_behavior: longitudinal behavior

        Args:
            pred_df: prediction output dataframe.
            gt_behavs_df: groundtruth behaviors dataframe.
            frame_id: the frame index of input sample.
            ego_track_id: track id of the ego car.

        Returns:
            pred_gt_behavs: dict of predicted behaviors. Each key is
                the track_id. Each element is lateral, longitudinal
                behaviors flag at the current frame.
        """
        # Column assertions.
        set_musthaves = [
            "frame_id",
            "track_id",
            "lat_behavior",
            "lon_behavior",
        ]
        set_musthaves = set(set_musthaves)
        set_columns = set(pred_df.columns)
        assert set_musthaves.issubset(set_columns), (
            "Prediction DataFrame must have the following "
            "columns: {}".format(set_musthaves)
        )
        # Extract behaviors of different obstacles
        mask = pred_df.frame_id == frame_id
        pred_df = pred_df[mask]
        track_ids = list(pred_df.track_id.unique())
        pred_gt_behavs = {}
        for track_id in track_ids:
            if track_id == ego_track_id:
                continue
            mask = pred_df.track_id == track_id
            single_obs_pred_df = pred_df[mask]
            mask = gt_behavs_df.track_id == track_id
            single_obs_gt_df = gt_behavs_df[mask]
            pred_gt_behavs[int(track_id)] = np.concatenate(
                [
                    single_obs_gt_df.values[0, 1:].astype("int"),
                    single_obs_pred_df[["lat_behavior", "lon_behavior"]]
                    .values[0, :]
                    .astype("int"),
                ],
                axis=0,
            )
        return pred_gt_behavs

    @staticmethod
    def construct_pred_df(
        gt_df: DataFrame,
        track_ids: List[int],
        agent_classes: List[int],
        pred_trajectories: ndarray,
        seq_index: Any,
        seq_mask: ndarray,
        seq_prob: ndarray,
        anc_indices: ndarray,
        seq_center: SeqCenter,
        ego_track_id: int,
        save_dataframe: bool,
        selected_yaw: Optional[Dict] = None,
        pred_lat_behav: Optional[ndarray] = None,
        pred_lon_behav: Optional[ndarray] = None,
        freq_ratio: int = 1,
        fut_len: int = 12,
    ) -> Tuple[DataFrame]:
        """Construct pred_df for visualization with predicted trajectories.

        Args:
            gt_df: Ground-truth DataFrame obtained from dataloader.
            track_ids: Track ids in the sequence.
            agent_classes: agent classes in the sequence.
            pred_trajectories: predicted trajectories output by model
            seq_index: SeqIndex for this sequence.
            seq_mask: the masks of the sequence.
            seq_prob: the probabilities of the sequence.
            anc_indices: the used anchor indices of the sequence.
            seq_center: the sequence center.
            ego_track_id: the track id of the ego vehicle.
            classify_by_shape: whether to classify pedestrains by shape.
            ped_shape_thr: the maximum length and width of pedestrains.
            save_dataframe: whether to save the dataframe.
            selected_yaw: the selected yaw array.
            pred_lat_behav: predicted lateral behaviors of obstacles
                in one sample.
            pred_lon_behav: predicted longitudinal behaviors of obstacles
                in one sample.
            freq_ratio: ratio of data frequency compared to 2Hz.
            fut_len: the len of future frames.

        Returns:
            pred_df (Dataframe): the dataframe contains the predicted result
                for visualization.
            save_df (Dataframe): the dataframe to save. If the `save_dataframe`
                is False, this parameter will be None.
        """
        (
            obj_num,
            anchor_num,
            sample_times,
            timestep,
            dim,
        ) = pred_trajectories.shape
        last_context_frame_id = gt_df["last_context_frame_id"].values[0]
        if isinstance(seq_index, SeqIndex):
            frame_slice = seq_index.frame_slice
            frame_slice_start = int(frame_slice.start)
            frame_slice_stop = int(frame_slice.stop)
            frame_slice_step = int(frame_slice.step)
            frame_ids = list(
                range(frame_slice_start, frame_slice_stop, frame_slice_step)
            )
            future_len = (frame_slice_stop - last_context_frame_id - 1) // (
                frame_slice_step * freq_ratio
            )
        elif isinstance(seq_index, List):
            frame_ids = seq_index
            future_len = fut_len
        else:
            raise ValueError("Unsupported type of seq_index.")

        last_context_frame_idx = frame_ids.index(last_context_frame_id)
        rows = []
        if save_dataframe:
            save_rows = []
            lcf_mask = gt_df["frame_id"] == last_context_frame_id
            timestamp = gt_df.loc[lcf_mask, "timestamp"].values[0]
        pred_trajectories = pred_trajectories.reshape(
            (obj_num, -1, timestep, dim)
        )

        gt_df_vals = gt_df.values
        gt_df_cols = gt_df.columns
        x_col = gt_df_cols.get_loc("x")
        y_col = gt_df_cols.get_loc("y")
        obs_yaw_col = gt_df_cols.get_loc("obs_yaw")
        for i, track_id in enumerate(track_ids):
            # Trans to local phy from car centric
            cur_mask = (gt_df["frame_id"] == last_context_frame_id) & (
                gt_df["track_id"] == track_id
            )
            gt_lctx_vals = gt_df_vals[cur_mask, :]

            phy_obj_center = gt_lctx_vals[
                :, [x_col, y_col, obs_yaw_col]
            ].squeeze()
            if not len(phy_obj_center):
                continue
            phy_obj_coord = phy_obj_center[:2].astype("float")
            phy_obj_yaw = phy_obj_center[-1]

            if selected_yaw is not None:
                if selected_yaw.get(track_id) is not None:
                    phy_obj_yaw = selected_yaw[track_id][-1]
                else:
                    phy_obj_yaw = selected_yaw[ego_track_id][-1]

            trajectories = pred_trajectories[i, :, :, :]
            traj_shape = trajectories.shape
            trajectories = trajectories.reshape([-1, traj_shape[-1]])
            trajectories = TCH.agent_centric_phy_to_global_phy(
                trajectories, phy_obj_coord, phy_obj_yaw
            ).reshape(traj_shape)

            mask = seq_mask[i, :]
            prob = seq_prob[i, :]
            anc_idx = anc_indices[i, :]
            use_behav_head = False
            if not (pred_lat_behav is None or pred_lon_behav is None):
                use_behav_head = True
                each_lat_behav = pred_lat_behav[i]
                each_lon_behav = pred_lon_behav[i]

            frame_id = frame_ids[last_context_frame_idx + 1]
            for k in range(future_len):
                target_frame_id = frame_ids[k + last_context_frame_idx + 1]
                for idx in range(anchor_num * sample_times):
                    if mask[int(idx / sample_times)]:
                        add_row = [
                            track_id,
                            frame_id,
                            target_frame_id,
                            idx,
                            trajectories[idx, k, 0],
                            trajectories[idx, k, 1],
                            prob[int(idx / sample_times)],
                            anc_idx[int(idx / sample_times)],
                        ]
                        if use_behav_head:
                            add_row += [
                                each_lat_behav,
                                each_lon_behav,
                            ]
                        rows.append(add_row)

            if save_dataframe:
                ego_yaw = seq_center.yaw
                offset_x = -(
                    np.cos(ego_yaw) * seq_center.pos_x
                    + np.sin(ego_yaw) * seq_center.pos_y
                )
                offset_y = -(
                    np.cos(ego_yaw) * seq_center.pos_y
                    - np.sin(ego_yaw) * seq_center.pos_x
                )
                global2vcs = np.array(
                    [
                        [np.cos(ego_yaw), np.sin(ego_yaw), offset_x],
                        [-np.sin(ego_yaw), np.cos(ego_yaw), offset_y],
                        [0, 0, 1],
                    ]
                )

                trajectories = trajectories.reshape([-1, traj_shape[-1]]).T
                norm = np.array([1] * trajectories.shape[-1]).reshape([1, -1])
                trajectories = np.concatenate([trajectories, norm], 0)
                vcs_trajectories = np.matmul(global2vcs, trajectories)[:2, :]
                vcs_trajectories = vcs_trajectories.T.reshape(
                    [traj_shape[0], -1]
                )

                for idx in range(anchor_num * sample_times):
                    if mask[int(idx / sample_times)]:
                        save_rows.append(
                            [track_id, last_context_frame_id, timestamp]
                            + vcs_trajectories[idx].tolist()
                            + [prob[int(idx / sample_times)]]
                        )

        pred_df = pd.DataFrame(rows, columns=ImageRendererWrapper.PRED_DF_COLS)

        save_df = None
        if save_dataframe:
            save_df = pd.DataFrame(
                save_rows, columns=ImageRendererWrapper.SAVE_DF_COLS
            )
        return pred_df, save_df

    def plot_image(
        self,
        dataset_index: int,
        seq_index: Any,
        roadnet: np.array,
        pred_df: DataFrame,
        gt_df: DataFrame,
        track_status: Dict,
        plotting_helpers: List[Callable],
        truncation_helpers: List[Callable],
        offset: List,
        image_dir: str,
        exp_important_obs: List[tuple],
    ):
        """Core animation function for updating at one frame.

        This is the core animation function to draw various elements on
        each of the video frames.

        Args:
            dataset_index: the index of the event in the data loader.
            seq_index: index information to slice the gt_df from the
                data loader.
            roadnet ([H, W]): background roadnet image.
            pred_df: prediction result DataFrame.
            gt_df: ground-truth DataFrame.
            track_status: the prediction status of different obstacles.
            plotting_helpers: list of helpers for plotting different
                visual elements.
            truncation_helpers: list of helpers for truncating trajectory.
            offset: offset added to the whole local physical coordinates.
            image_dir: the path to save the rendered image.
            exp_important_obs: the expand important obstacles.
        """
        # Setup figure for visualization.
        fig = plt.figure(dpi=300, figsize=(5, 4))
        ax_l = fig.add_subplot(1, 1, 1)

        plot_roadnet = plotting_helpers[0]
        plot_ego_bbox = plotting_helpers[1]
        plot_obs_bbox = plotting_helpers[2]
        plot_future_pred = plotting_helpers[3]
        plot_future_gt = plotting_helpers[4]
        plot_behav = plotting_helpers[5]
        plot_exp_traj = plotting_helpers[6]
        apply_traj_truncation = len(truncation_helpers) > 0
        if apply_traj_truncation:
            trunc_traj_lim = truncation_helpers[0]
            trunc_traj_bounce = truncation_helpers[1]

        # -- Clear previous frame.
        ax_l.clear()

        last_context_frame_id = gt_df["last_context_frame_id"].values[0]
        lcf_mask = gt_df.frame_id == last_context_frame_id
        lcf_gt_df = gt_df[lcf_mask]
        lcf_gt_df[self.PHY_COLS] = lcf_gt_df[self.IMG_COLS]

        # -- Set current title.
        title = "idx: {}".format(dataset_index)
        if "timestamp" in gt_df.columns:
            timestamp = lcf_gt_df["timestamp"].values[0]
            title += "   timestamp: {}".format(int(np.round(timestamp)))
        ax_l.set_title(title)

        # -- Set label.
        ax_l.set_xlabel("x")
        ax_l.set_ylabel("y")

        # -- Guarantee x and y has same range.
        if len(roadnet.shape) == 2:
            h, w = roadnet.shape
        else:
            h, w, _ = roadnet.shape
        low_xlim = -h / 2 + offset[0]
        high_xlim = h / 2 + offset[0]
        low_ylim = -w / 2 + offset[1]
        high_ylim = w / 2 + offset[1]
        ax_l.set(xlim=(low_xlim, high_xlim), ylim=(low_ylim, high_ylim))

        # -- Plot roadnet.
        plot_roadnet(ax_l, roadnet)

        # -- Extract and plot boxes.
        gt_df_frame = lcf_gt_df
        ego_columns = ["pos_x", "pos_y", "yaw", "length", "width"]
        obs_columns = [
            "x",
            "y",
            "obs_yaw",
            "obs_length",
            "obs_width",
            "track_id",
        ]
        obs_behav_colums = ["track_id", "lat_behav_label", "lon_behav_label"]

        ego_data = gt_df_frame[ego_columns].values.astype("float")
        obs_data = gt_df_frame[obs_columns].values.astype("float")
        if self.use_behav_head:
            gt_behavs_df = gt_df_frame[obs_behav_colums]

        # ---- Plot ego box, also record pos for plotting trajectories
        #   in the next section.
        pos_x, pos_y, yaw, length, width = ego_data[0]
        pos_x += offset[0]
        pos_y += offset[1]
        track_id = int(self.ego_track_id)
        color = self.id_to_color(track_id)
        args = (pos_x, pos_y, yaw, length, width, color, track_id)
        plot_ego_bbox(ax_l, *args)
        ego_pos = np.array([[pos_x], [pos_y]])
        ego_id = track_id
        obs_pos = {ego_id: ego_pos}

        # ---- Plot obstacle box, also record pos for plotting trajectories
        #   in the next section.
        for i in range(obs_data.shape[0]):
            if np.any(np.isnan(obs_data[i])):
                assert (
                    obs_data.shape[0] == 1
                ), "Should only have one row if no obs"
                break
            pos_x, pos_y, yaw, length, width, track_id = obs_data[i]
            pos_x += offset[0]
            pos_y += offset[1]
            track_id = int(np.round(track_id))
            if track_id in track_status:
                cur_status = track_status[track_id]
            else:
                cur_status = 0
            color = self.id_to_color(track_id)
            args = (
                pos_x,
                pos_y,
                yaw,
                length,
                width,
                color,
                track_id,
                cur_status,
            )
            plot_obs_bbox(ax_l, *args)
            # jiajie modified for ego predicting
            # assert track_id not in obs_pos, (
            #     f"Duplicate track ID {track_id} for frame_id {frame}")
            obs_pos[track_id] = np.array([[pos_x], [pos_y]])

        if isinstance(seq_index, SeqIndex):
            frame = last_context_frame_id + seq_index.frame_slice.step
        elif isinstance(seq_index, List):
            frame = seq_index[self.first_fut_frame_idx]
        else:
            raise ValueError("Unsupported type of seq_index.")
        if self.use_behav_head:
            # ---- Extract and Plot obstacle predicted behaviors
            pred_gt_behavs = self.extract_cur_behavs(
                pred_df, gt_behavs_df, frame, self.ego_track_id
            )
            plot_behav(ax_l, pred_gt_behavs)
        # -- Extract, truncate, and plot predicted trajectories.
        cur_pred_df = pred_df[pred_df.frame_id == frame]
        track_ids = list(cur_pred_df.track_id.unique())
        track_ids = track_ids
        for track_id in track_ids:
            track_id = int(track_id)
            color = self.id_to_color(track_id)
            # Extract future trajectories.
            trajs, probs, anc_indices = self.extract_future_trajs(
                pred_df, frame, track_id
            )
            trajs = [traj + np.array(offset).reshape(-1, 1) for traj in trajs]
            max_prob = max(probs)
            min_prob = min(probs)
            for traj, prob, anc_idx in zip(trajs, probs, anc_indices):
                if prob < self.min_pred_prob:
                    continue
                # Prepend current position to make the ploted trajectory
                #   look better.
                if self.__prepend_last_groundtruth and track_id in obs_pos:
                    traj = np.concatenate([obs_pos[track_id], traj], axis=1)
                # Truncate trajectories at roadnet boundaries.
                if apply_traj_truncation:
                    traj = trunc_traj_lim(traj)
                    # Truncate bouncing trajectories.
                    traj = trunc_traj_bounce(traj)
                # Store final trajs for mono vision.
                if self.__save_final_trajs_path:
                    self.__save_final_trajs(
                        traj, gt_df_frame, frame, track_id, color
                    )
                # Plot trajectory on left axis.
                fade = (prob - min_prob + self.fade_eps) / (
                    max_prob - min_prob + self.fade_eps
                )
                plot_future_pred(ax_l, traj, color, fade, anc_idx)

            mask = (gt_df.frame_id >= frame) & (gt_df.track_id == track_id)
            future_track_gt_df = gt_df[mask]
            future_track_pos = future_track_gt_df.loc[
                :, ["img_x", "img_y"]
            ].values
            plot_future_gt(ax_l, future_track_pos, color)

        if exp_important_obs is not None and len(exp_important_obs) > 0:
            for exp_t_id, exp_type, safe_area in exp_important_obs:
                tmp_color = self.id_to_color(exp_t_id)
                # TODO (shengzhe.dai): we only plot the import obstacle
                # that filtered due to "too far", we may add other types
                # in the future.
                if exp_type == FILTER_FLAG["obs_too_far"]:
                    plot_exp_traj(ax_l, safe_area, tmp_color, offset)

        plt.savefig("{}/{}.png".format(image_dir, str(timestamp)))

        # Close the figure. MPL will leave all un-closed figures in memory.
        # Without this fix, it may not work every time when renderring using
        # multiple processes.
        plt.close(fig=fig)
        return True

    def render_event(
        self,
        dataset_index: List[int],
        seq_index: List[Any],
        event_road: List[np.array],
        pred_df: List[DataFrame],
        gt_df: List[DataFrame],
        track_status: List,
        video_dir: str,
        resolution: float = 1.0,
        offset: tuple = (0, 0),
        apply_traj_truncation: bool = True,
        filename_info: Optional[str] = None,
        exp_important_obs: Optional[List[tuple]] = None,
    ) -> bool:
        """Render the ground-truth and predicted result of one event.

        The required columns of pred_df defined in self.PRED_DF_COLS.
        The required columns of pred_df defined in self.PHY_COLS and
        self.IMG_COLS.

        To be consistent about the coordinate system, all processing in this
        method is performed in the **local physical coordinate**, which is
        centered at the ego vehicle position at one frame in a sequence. The
        ground-truth and predicted DataFrame should be already transformed to
        this coordinate. The roadnet image is still in the image coordinate and
        thus needs to be transformed to local physical coordinate to
        facilitate image boundary limiting operations.

        Args:
            dataset_index: the index of the event in the data loader.
            seq_index: index information to slice the gt_df from the data
                loader.
            event_road: the list of roadnet image.
            pred_df: the dataframes contain prediction results.
            gt_df: the dataframe contain the ground-truth data.
            track_status: the prediction status of different obstacles.
            video_dir: prefix to the directory where the output videos
                will be saved.
            resolution: resolution of image context.
            offset: offset added to the whole local physical coordinates.
            apply_traj_truncation: whether to do post processing of predicted
                trajectories.
            filename_info: the information to be displayed in the file name.
            exp_important_obs: the expand important obstacles.

        Returns:
            a dummy True for tqdm to work.
        """
        frame_length = len(event_road)
        h, w, _ = event_road[0].shape

        plot_context = PlotImage(resolution, grayscale=False)
        plot_ego_bbox = PlotObjBBox(bowtie=True)
        plot_obs_bbox = PlotObjBBox(bowtie=False)
        plot_future_pred = PlotTraj()
        plot_future_gt = PlotGT()
        plot_behav = PlotBehav(position=[w, h])
        plot_expand_obs = PlotExpandObs()
        plotting_helpers = (
            plot_context,
            plot_ego_bbox,
            plot_obs_bbox,
            plot_future_pred,
            plot_future_gt,
            plot_behav,
            plot_expand_obs,
        )

        # Get coordinate range with the help of roadnet image size
        ret = trans_coord_img_to_phy(
            np.array([0, h, 0, h]),
            np.array([0, 0, w, w]),
            self.height,
            self.width,
            self.resolution,
        )
        ((x0, x1, x2, x3), (y0, y1, y2, y3)) = ret
        min_x = min(x0, x1, x2, x3) + offset[0]
        min_y = min(y0, y1, y2, y3) + offset[1]
        max_x = max(x0, x1, x2, x3) + offset[0]
        max_y = max(y0, y1, y2, y3) + offset[1]

        # Set trajectory truncation helpers.
        if apply_traj_truncation:
            trunc_traj_lim = TruncateTrajByLim(
                min_x + self.__epsilon,
                min_y + self.__epsilon,
                max_x - self.__epsilon,
                max_y - self.__epsilon,
            )
            trunc_traj_bounce = TruncateTrajBounce()
            truncation_helpers = [trunc_traj_lim, trunc_traj_bounce]
        else:
            truncation_helpers = []

        # Save results.
        image_dir = os.path.join(video_dir, "img")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Perform animation on unique frame in ground-truth dataframe.
        with ProcessPoolExecutor(max_workers=self.__num_workers) as pool:
            with tqdm(
                desc="Rendering", total=frame_length, unit="events"
            ) as progress_bar:
                result_futures = []
                for idx in range(frame_length):
                    args = [
                        dataset_index[idx],
                        seq_index[idx],
                        event_road[idx],
                        pred_df[idx],
                        gt_df[idx],
                        track_status[idx],
                        plotting_helpers,
                        truncation_helpers,
                        offset,
                        image_dir,
                    ]
                    if (
                        exp_important_obs is not None
                        and len(exp_important_obs) != 0
                    ):
                        args.append(exp_important_obs[idx])
                    else:
                        args.append(None)
                    result = pool.submit(self.plot_image, *args)
                    result_futures.append(result)

                # Check progress.
                for _ in as_completed(result_futures):
                    progress_bar.update(n=1)

        # Render the video based on the generated images.
        size = (1500, 1200)
        if filename_info is None:
            video_path = f"{video_dir}/event_test.mp4"
        else:
            video_path = f"{video_dir}/{filename_info}.mp4"
        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self.fps,
            size,
        )
        filelist = sorted(os.listdir(image_dir))
        for item in filelist:
            if item.endswith(".png"):
                item = os.path.join(image_dir, item)
                img = cv2.imread(item).astype("uint8")
                video.write(img)
        video.release()
        cv2.destroyAllWindows()
        # if self.del_img_dir:
        #     shutil.rmtree(image_dir)

        return True
