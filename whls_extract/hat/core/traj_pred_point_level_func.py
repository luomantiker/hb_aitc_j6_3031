# Copyright (c) Horizon Robotics. All rights reserved.
# This file defines point-level post processing functions.

from typing import Optional, Tuple, Union

import numpy as np


class TruncateTrajBounce:
    """Truncate the bouncing trajectories.

    This class builds a callable to truncate trajectories which bounce off
    against some boundaries. The method is to check the heading difference
    between two adjacent trajectory segments. If the different is too big, we
    know there's a bouncing event happening. Then the trajectory after that
    will be removed.
    """

    def __init__(
        self, threshold: float = np.pi / 10, flag_out: Optional[bool] = False
    ):
        """Initialize method.

        Args:
            threshold: heading change threshold for bouncing
                detection, The recommended value is the maximum absolute yaw
                rate for the scenario. Defaults to pi / 10.
            flag_out: whether to output the result of flag.
                Default to False.
        """
        self.__th = threshold
        self.flag_out = flag_out

    def __call__(
        self, traj: np.ndarray
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        """Truncate trajectory to remove bouncing segments.

        This method detect bouncing in trajectories by calculating the heading
        angle change among trajectory segments and detect changes that is
        larger than a pre-defined threshold.

        Args:
            traj: [2, N], the trajectory to truncate.

        Returns:
            traj_out: [2, M], truncated trajectory.
            flag: [2, N], flag used to judge whether the change of
                yaw angle exceeds the threshold.
        """
        # Bouncing would not happend for trajectory shorter than 2 frames, so
        # we can safely return the trajectory passed in.
        if traj.shape[1] <= 2:
            if self.flag_out:
                return traj, np.array([True])
            else:
                return traj
        # Coordinate temporal difference.
        pos_diff = np.diff(traj, axis=-1)
        # Calculate heading angle diff. Unwrap is needed due to the
        # discontinuity in -pi.
        yaw = np.arctan2(pos_diff[0, :], pos_diff[1, :])
        yaw_diff = np.diff(np.unwrap(yaw))
        # Detect bounce by thresholding absolute yaw changes.
        flag = np.abs(yaw_diff) < self.__th
        # Calculate the valid head before first bounce.
        flag = np.cumprod(flag) > 0
        # Assume first two points is always valid.
        flag = np.concatenate([[True, True], flag])
        # Filter out trajectory.
        traj_out = traj[:, flag]

        if self.flag_out:
            return traj_out, flag
        else:
            return traj_out


class TruncateTrajByLim:
    """Truncate trajectory by coordinate limits.

    This class builds a callable to truncate trajectories w.r.t user-provided
    coordinate limits. Only the head of trajectory within the limits are
    preserved.
    """

    def __init__(
        self,
        x_min: Optional[float] = None,
        y_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_max: Optional[float] = None,
    ):
        """Initialize method.

        Args:
            x_min: lower limit for x coordinates.
            y_min: lower limit for y coordinates.
            x_max: upper limit for x coordinates.
            y_max: upper limit for y coordinates.
        """
        self.__x_min = x_min if x_min is not None else -np.inf
        self.__y_min = y_min if y_min is not None else -np.inf
        self.__x_max = x_max if x_max is not None else np.inf
        self.__y_max = y_max if y_max is not None else np.inf

    def __call__(self, traj) -> np.ndarray:
        """Truncate a trajectory.

        Truncate the trajectory with limits. Only the head part with is within
        limits is preserved.

        Args:
            traj: [2, N], the trajectory to truncate.

        Returns:
            traj_out: [2, M], truncated trajector.
        """
        # Flag for mating limits.
        lower_x = traj[0, :] >= self.__x_min
        upper_x = traj[0, :] <= self.__x_max
        lower_y = traj[1, :] >= self.__y_min
        upper_y = traj[1, :] <= self.__y_max
        # Flag for head part matching limits.
        flag = np.cumprod(lower_x & upper_x & lower_y & upper_y)
        # Length of the filtered trajectory.
        valid_idx = np.nonzero(flag)[0]
        len_traj_filtered = valid_idx[-1] + 1 if len(valid_idx) > 0 else 0
        # Filter.
        traj_out = traj[:, :len_traj_filtered]
        return traj_out


class TruncateTrajByVar:
    """Truncate trajectories with large variance.

    This method truncate trajectory segments that have large variance. This is
    equivalent to truncating trajectories with less prediction confidence.
    """

    def __init__(self, threshold: Optional[float] = 1e-2):
        """Initialize method.

        Args:
            threshold: threshold for truncating trajectory.
        """
        self.__th = threshold

    def __call__(self, traj: np.ndarray, traj_std: np.ndarray) -> np.ndarray:
        """Truncate trajectory to remove segments with large variance.

        This method inspects the prediction variance of each trajectory point.
        If the variance is larger than a predefined threshold, then the
        trajectory after that point is truncated.

        Args:
            traj: [2, N], the trajectory to truncate.
            traj_std: [2, N], the trajectory standard deviation
                in the x and y direction.

        Returns:
            traj_out: [2, M], truncated trajectory.
        """
        # Check if any of the var_x or var_y exceeds the threshold.
        flag = np.maximum(np.square(traj_std), axis=0) < self.__th
        # Calculate the valid flag.
        flag = np.cumprod(flag) > 0
        # Filter trajectory but keep at least two points.
        flag[:2] = True
        traj_out = traj[:, flag]
        return traj_out


class TruncateStationary:
    """Truncate the stationary part of vehicle trajectory."""

    def __init__(self, threshold: Optional[float] = 1e-2):
        """Initialize method.

        Args:
            threshold: the threshold to determine whether
                the vehicle is stationary. Defaults to 1e-2 (meters).
        """
        self.__threshold = threshold

    def __call__(self, traj: np.ndarray) -> np.ndarray:
        """Truncate the stationary part of vehicle trajectory.

        Args:
            traj: [2, N], the trajectory to truncate.

        Returns:
            traj_out: [2, M], truncated trajectory.
        """
        if traj.shape[1] < 2:
            return traj
        # Coordinate temporal difference.
        pos_diff = np.diff(traj, axis=-1)
        pos_diff = np.sqrt(np.sum(pos_diff ** 2, axis=0))
        # Detect stationary part by threshold. We truncate at the first found
        # stationary point.
        flag = np.cumprod(pos_diff > self.__threshold) > 0
        flag = np.concatenate([[True], flag])
        traj_out = traj[:, flag]
        return traj_out


class ExtrapolateTraj:
    """Extrapolate the vehicle trajectory to a certain length."""

    EXTRAPOLATE_METHOD_KEYS = ["CV"]

    def __init__(self, max_seq_len: int, method: Optional[str] = "CV"):
        """Initialize method.

        Args:
            max_seq_len: the maximum length of trajectory sequence.
            method: the selected trajectory extrapolating
                method. defaults to CV (constant velocity).
        """
        self.__max_seq_len = max_seq_len
        assert (
            method in self.EXTRAPOLATE_METHOD_KEYS
        ), f"Request an unknown extrapolating method {method}"
        self.__method = method

    def __call__(self, traj: np.ndarray) -> np.ndarray:
        """Extrapolate the vehicle trajectory to a certain length.

        Args:
            traj: [2, N], the trajectory to truncate.

        Returns:
            traj_out: [2, max_seq_len], extrapolated trajectory.
        """
        num_coors, traj_len = traj.shape
        if traj_len < 1:  # empty traj
            return traj
        elif traj_len < 2:  # stationary obs
            return np.full([num_coors, self.__max_seq_len], traj)
        elif traj_len >= self.__max_seq_len:  # too long traj
            return traj[:, : self.__max_seq_len]

        if self.__method == "CV":
            # Constant velocity: extrapolate based on the the last valid diff.
            last_diff = np.diff(traj[:, (traj_len - 2) : traj_len])
            last_coor = traj[:, (traj_len - 1) : traj_len]
            extrapolate_len = self.__max_seq_len - traj_len
            extrapolated_diff = last_diff * np.arange(1, extrapolate_len + 1)
            extrapolated_traj = last_coor + extrapolated_diff
            traj_out = np.concatenate((traj, extrapolated_traj), axis=1)
            return traj_out
