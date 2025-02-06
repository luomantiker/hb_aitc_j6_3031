import pickle
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from skimage import color

from hat.core.traj_pred_typing import VehicleSafeArea
from hat.core.traj_pred_utils import Affine2D
from hat.core.traj_pred_utils import TdtCoordHelper as TCH


def trans_coord_img_to_phy(
    arr_x: np.ndarray,
    arr_y: np.ndarray,
    height: np.ndarray,
    width: np.ndarray,
    resolution: np.ndarray,
    center: Optional[Tuple[int]] = (0, 0),
) -> Tuple[np.ndarray]:
    """Transform coordinates from image to physical coordinates.

    Args:
        arr_x: [2, N], x coordinates array.
        arr_y: [2, N], y coordinates array.
        height: height of img
        width: width of img
        center: physical coordinates of local sequence's
            center. Consist of [x, y].
    Returns:
        arr_x: [2, N], the physical x coordinates.
        arr_y: [2, N], the physical y coordinates.
    """
    arr_x, arr_y = Affine2D.coord_translate(
        arr_x, arr_y, height / 2, width / 2
    )
    arr_x, arr_y = Affine2D.coord_scale(
        arr_x, arr_y, 1 / resolution, 1 / resolution
    )
    arr_x, arr_y = Affine2D.coord_translate(
        arr_x, arr_y, -center[0], -center[1]
    )
    return arr_x, arr_y


class PlotBinaryRoadnet:
    """Plot binary roadnet on axis.

    This class draw a binary roadnet image onto MatplotLib axis. The class
    instances are callable.
    """

    def __init__(
        self,
        height: int,
        width: int,
        resolution: float,
        marker: Optional[float] = ".",
        color: Optional[str] = "red",
        markersize: Optional[float] = 0.2,
        alpha: Optional[int] = 0.4,
    ):
        """Initialize method.

        Args:
            height: the map height.
            width: the map width.
            resolution: the map resolution.
            marker: marker type. Defaults to ".".
            color: marker color. Defaults to "red".
            markersize: marker size. Defaults to 0.2.
            alpha: marker opacity. Defaults to 0.4.
        """
        self.height = height
        self.width = width
        self.resolution = resolution
        self.__marker = marker
        self.__color = color
        self.__markersize = markersize
        self.__alpha = alpha

    def __call__(
        self, ax: "matplotlib.axis", roadnet: np.ndarray  # noqa: F821
    ):
        """Draw binary roadnet on the axis specified.

        This method plot the binary roadnet on the axis pass in by the caller.
        The roadnet is first re-binarized to get the coordinates of non-zero
        elements. Then, these elements are plot onto the axis with proper
        coordinate transformation provided by `dataframe_render`

        Args:
            ax: the axis to plot on.
            roadnet: the roadnet to plot.
        """
        roadnet_h, roadnet_w = np.nonzero(roadnet > 1e-3)
        phy_x, phy_y = trans_coord_img_to_phy(
            roadnet_h, roadnet_w, self.height, self.width, self.resolution
        )
        ax.plot(
            phy_x,
            phy_y,
            self.__marker,
            color=self.__color,
            markersize=self.__markersize,
            alpha=self.__alpha,
        )


class PlotImage:
    """Plot image on axis.

    This class draw an image onto MatplotLib axis. The class instances are
    callable.
    """

    def __init__(self, resolution: float, grayscale: Optional[bool] = True):
        """Initialize method.

        Args:
            resolution: resolution of image.
            grayscale: whether to convert grayscale image.
                Defaults to True.
        """
        self.resolution = resolution
        self.grayscale = grayscale
        self.alpha = 0.2

    def __call__(self, ax: "matplotlib.axis", image: np.ndarray):  # noqa: F821
        """Draw image on the axis specified.

        This method shows an image on Matplotlib axis.

        Args:
            ax : the axis to plot on.
            image: image to plot.
        """
        # Resize.
        h, w, _ = image.shape
        image = np.swapaxes(image, 0, 1)
        image.astype(np.uint8)
        if self.grayscale:
            image = color.rgb2gray(image)
        ax.set(xlim=(0, h), ylim=(0, w))
        ax.grid(False)
        ax.imshow(image, alpha=self.alpha)


class PlotObjBBox:
    """Plot object bounding box.

    This class plots vehicle bounding boxes onto a user specified
    matplotlib axis. The caller has the choice to choose between regular
    rectangular box or the bow-tie shaped box.
    """

    def __init__(
        self,
        linewidth: Optional[float] = 0.6,
        bowtie: Optional[bool] = True,
        show_id: Optional[bool] = True,
        fontsize: Optional[float] = 4,
    ):
        """Initialize method.

        Args:
            linewidth: bbox line width. Defaults to 0.6.
            bowtie: whether to draw bowtie or regular shaped
                rectangular box. Defaults to True.
            show_id: whether to show object tracking id test.
                Defaults to True.
            fontsize: the font size of the track id.
        """
        self.__linewidth = linewidth
        self.__bowtie = bowtie
        self.__show_id = show_id
        self.__fontsize = fontsize

    def __call__(
        self,
        ax: "matplotlib.axis1",  # noqa: F821
        pos_x: float,
        pos_y: float,
        yaw: float,
        length: float,
        width: float,
        color: Tuple[float],
        track_id: int = None,
        track_status: int = None,
    ):
        """Plot object bounding box on axis.

        Args:
            ax: the axis to plot on.
            pos_x: x position of vehicle center.
            pos_y: y position of vehicle center.
            yaw: heading angle of vehicle (in radians).
            length: length of vehicle.
            width: width of vehicle.
            color: color of vehicle bbox.
            track_id: tracking id, defaults to None.
            track_status: the prediction status of the
                obstacle. Defaults to None.
        """
        (x0, x1, x2, x3), (y0, y1, y2, y3) = TCH.get_bbox_corners(
            pos_x, pos_y, yaw, length, width
        )
        if self.__bowtie:
            box_x = [x0, x1, x3, x2, x0]
            box_y = [y0, y1, y3, y2, y0]
        else:
            box_x = [x0, x1, x2, x3, x0]
            box_y = [y0, y1, y2, y3, y0]
        ax.plot(box_x, box_y, "-", lw=self.__linewidth, color=color)

        if self.__show_id:
            assert track_id is not None
            # Use annotate to account for axis limits, text would ignore
            # limits.
            name_str = f"{track_id}_{track_status}"
            ax.annotate(
                name_str, (x0, y0), color=color, fontsize=self.__fontsize
            )


class PlotBehav:
    """Plot behaviors.

    This class plot behaviors on the user specified axis with user specified
    appearance.
    """

    def __init__(
        self,
        position: Optional[Tuple[float]] = (520, 512),
        fontsize: Optional[int] = 4,
    ):
        """Initialize method.

        Args:
            position: text position. Defaults
                to [512, 512].
            fontsize: the font size of the anchor indices.
        """
        self.__position = position
        self.__fontsize = fontsize
        self.__cats = {
            -1: "filter-label",
            0: "keep-lane",
            1: "left-change",
            2: "right-change",
        }

    def __call__(
        self, ax: "matplotlib.axis1", behavs_dict: Dict  # noqa: F821
    ):
        """Plot object bounding box on axis.

        Args:
            ax: the axis to plot on.
            behavs_dict: ground truth and prediction behavior for
                each obstacle.
        """
        obs_index = 0
        for track_id in behavs_dict.keys():
            (
                gt_lat_behav,
                gt_lon_behav,
                lat_behav,
                lon_behav,
            ) = behavs_dict[track_id]
            if gt_lat_behav < 0:
                gt_lat_behav = -1
            ax.text(
                self.__position[0] + 5,
                self.__position[1] - 40 * obs_index,
                f"obstacle id: {track_id}\n"
                f" - pred: {self.__cats[lat_behav]}\n"
                f" - label: {self.__cats[gt_lat_behav]}\n"
                "-------------------------",
                weight="light",
                ma="left",
                va="top",
                fontsize=self.__fontsize,
            )
            obs_index += 1


class PlotTraj:
    """Plot trajectories.

    This class plot trajectories on the user specified axis with user specified
    appearance.
    """

    def __init__(
        self,
        marker: Optional[str] = "-",
        linewidth: Optional[float] = 0.6,
        fontsize: Optional[int] = 4,
    ):
        """Initialize method.

        Args:
            marker: marker type. Defaults to '-'.
            linewidth: trajectory linewidth. Defaults to 0.6.
            fontsize: the font size of the anchor indices.
        """
        self.__marker = marker
        self.__linewidth = linewidth
        self.__fontsize = fontsize

    def __call__(
        self,
        ax: "matplotlib.axis",  # noqa: F821
        traj: np.ndarray,
        color: Tuple[float],
        fade: float,
        anc_idx: Optional[int] = None,
    ):
        """Plot trajectory on axis.

        Args:
            ax (matplotlib.axis): the axis to plot on.
            traj (numpy.array, [2, N]): trajectory points.
            color (Tuple[float]): color of trajectory marker.
            fade: transparency value to plot on
            anc_idx: the index of the used anchor. It will be plot near
                the last point of the trajectory. If this value is None, no
                text will be plotted.
        """
        ax.plot(
            traj[0, :],
            traj[1, :],
            self.__marker,
            lw=self.__linewidth,
            alpha=fade,
            color=color,
        )
        if anc_idx is not None:
            ax.annotate(
                str(int(anc_idx)),
                (traj[0, -1], traj[1, -1]),
                color=color,
                fontsize=self.__fontsize,
            )


class PlotPastPred:
    """Plot past predictions.

    This class plot past predictions on the user specified axis with user
    specified appearance. When the flag `fade` is True, the more recent
    predictions takes on higher alpha values.
    """

    def __init__(
        self,
        marker: Optional[str] = ".",
        markersize: int = 0.3,
        fade: Optional[bool] = True,
    ):
        """Initialize method.

        Args:
            marker: marker type. Defaults to ".".
            markersize: marker size. Defaults to 0.3.
            fade: whether the trajectory fades with history.
                Defaults to True.
        """
        self.__marker = marker
        assert (
            "-" not in self.__marker
        ), "Lines or dashes won't make sense for past predictions"
        self.__markersize = markersize
        self.__fade = fade

    def __call__(
        self,
        ax: "matplotlib.axis",  # noqa: F821
        pred_dict: Dict,
        color: Tuple[float],
    ):
        """Plot past predictions on axis.

        Args:
            ax: the axis to plot on.
            pred_dict: mapping from prediction horizon (int) to list of
                predicted coordinates, each is a numpy.array with shape [2].
            color: color of trajectory marker.
        """
        assert isinstance(pred_dict, dict), "Please pass in a dictionary."
        horizons = list(pred_dict.keys())
        if len(horizons) == 0:
            return
        horizons.sort()
        n = max(horizons)
        for i in horizons:
            fade_alpha = (n - i + 1) / (n + 1) if self.__fade else 1.0
            for x, y in pred_dict[i]:
                ax.plot(
                    x,
                    y,
                    self.__marker,
                    markersize=self.__markersize,
                    alpha=fade_alpha,
                    color=color,
                )


class PlotGT:
    """Plot past predictions.

    This class plot past predictions on the user specified axis with user
    specified appearance. When the flag `fade` is True, the more recent
    predictions takes on higher alpha values.
    """

    def __init__(
        self, marker: Optional[str] = ".", markersize: Optional[float] = 0.3
    ):
        """Initialize method.

        Args:
            marker: marker type. Defaults to ".".
            markersize: marker size. Defaults to 0.3.
        """
        self.__marker = marker
        assert (
            "-" not in self.__marker
        ), "Lines or dashes won't make sense for past predictions"
        self.__markersize = markersize

    def __call__(
        self,
        ax: "matplotlib.axis",  # noqa: F821
        gt_pos: Dict,
        color: Tuple[float],
    ):
        """Plot past predictions on axis.

        Args:
            ax: the axis to plot on.
            pred_dict: mapping from prediction horizon (int) to list of
                predicted coordinates, each is a numpy.array with shape [2].
            color: color of trajectory marker.
        """
        num = gt_pos.shape[0]
        for i in range(num):
            ax.plot(
                gt_pos[i][0],
                gt_pos[i][1],
                self.__marker,
                markersize=self.__markersize,
                color=color,
            )


class PlotExpandObs:
    """Plot expanded important obstacles."""

    def __init__(
        self, marker: Optional[str] = "^", markersize: Optional[float] = 1
    ):
        """Initialize method.

        Args:
            marker: marker type.
            markersize: marker size.
        """
        self.__marker = marker
        assert (
            "-" not in self.__marker
        ), "Lines or dashes won't make sense for past predictions"
        self.__markersize = markersize

    def __call__(
        self,
        ax: "matplotlib.axis",  # noqa: F821
        safe_area: Callable,
        color: Tuple,
        offset: Tuple[int] = (0, 0),
    ):
        """Plot past predictions on axis.

        Args:
            ax: the axis to plot on.
            safe_area: the safe area instance.
            color: color of trajectory marker.
            offset: the coordinate offset of the VCS center in the image.
        """
        if isinstance(safe_area, VehicleSafeArea):
            traj = safe_area.img_traj
            ax.plot(
                traj[:, 0] + offset[0],
                traj[:, 1] + offset[1],
                self.__marker,
                markersize=self.__markersize,
                color=color,
            )


def load_anchors(
    anchor_cfg: Dict, type: Optional[str] = "tensor"
) -> torch.Tensor:
    """Load anchor trajectories from pickled files.

    Args:
        anchor_cfg: the anchor configuration dictionary with
            the following keys:
            1. 'anchor_file': the file path of the pickle anchors.
            2. 'anchor_method': the method to obtain the anchors, it
                should be one of [kmeans, uniform].
            3. 'anchor_num': the number of anchors.
        type: the output type of the anchors.

    Return:
        anchors: the anchor trajectories, shape is [num_anchors, traj_len, 2]
    """
    if anchor_cfg is None:
        return None
    elif "anchor_file" in anchor_cfg:
        required_keys = ["anchor_file", "anchor_method", "anchor_num"]
        for key in required_keys:
            assert (
                key in anchor_cfg
            ), f"The dict anchor_cfg do not have key {key}."
        method = anchor_cfg["anchor_method"]
        with open(anchor_cfg["anchor_file"], "rb") as f:
            anchors = pickle.load(f)
            if method in ["kmeans", "uniform"]:
                anchors = anchors[anchor_cfg["anchor_num"]]
            else:
                raise ValueError(
                    f"Undefined anchor method {method}. It should in"
                    "[kmeans, uniform]."
                )
        if type == "tensor":
            anchors = torch.Tensor(anchors)
        return anchors
    else:
        required_keys = ["anchor_key", "anchor_num", "traj_len"]
        for key in required_keys:
            assert (
                key in anchor_cfg
            ), f"The dict anchor_cfg do not have key {key}."
        ret = {
            "anchor_key": anchor_cfg["anchor_key"],
            "num_anchors": anchor_cfg["anchor_num"],
            "traj_len": anchor_cfg["traj_len"],
        }
        return ret
