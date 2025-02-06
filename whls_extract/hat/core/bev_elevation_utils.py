from decimal import Decimal
from typing import List, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np

HOMO_PAD_VALUE = 65535


def decimal_div(a: Union[int, float], b: Union[int, float]):
    """Calculate a/b with decimal module to avoid loss of accuracy."""
    res = Decimal(str(a)) / Decimal(str(b))
    return float(res)


def decimal_minus(a: Union[int, float], b: Union[int, float]):
    """Calculate a-b with decimal module to avoid loss of accuracy."""
    res = Decimal(str(a)) - Decimal(str(b))
    return float(res)


def compute_vismask(
    ori_vismask: object,
    visible_flag: List[int],
    occlusion_flag: List[int],
    use_3d_mask: bool = False,
) -> object:
    """Compute original vismask into binary map.

    Args:
        ori_vismask: In bev coordinate,
            input original vismask with shape(h, w, c).
        visible_flag: Flag bit of the visible area, default [1, 2].
        occlusion_flag: Flag bit of the occlusion area, default [0, 1, 2, 3].
        use_3d_mask: default False, remain the same as above.
            if True:
                visible_flag: modifly 3d parsing area flag, default [0].
                occlusion_flag: modifly 3d occlusion area flag, default [1].
                Returns (h, w) the 3d area is valued 1, 3d occlusion area is 2.

    Returns: (h, w) the visible area is valued 1, else 0.
    """
    if ori_vismask is None:
        return None
    ori_vismask = np.asarray(ori_vismask)
    assert len(ori_vismask.shape) in [2, 3], (
        "The dimensions of the vismask should be two or three"
        f"not {len(ori_vismask.shape)}"
    )
    if len(ori_vismask.shape) == 3:  # mask created by teng.chen
        # ori_vismask is visible area mask, which has 3 channels.
        vis_mask = np.zeros(ori_vismask.shape[:2], np.uint8)
        # Channel0 stands for visible area, in which, the value 1
        # is visible area filtering the opposite lane, the value 2
        # is visible area not filtering the opposite lane.
        for v_flag in visible_flag:
            vis_mask[ori_vismask[:, :, 0] == v_flag] = 1
        # Channel2 stands for occlusion area, in which, the value 0
        # is vehicle,pedestration, cyclist occupancy. the value 1
        # is vehicle,pedestration,cyclist occlusion. the value 2 is
        # tree occlusion. the value 3 is solid barrier.
        for o_flag in occlusion_flag:
            vis_mask[ori_vismask[:, :, 2] == o_flag] = 0
        if use_3d_mask:
            vis_mask = np.zeros(ori_vismask.shape[:2], np.uint8)
            for v_flag in visible_flag:
                vis_mask[ori_vismask[:, :, 2] == v_flag] = 1
            for o_flag in occlusion_flag:
                vis_mask[ori_vismask[:, :, 2] == o_flag] = 2
    elif len(ori_vismask.shape) == 2:
        vis_mask = ori_vismask
    return vis_mask


def get_vcsrange_bbox_data(
    raw_data: np.ndarray,
    vcs_range: List,
    raw_vcs_range: List,
    pad_index: int,
) -> Union[np.ndarray, Tuple]:
    """Get min vcs_range contain vcs_range and raw_vcs_range at same vcs origin.

    raw_data is saved raw data correspond to raw_vcs_range.
    vcs_range and raw_vcs_range with same vcs orgin, but can
    have different intersections.

    Args:
        data : (h, w) or (h, w, 3)
        vcs_range : values in (bottom, right, top, left) order.
        raw_vcs_range : values in (bottom, right, top, left) order.
        pad_index: padded int label to raw_data.

    Returns:
        _type_: _description_
    """
    h, w = raw_data.shape[:2]
    vcsrange_bbox = (
        min(vcs_range[0], raw_vcs_range[0]),
        min(vcs_range[1], raw_vcs_range[1]),
        max(vcs_range[2], raw_vcs_range[2]),
        max(vcs_range[3], raw_vcs_range[3]),
    )
    raw_resolution = (
        decimal_div(abs(decimal_minus(raw_vcs_range[3], raw_vcs_range[1])), w),
        decimal_div(abs(decimal_minus(raw_vcs_range[2], raw_vcs_range[0])), h),
    )  # (y, x)
    pad_top = int(
        decimal_div(
            decimal_minus(vcsrange_bbox[2], raw_vcs_range[2]),
            raw_resolution[1],
        )
    )
    pad_bottom = int(
        decimal_div(
            decimal_minus(raw_vcs_range[0], vcsrange_bbox[0]),
            raw_resolution[1],
        )
    )
    pad_left = int(
        decimal_div(
            decimal_minus(vcsrange_bbox[3], raw_vcs_range[3]),
            raw_resolution[0],
        )
    )
    pad_right = int(
        decimal_div(
            decimal_minus(raw_vcs_range[1], vcsrange_bbox[1]),
            raw_resolution[0],
        )
    )

    out_h = h + pad_top + pad_bottom
    out_w = w + pad_left + pad_right
    out_shape = list(raw_data.shape)
    out_shape[:2] = [out_h, out_w]
    out = np.ones(out_shape, raw_data.dtype) * pad_index
    out[pad_top : pad_top + h, pad_left : pad_left + w, ...] = raw_data

    return out, vcsrange_bbox


def get_roi_resize_data(
    data: np.ndarray,
    vcs_range: Tuple,
    roi_vcs_range: Tuple,
    target_size: Tuple,
    pad_index: int,
) -> np.ndarray:
    """Get roi vcs range data from the original vcs range data.

    Args:
        data: (h, w) or (h, w, 3), input origin array in bev coordinate.
        vcs_range: (bottom, right, top, left)m,
            corresponding data vcs range.
        roi_vcs_range: (bottom, right, top, left)m,
            region of interest vcs range.
        target_size: (h_roi, w_roi) or (h_roi, w_roi, 3),
            roi_vcs_range corresponding bev size
        pad_index: padded int label to raw_data.

    Returns: the target_size data in bev coordinate.
    """
    union_data, union_vcs_range = get_vcsrange_bbox_data(
        data,
        roi_vcs_range,
        vcs_range,
        pad_index,
    )
    roi_data = crop_roi_vcs_range(
        union_data,
        roi_vcs_range,
        union_vcs_range,
        return_crop_coord=False,
    )
    if target_size is not None:
        h, w = roi_data.shape[:2]
        if (h, w) != target_size:
            h, w = target_size
            roi_data = cv2.resize(
                roi_data, (w, h), interpolation=cv2.INTER_NEAREST
            )
    return roi_data


def calculate_points_on_line(
    start_point: List,
    end_point: List,
) -> np.ndarray:
    """
    Calculate a list of points that lie on a straight line.

    Args:
        start_point: The coordinates of the start point (x1, y1).
        end_point: The coordinates of the end point (x2, y2).

    Returns:
        An array of integer points that lie on the line.
    """
    x1, y1 = start_point
    x2, y2 = end_point
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    num_points = int(max(dx, dy) + 1)

    x = np.linspace(x1, x2, num=num_points)
    y = np.linspace(y1, y2, num=num_points)
    points = np.column_stack((x, y)).astype(int)
    return points


def reformat_homo_info(
    homo_info: Mapping, camera_view_names: List, pad_val: int = 65535
) -> np.ndarray:
    """Reorder homo info as camera_view_names, and pad fake homography.

       and homo_offset.

        This function used for the fake homo info (homo_offset and homo_mat).
        The data in meta_info indicate the sensor data of the dataset,
        while the camera_view_names indicate the training multi-view.

        For example, in 11v pipeline:
            the camera_view_names is :
                [camera_front, camera_front_left, ...., fisheye_front, ...,camera_front_30fov] # noqa
            while thr meta info is the 6V sensor data:
                [camera_front, camera_front_left,...., camera_rear]
            therefore, we should pad the miss 5v (fisheye_front, ...,camera_front_30fov) data. # noqa

    Args:
        homo_info: dict contains each view's homography or
            homooffset.
        camera_view_names: camera view names.
        pad_val: pad value of homooffset or homography.
    """
    homo_info_size = list(homo_info.values())[0].shape
    if len(homo_info_size) == 2:  # homography
        fake_homo_info = np.eye(3, dtype="float32")
        fake_homo_info[0, 2] = pad_val
        fake_homo_info[1, 2] = pad_val
    else:  # homooffset
        fake_homo_info = np.ones(homo_info_size, dtype="float32") * pad_val
    # pad fake homography or homo_offset
    if len(homo_info) != len(camera_view_names):
        for camera_name in camera_view_names:
            homo_info.setdefault(camera_name, fake_homo_info)
    # reorder dict as camera_view_names
    ret_homo_info = np.concatenate(
        [homo_info[k] for k in camera_view_names], axis=0
    ).astype("float32")

    return ret_homo_info


def reformat_meta_info(
    meta_info: Mapping,
    camera_view_names: list,
    num_plane: int = 1,
) -> dict:
    """Reorder meta info as camera_view_names, and pad fake meta info.

    This function used for the fake info.
    The data in meta_info indicate the sensor data of the dataset,
    while the camera_view_names indicate the training multi-view.

    For example, in 11v pipeline:
        the camera_view_names is :
            [camera_front, camera_front_left, ...., fisheye_front, camera_front_30fov] # noqa
        while thr meta info is the 6V sensor data:
            [camera_front, camera_front_left,...., camera_rear]
        therefore, we should pad the miss 5v data.

    Args:
        meta_info: dict contains each view's calibration param.
        camera_view_names: camera view names.
        num_plane: num of vcs planes.
    """
    assert "intrinsics" in meta_info
    meta_camera_nums = len(meta_info["intrinsics"])

    # The fake data is used to pad the data not contains in dict
    fake_homo_flag = np.zeros((len(camera_view_names) * num_plane,))
    fake_dict = {
        "T_vcs2cam": np.eye(4)[np.newaxis, ...].astype(np.float64),
        "intrinsics": np.eye(3)[np.newaxis, ...].astype(np.float64),
        "distort_coeffs": np.zeros((8,), dtype=np.float64),
        "transformats": np.eye(3)[np.newaxis, ...].astype(np.float64),
        # NOTE: for `ipm_img_sizes`, in fake data pad mode, e.g.
        # 11v training  using 7v data, we should pad 4v fisheye data,
        # while in image warp situation, the  ipm_img_sizes x 4 as the
        # fake input feed to the backbone, therefore the ipm_img_sizes
        # should be divide by 64.
        "ipm_img_sizes": np.array([128, 128], dtype=np.float64),
        "img_shape": np.ones((1, 2), dtype=np.float64),
        "cam2local_rot": np.zeros(3, dtype=np.float64)[np.newaxis, ...],
        "cam2local_translation": np.zeros(3, dtype=np.float64)[
            np.newaxis, ...
        ],
        "T_local2vcs": np.eye(4, dtype=np.float64)[np.newaxis, ...],
    }

    # used for pad the homo_offset calculate on gpu
    if meta_camera_nums != len(camera_view_names):
        for meta_name, meta_value in meta_info.items():
            if meta_name in fake_dict:
                fake_mate = fake_dict[meta_name]
                for cam_idx, camera_name in enumerate(camera_view_names):
                    if camera_name not in meta_value:
                        meta_value[camera_name] = fake_mate
                        fake_homo_flag[
                            cam_idx * num_plane : (cam_idx + 1) * num_plane
                        ] = 1

    # reorder dict as camera_view_names
    for meta_name, meta_value in meta_info.items():
        if meta_name in fake_dict:
            meta_info[meta_name] = [
                meta_value[cam] for cam in camera_view_names
            ]
    meta_info["fake_homo_flag"] = fake_homo_flag
    # `aug_flag` is default set to False, if do augmentation e.g.
    # RPYAug, aug_flag will be set to True, and in GeneratBEVOffset
    # will re-calculate the homo_offset.
    meta_info["aug_flag"] = False

    if "homo_mat" in meta_info:
        homo_mat = meta_info.pop("homo_mat")
        meta_info["homography"] = reformat_homo_info(
            homo_mat, camera_view_names, HOMO_PAD_VALUE
        )
    if "homo_offset" in meta_info:
        meta_info["homo_offset"] = reformat_homo_info(
            meta_info["homo_offset"], camera_view_names, HOMO_PAD_VALUE
        )

    return meta_info


def get_transform_ipm2vcs(spatial_resolution: Sequence, vcs_range: Sequence):
    """
    Calculate transform matrix from ipm to vcs ground.

    Args:
        spatial_resolution: bev spatial resolution.(unit is meters)
        vcs_range: visbile range of bev, (bottom, right, top, left)
                in order.
    """

    ipm_height = int(abs(vcs_range[2] - vcs_range[0]) / spatial_resolution[0])
    ipm_width = int(abs(vcs_range[3] - vcs_range[1]) / spatial_resolution[1])
    ipm_bottom = ipm_height - 1
    ipm_right = ipm_width - 1
    ipm_top, ipm_left = 0, 0

    ipm_region = [
        [ipm_right, ipm_bottom],
        [ipm_right, ipm_top],
        [ipm_left, ipm_top],
        [ipm_left, ipm_bottom],
    ]

    vcs_bottom = vcs_range[0] + spatial_resolution[0] / 2
    vcs_right = vcs_range[1] + spatial_resolution[1] / 2
    vcs_top = vcs_range[2] - spatial_resolution[0] / 2
    vcs_left = vcs_range[3] - spatial_resolution[1] / 2

    vcs_region = [
        [vcs_bottom, vcs_right],
        [vcs_top, vcs_right],
        [vcs_top, vcs_left],
        [vcs_bottom, vcs_left],
    ]

    T_ipm2vcsgnd = cv2.getPerspectiveTransform(
        np.array(ipm_region).astype(np.float32),
        np.array(vcs_region).astype(np.float32),
    ).astype("float32")

    return T_ipm2vcsgnd


def get_center_coord(vcs_range: tuple, size: tuple):
    """Get vcs origin coord for given vcs range and output size.

    Args:
        vcs_range: (bottom, right, top, left) order.
        size: (h, w) order
    """
    vcs_origin_coord = [
        int(
            decimal_div(
                vcs_range[2], decimal_minus(vcs_range[2], vcs_range[0])
            )
            * size[0]
        ),
        int(
            decimal_div(
                vcs_range[3], decimal_minus(vcs_range[3], vcs_range[1])
            )
            * size[1]
        ),
    ]
    return vcs_origin_coord


def get_resolution(vcs_range, size):
    """Get output resolution for given vcs range and output size.

    Args:
        vcs_range: (bottom, right, top, left) order.
        size: (h, w) order
    """
    output_resolution = (
        decimal_div(decimal_minus(vcs_range[2], vcs_range[0]), size[0]),
        decimal_div(decimal_minus(vcs_range[3], vcs_range[1]), size[1]),
    )
    return output_resolution


def crop_roi_vcs_range(
    data: np.ndarray,
    vcs_range: List,
    raw_vcs_range: List,
    return_crop_coord: bool = False,
) -> Union[Tuple[int, int, int, int], np.ndarray]:
    """Crop roi vcs range from raw gt vcs range at same vcs origin.

    Crop data such as freespace. These
    data with different size, but in original vcs range
    (-30, -51.2, 72.4, 51.2). raw_vcs_range must contain
    vcs_range.

    Args:
        data: (h, w) or (h, w, 3)
        vcs_range: values in (bottom, right, top, left) order.
        raw_vcs_range: values in (bottom, right, top, left) order.
        return_crop_coord: bool, default is false. If true, only
            return crop coord (top, left, crop_h, crop_w).
    """
    assert (
        vcs_range[0] >= raw_vcs_range[0]
        and vcs_range[1] >= raw_vcs_range[1]
        and vcs_range[2] <= raw_vcs_range[2]
        and vcs_range[3] <= raw_vcs_range[3]
    )

    h, w = data.shape[:2]
    raw_resolution = (
        decimal_div(abs(decimal_minus(raw_vcs_range[3], raw_vcs_range[1])), w),
        decimal_div(abs(decimal_minus(raw_vcs_range[2], raw_vcs_range[0])), h),
    )  # (y, x)
    top = int(
        decimal_div(
            decimal_minus(raw_vcs_range[2], vcs_range[2]), raw_resolution[1]
        )
    )  # v
    left = int(
        decimal_div(
            decimal_minus(raw_vcs_range[3], vcs_range[3]), raw_resolution[0]
        )
    )  # u
    crop_h = int(
        decimal_div(
            decimal_minus(vcs_range[2], vcs_range[0]), raw_resolution[1]
        )
    )
    crop_w = int(
        decimal_div(
            decimal_minus(vcs_range[3], vcs_range[1]), raw_resolution[0]
        )
    )
    if return_crop_coord:
        return (top, left, crop_h, crop_w)
    else:
        return data[top : top + crop_h, left : left + crop_w]


def get_area_threshod_mask(
    label: np.ndarray,
    upper_area_thresh: int = 30,
    lower_area_thresh: int = 1,
    ignore_index: int = 255,
    mask_render_label: int = 1,
) -> np.ndarray:
    """Get mask of object with area in given area range.

    Get mask of object with area >=lower_area_thresh and area<
    upper_area_thresh, where both upper_area_thresh and lower_area_thresh
    are measured in pixel numbers.

    Args:
        label: label data values in {0, 1} or {0, 1, 255}, where
            1 denotes foreground, 0 denotes background, 255 denotes ingore
            label id.
        upper_area_thresh: upper bound of small object area, measured
            in pixels. Defaults to 30.
        lower_area_thresh: lower bound of small object area,
            measured in pixels. Defaults to 1.
        ignore_index: ignore label id. Defaults to 255.
        mask_render_label: render label for small obj mask.

    Returns:
        binary mask where 1 means small object.
    """
    img = label.copy().astype(np.uint8)
    # convert to binary with white foreground and black background
    # just consider foreground obj, set ignore region to background
    img[img == ignore_index] = 0

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape, dtype=np.uint8)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if (
            contour_area < upper_area_thresh
            and contour_area >= lower_area_thresh
        ):
            cv2.fillPoly(mask, [contour], mask_render_label)
    return mask


def get_max_contour_mask(data: np.ndarray, loc: List) -> np.ndarray:
    """Get max contour contain loc coord.

    Args:
        data: binary image, (h, w).
        loc: target point location, (u, v).
    """
    mask = np.zeros(data.shape, np.uint8)
    contours, hierarchy = cv2.findContours(
        data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    max_area = 0
    if len(contours) > 0:
        cnt = contours[0]
        for contour in contours:
            if (
                cv2.contourArea(contour) > max_area
                and cv2.pointPolygonTest(contour, loc, False) >= 0
            ):
                cnt = contour
                max_area = cv2.contourArea(contour)
        # mask = np.zeros(data.shape, np.uint8)
        cv2.fillPoly(mask, [cnt], 1)

    return mask


def max_freespace_contain_ego(
    data: np.ndarray,
    vcs_range: Tuple,
    freespace_label: int = 0,
    kernel_size: int = 5,
    use_erode: bool = False,
) -> np.ndarray:
    """Get max freespace region contain ego.

    Args:
        data: (h, w)
        freespace_label: freespace label id. Defaults to 0.
        kernel_size: erode kernel size. Defaults to 5.
        use_erode: whether erode boundary to filter noise.

    Returns:
        0-1 mask, where 1 means freespace.
    """
    mask = np.zeros_like(data)
    mask[data == freespace_label] = 255  # freespace as foreground
    mask = mask.astype(np.uint8)

    vcs_origin_coord = get_center_coord(vcs_range, mask.shape)
    mask = get_max_contour_mask(mask, vcs_origin_coord[::-1])
    if use_erode:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # get max contour after filter noise
        mask = get_max_contour_mask(mask, vcs_origin_coord[::-1])

    return mask


def draw_ego(img_bev, mat_vcs2bev, dx=0, dy=0, thickness=1):
    """Draw ego car on bev img.

    Args:
        lane: lane pts
        mat_vcs2bev: transformation matrix
        dx, dy: the offset of ploted point
        thickness: line thickness
    """
    bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
    bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
    bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
    bev_ego_pts = bev_ego_pts[0, :2]
    cv2.rectangle(
        img_bev,
        (int(bev_ego_pts[0]) - 5 + dx, int(bev_ego_pts[1]) - 10 + dy),
        (int(bev_ego_pts[0]) + 5 + dx, int(bev_ego_pts[1]) + 10 + dy),
        color=(0, 255, 0),
        thickness=thickness,
    )
    return img_bev
