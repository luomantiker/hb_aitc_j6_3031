from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from shapely.geometry import MultiLineString, MultiPolygon, Polygon

from hat.utils.apply_func import convert_numpy


class Mask2Polygon(object):
    """Convert binary mask to polygon.

    Args:
        method: The method in {`none`, `simple`, `l1`, `kcos`},
            used in cv2.findContours. Default is `simple`.
        max_size_only: If ture, only return the polygon with the largest area.
            Default is False,
    """

    def __init__(self, method: str = "simple", max_size_only: bool = False):
        cv2_ploygon_codes = {
            "none": cv2.CHAIN_APPROX_NONE,
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "l1": cv2.CHAIN_APPROX_TC89_L1,
            "kcos": cv2.CHAIN_APPROX_TC89_KCOS,
        }
        self.method = cv2_ploygon_codes[method]
        self.max_size_only = max_size_only

    def __call__(
        self, mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        mask = convert_numpy(mask, dtype=np.uint8)
        polygon = self.mask2polygon(mask)
        return polygon

    def mask2polygon(self, mask: np.ndarray):
        *_, contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            self.method,
        )
        contours = [c[:, 0] for c in contours]
        if self.max_size_only:
            contour_areas = [cv2.contourArea(c) for c in contours]
            contour = contours[contour_areas.index(max(contour_areas))]
            return contour
        else:
            return contours


class LaneMask2Polygon(Mask2Polygon):
    """Convert lane binary mask to polygon with curve fitting.

    Args:
        method: The method in {'none', 'simple', 'l1', 'kcos'}, used in
            cv2.findContours. Default is simple.
        polygon_dist_thr: The distance threshold used to estimate lane polygon.
            Default is 12.
        poly_deg: The degree to estimate curve function. Default is 5,
        cost_thr: The cost threshold used to sample points from curve.
            Default is 10.0,
        point_dist_thr: The point distance threshold used to sample points from
            curve. Default is 20.0,
        angle_thr: The angle threshold used to determine endpoints.
            Default is -0.95.
    """

    def __init__(
        self,
        method: str = "simple",
        polygon_dist_thr: int = 12,
        poly_deg: int = 5,
        cost_thr: float = 10.0,
        point_dist_thr: float = 20.0,
        angle_thr: float = -0.95,
    ):
        super().__init__(method, True)
        self.polygon_dist_thr = polygon_dist_thr
        self.poly_deg = poly_deg
        self.cost_thr = cost_thr
        self.point_dist_thr = point_dist_thr
        self.angle_thr = angle_thr

    def mask2polygon(self, mask):
        # pad and dilate to align with prelabel platform
        mask = np.pad(mask, 1)[1:, 1:]
        kernel = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]).astype(np.uint8)
        mask = cv2.dilate(mask, kernel=kernel)
        *_, contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            self.method,
        )
        # filter out contours with points_num less than 3
        contours = [contour[:, 0] for contour in contours if len(contour) >= 3]
        contour_areas = [cv2.contourArea(c) for c in contours]
        contour = contours[contour_areas.index(max(contour_areas))]
        # convert to shapely.polygon
        raw_polygon = self.contour2polygon(contour, 0)
        # find head and tail to split_into_point_sets
        points_set1, points_set2 = self.split_contour_to_point_sets(contour)
        # points to polyline
        points_set_curve1, convert_xy = self.get_curve_point_set(points_set1)
        points_set_curve2, _ = self.get_curve_point_set(
            points_set2, convert_xy
        )
        # get lane contour
        lane_contour = np.vstack((points_set_curve1, points_set_curve2[::-1]))
        # within image
        lane_polygon = self.contour2polygon(lane_contour, 0)
        h, w = mask.shape[:2]
        lane_polygon = lane_polygon.intersection(
            Polygon(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
        )
        # lane estimation failed
        distance = raw_polygon.hausdorff_distance(lane_polygon)
        if distance > self.polygon_dist_thr or lane_polygon.area == 0:
            lane_polygon = raw_polygon.simplify(1)
        # convert to np.ndarray
        lane_contour = self.polygon2contour(lane_polygon)
        return lane_contour

    def split_contour_to_point_sets(self, contour):
        # start from one of the most distant point-pair
        distances = ((contour[:, None, :] - contour[None, :, :]) ** 2).sum(
            axis=2
        )
        i, j = [_[0] for _ in np.where(distances == distances.max())]
        contour = np.roll(contour, -i, 0)

        polygon = self.contour2polygon(contour, 0)
        polygon = polygon.simplify(8)
        contour_simple = self.polygon2contour(polygon)

        angle = self.get_angle(contour_simple, 1)
        keep = angle > self.angle_thr
        contour_simple = contour_simple[keep]
        angle = angle[keep]

        # find most distant point-pair
        distances = (
            (contour_simple[:, None, :] - contour_simple[None, :, :]) ** 2
        ).sum(axis=2)
        i, j = [_[0] for _ in np.where(distances == distances.max())]
        i, j = min(i, j), max(i, j)

        head_candidates = [(i, angle[i])]
        tail_candidates = [(j, angle[j])]
        point_num = len(contour_simple)
        candidates = set(  # noqa: C401
            index % point_num for index in [i - 1, i + 1, j - 1, j + 1]
        )
        dist_head2tail = (
            sum((contour_simple[j] - contour_simple[i]) ** 2) ** 0.5
        )
        for index in candidates:
            if index in [i, j]:
                continue
            dist2head = (
                sum((contour_simple[index] - contour_simple[i]) ** 2) ** 0.5
            )
            dist2tail = (
                sum((contour_simple[index] - contour_simple[j]) ** 2) ** 0.5
            )
            if min(dist2head, dist2tail) > 0.25 * dist_head2tail:
                continue
            if dist2head < dist2tail:
                head_candidates.append((index, angle[index]))
            else:
                tail_candidates.append((index, angle[index]))
        if len(head_candidates) == 1:
            head_candidates.append(head_candidates[0])
        if len(head_candidates) > 2:
            head_candidates = sorted(head_candidates, key=lambda x: x[1])[-2:]
        head_candidates = [contour_simple[_[0]] for _ in head_candidates]
        if len(tail_candidates) == 1:
            tail_candidates.append(tail_candidates[0])
        if len(tail_candidates) > 2:
            tail_candidates = sorted(tail_candidates, key=lambda x: x[1])[-2:]
        tail_candidates = [contour_simple[_[0]] for _ in tail_candidates]

        head_index = [
            np.argmin(((contour - head_candidates[0]) ** 2).sum(axis=1)),
            np.argmin(((contour - head_candidates[1]) ** 2).sum(axis=1)),
        ]
        tail_index = [
            np.argmin(((contour - tail_candidates[0]) ** 2).sum(axis=1)),
            np.argmin(((contour - tail_candidates[1]) ** 2).sum(axis=1)),
        ]
        if min(head_index) > min(tail_index):
            head_index, tail_index = tail_index, head_index

        head_index = sorted(head_index)
        tail_index = sorted(tail_index)
        # split into to sets
        if head_index[1] <= tail_index[0]:
            points_set1 = contour[head_index[1] : tail_index[0] + 1]
            points_set2 = np.vstack(
                (contour[tail_index[1] :], contour[: head_index[0] + 1])
            )
        elif head_index[1] >= tail_index[1]:
            points_set1 = contour[head_index[0] : tail_index[0] + 1]
            points_set2 = contour[tail_index[1] : head_index[1] + 1]
        else:
            if tail_index[0] - head_index[0] > head_index[1] - tail_index[0]:
                points_set1 = contour[head_index[0] : tail_index[0] + 1]
                points_set2 = contour[head_index[1] : tail_index[1] + 1]
            else:
                points_set1 = contour[tail_index[0] : head_index[1] + 1]
                points_set2 = np.vstack(
                    (contour[tail_index[1] :], contour[: head_index[0] + 1])
                )
        return points_set1, points_set2

    def sample_points_from_curve(self, points_set):
        point_num = len(points_set)
        if point_num <= 2:
            return points_set
        if points_set[-1, 0] - points_set[0, 0] <= self.point_dist_thr:
            return points_set[np.array([0, -1])]

        cost_left = (points_set[:, 1] - points_set[0, 1]) / (
            points_set[:, 0] - points_set[0, 0]
        ).clip(min=1)
        cost_left = (
            cost_left.reshape(point_num, 1)
            * np.arange(point_num).reshape(1, point_num)
            + points_set[0, 1]
        )
        cost_left = np.abs(cost_left - points_set[:, 1])

        cost_right = (points_set[:, 1] - points_set[-1, 1]) / (
            points_set[-1, 0] - points_set[:, 0]
        ).clip(min=1)
        cost_right = (
            cost_right.reshape(point_num, 1)
            * (np.arange(point_num, 0, -1) - 1).reshape(1, point_num)
            + points_set[-1, 1]
        )
        cost_right = np.abs(cost_right - points_set[:, 1])

        cost = np.tril(cost_left) + np.triu(cost_right)
        cost = cost.sum(1)
        max_cost = cost.max()
        point_dist_thr_for_split = int(max(1, self.point_dist_thr / 4))
        cost[:point_dist_thr_for_split] = np.inf
        cost[-point_dist_thr_for_split:] = np.inf

        min_cost = cost.min()
        if max_cost - min_cost < self.cost_thr:
            return points_set[np.array([0, -1])]

        split_index = np.where(cost == min_cost)[0][0]

        left_points_set = points_set[: split_index + 1]
        right_points_set = points_set[split_index:]

        left_points_set = self.sample_points_from_curve(left_points_set)
        right_points_set = self.sample_points_from_curve(right_points_set)
        return np.vstack((left_points_set[:-1], right_points_set))

    def get_curve_point_set(self, points_set, convert_xy=None):
        if len(points_set) <= 2:
            return points_set, None

        xdata = points_set[:, 0]
        ydata = points_set[:, 1]

        if convert_xy is None:
            convert_xy = max(ydata) - min(ydata) > 0.5 * (
                max(xdata) - min(xdata)
            )

        if convert_xy:
            xdata, ydata = ydata, xdata

        curve_func = np.polynomial.polynomial.Polynomial.fit(
            xdata, ydata, self.poly_deg
        )
        x_range = np.arange(min(xdata), max(xdata) + 1)
        y_range = curve_func(x_range)
        curve_points_set = np.stack((x_range, y_range), 1)
        curve_points_set = self.sample_points_from_curve(curve_points_set)

        if convert_xy:
            curve_points_set = curve_points_set[:, ::-1]
        return curve_points_set, convert_xy

    def get_angle(self, contour, sample_distance):
        angle = self.get_angle_core(
            np.roll(contour, sample_distance, 0) - contour,
            np.roll(contour, -sample_distance, 0) - contour,
        )
        return angle

    def get_angle_core(self, v1, v2):
        return (v1 * v2).sum(1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        ).clip(min=1)

    def contour2polygon(self, contour, buffer=0):
        polygon = Polygon(contour).buffer(buffer)
        if isinstance(polygon, MultiPolygon):
            areas = [p.area for p in polygon.geoms]
            polygon = polygon.geoms[areas.index(max(areas))]
        return polygon

    def polygon2contour(self, polygon):
        if isinstance(polygon, MultiPolygon):
            areas = [p.area for p in polygon.geoms]
            polygon = polygon.geoms[areas.index(max(areas))]

        boundary = polygon.boundary
        if isinstance(boundary, MultiLineString):
            contours = []
            areas = []
            for boundary_i in boundary.geoms:
                contour = np.array(
                    [c for c in boundary_i.coords][:-1]  # noqa: C416
                )
                contours.append(contour)
                areas.append(Polygon(boundary_i).area)
            contour = contours[areas.index(max(areas))]
        else:
            contour = np.array([c for c in boundary.coords][:-1])  # noqa: C416
        return contour


class Parsing2Polygon(object):
    """Convert class_id map to polygons.

    Args:
        method: The method in {`none`, `simple`, `l1`, `kcos`},
            used in cv2.findContours. Default is `simple`.
        tolerant: The maximum distance between the original curve and its
            approximation. Default is 1.
        area_thr: Regions with pixels less than area_thr will be deleted.
            Default is 9.
    """

    def __init__(
        self,
        method: str = "simple",
        tolerant: int = 1,
        area_thr: int = 9,
    ):
        cv2_ploygon_codes = {
            "none": cv2.CHAIN_APPROX_NONE,
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "l1": cv2.CHAIN_APPROX_TC89_L1,
            "kcos": cv2.CHAIN_APPROX_TC89_KCOS,
        }
        self.method = cv2_ploygon_codes[method]
        self.tolerant = tolerant
        self.area_thr = area_thr

    def mask2polygon(self, mask):
        mask = mask.astype(np.uint8)
        mask = np.pad(mask, 1)[1:, 1:]
        kernel = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]).astype(np.uint8)
        mask = cv2.dilate(mask, kernel=kernel)

        *_, polygons, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            self.method,
        )
        return polygons

    def get_polygons(self, label_map, class_id):
        mask = label_map == class_id
        region_num, id_map = cv2.connectedComponents(mask.astype(np.uint8))
        polygons_all = []
        for index in range(1, region_num):
            mask = id_map == index
            polygons = self.mask2polygon(mask.astype(np.uint8))
            polygons_all.extend(polygons)
        return polygons_all

    def parsing_to_polygon(self, pred_seg):
        polygon_classid_areas = []
        for class_id in np.unique(pred_seg):
            polygons = self.get_polygons(pred_seg, class_id)
            for polygon in polygons:
                point_num = len(polygon)
                # filter out unvalid
                if point_num <= 2:
                    continue
                # simplify
                if self.tolerant > 0 and point_num > 4:
                    polygon = cv2.approxPolyDP(polygon, self.tolerant, False)
                area = cv2.contourArea(polygon)
                # filter out tiny regions
                if area < self.area_thr:
                    continue
                polygon_classid_areas.append((polygon[:, 0], class_id, area))
        polygon_classid_areas.sort(key=lambda x: x[2], reverse=True)
        index_polygon_classids = [
            (ind, p, c) for ind, (p, c, _) in enumerate(polygon_classid_areas)
        ]
        return index_polygon_classids

    def __call__(
        self, pred_seg: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[int, np.ndarray, int]]:
        pred_seg = convert_numpy(pred_seg, dtype=np.uint8)
        index_polygon_classids = self.parsing_to_polygon(pred_seg)
        return index_polygon_classids
