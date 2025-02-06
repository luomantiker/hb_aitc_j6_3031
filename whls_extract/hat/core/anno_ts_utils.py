import json
import os
import sys
import warnings
from builtins import str as unicode
from typing import Dict, List, Optional

import cv2
import numpy as np

from hat.data.utils import decode_img

__all__ = [
    "check_obj",
    "ImageRecord",
    "RecordInstance",
    "drawcontour",
    "imread",
    "imwrite",
    "draw_mask",
    "check_parsing_ignore",
    "verify_image",
    "get_image_shape",
    "get_img_idx_to_img_record_map",
]

numeric_types = (float, int, np.generic, list, dict)


def _unicode2str(value):
    if isinstance(value, dict):
        for k, v in list(value.items()):
            value.pop(k)
            value[_unicode2str(k)] = _unicode2str(v)

    if isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = _unicode2str(v)

    if (sys.version_info[0] < 3 and isinstance(value, unicode)) or (
        sys.version_info[0] >= 3 and isinstance(value, str)
    ):
        return value.encode("utf-8")
    else:
        return value


def list2dict(cur_list):
    """Change a list into dict."""
    d = []
    for value in cur_list:
        if isinstance(value, DictSerializableObject):
            d.append(value.to_dict())
        elif isinstance(value, DictSerializableStringObject):
            d.append(value.to_dict())
        elif isinstance(value, (list, tuple)):
            d.append(list2dict(value))
        else:
            d.append(value)
    return d


class DictSerializableStringObject(object):
    """DictSerializableStringObject that serialize class into json.

    Note that it ignores all private or protected member.
    """

    def __init__(self):
        super(DictSerializableStringObject, self).__init__()

    def to_dict(self, flattened=False):
        d = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or key.startswith("__"):
                continue
            if isinstance(value, DictSerializableObject):
                d[key] = str(value.to_json())
            elif isinstance(value, DictSerializableStringObject):
                d[key] = str(value.to_json())
            elif isinstance(value, list):
                d[key] = json.dumps(
                    list2dict(value), indent=2, ensure_ascii=False
                )
            else:
                d[key] = str(value)

        def _flatten_dict(dict_obj, prefix):
            ret = {}
            for k, v in dict_obj.items():
                if not isinstance(v, dict):
                    k = "%s.%s" % (prefix, k) if prefix != "" else k
                    ret[k] = str(v)
                else:
                    k = "%s.%s" % (prefix, k) if prefix != "" else k
                    ret.update(_flatten_dict(v, prefix=k))
            return ret

        if flattened:
            d = _flatten_dict(d, "")
        return d

    def from_dict(self, dict_obj, allow_missing=True, flattened=False):
        if isinstance(dict_obj, str):
            dict_obj = json.loads(dict_obj)
        assert isinstance(dict_obj, dict), "Given: %s" % (type(dict_obj))
        ret = {}

        def _set_by_flattened_member(obj, mem_str, value):
            split_name = mem_str.split(".")
            cur_obj = obj
            last_id = len(split_name) - 1
            for i in range(len(split_name)):
                cur_name = split_name[i]
                obj_dict = (
                    cur_obj if isinstance(cur_obj, dict) else cur_obj.__dict__
                )
                if cur_name in obj_dict:
                    if i < last_id:
                        cur_obj = obj_dict[cur_name]
                    else:
                        obj_dict[cur_name] = value
                        return True
                else:
                    return False

        for k, v in dict_obj.items():
            if not flattened:
                if k not in self.__dict__:
                    if not allow_missing:
                        raise KeyError(
                            "Undefined key: %s. Options: %s"
                            % (k, self.__dict__.keys())
                        )
                    else:
                        ret[k] = v
                else:
                    cur_value = self.__dict__[k]
                    if isinstance(cur_value, DictSerializableObject):
                        ret.update(
                            self.__dict__[k].from_dict(
                                v, allow_missing=allow_missing
                            )
                        )
                    else:
                        self.__dict__[k] = v
            else:
                success = _set_by_flattened_member(self, k, v)
                if not success:
                    if not allow_missing:
                        raise KeyError(
                            "Undefined key: %s. Options: %s"
                            % (k, self.to_dict(True).keys())
                        )
                    else:
                        ret[k] = v
        return ret

    def to_json(self, indent=2, ensure_ascii=False):
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=ensure_ascii
        )

    def from_json(self, obj, allow_missing=True):
        assert isinstance(
            obj, (str, unicode)
        ), "only accept str. Given: %s" % (type(obj))
        obj_dict = json.loads(obj)
        self.from_dict(obj_dict, allow_missing=allow_missing)

    def __repr__(self):
        return self.to_json()


class DictSerializableObject(object):
    """DictSerializableObject that serialize class into json.

    Note that it ignores all private or protected member.
    """

    def __init__(self):
        super(DictSerializableObject, self).__init__()

    def to_dict(self, flattened=False):
        """Change this class into dict."""
        d = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or key.startswith("__"):
                continue
            if isinstance(value, DictSerializableObject):
                d[key] = value.to_dict()
            elif isinstance(value, DictSerializableStringObject):
                d[key] = str(value.to_json())
            elif isinstance(value, (list, tuple)):
                d[key] = list2dict(value)
            else:
                d[key] = value

        def _flatten_dict(dict_obj, prefix):
            ret = {}
            for k, v in dict_obj.items():
                if not isinstance(v, dict):
                    k = "%s.%s" % (prefix, k) if prefix != "" else k
                    ret[k] = v
                else:
                    k = "%s.%s" % (prefix, k) if prefix != "" else k
                    ret.update(_flatten_dict(v, prefix=k))
            return ret

        if flattened:
            d = _flatten_dict(d, "")
        return d

    def from_dict(self, dict_obj, allow_missing=True, flattened=False):
        """Construct member from dict."""
        assert isinstance(dict_obj, dict), "Given: %s" % (type(dict_obj))
        ret = {}

        def _set_by_flattened_member(obj, mem_str, value):
            split_name = mem_str.split(".")
            cur_obj = obj
            last_id = len(split_name) - 1
            for i, cur_name in enumerate(split_name):
                obj_dict = (
                    cur_obj if isinstance(cur_obj, dict) else cur_obj.__dict__
                )
                if cur_name in obj_dict:
                    if i < last_id:
                        cur_obj = obj_dict[cur_name]
                    else:
                        obj_dict[cur_name] = value
                        return True
                else:
                    return False
            return False

        for k, v in dict_obj.items():
            if not flattened:
                if k not in self.__dict__:
                    if not allow_missing:
                        raise KeyError(
                            "Undefined key: %s. Options: %s"
                            % (k, self.__dict__.keys())
                        )
                    else:
                        ret[k] = v
                else:
                    cur_value = self.__dict__[k]
                    if isinstance(cur_value, DictSerializableObject):
                        ret.update(
                            self.__dict__[k].from_dict(
                                v, allow_missing=allow_missing
                            )
                        )
                    else:
                        self.__dict__[k] = v
            else:
                success = _set_by_flattened_member(self, k, v)
                if not success:
                    if not allow_missing:
                        raise KeyError(
                            "Undefined key: %s. Options: %s"
                            % (k, self.to_dict(True).keys())
                        )
                    else:
                        ret[k] = v
        return ret

    def to_json(self, indent=2, ensure_ascii=False):
        """Convert to json."""
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=ensure_ascii
        )

    def from_json(self, obj, allow_missing=True):
        """Construct from json."""
        assert isinstance(
            obj, (str, unicode)
        ), "only accept str. Given: %s" % (type(obj))
        obj_dict = json.loads(obj)
        self.from_dict(obj_dict, allow_missing=allow_missing)

    def __repr__(self):
        return self.to_json()


class RoiDesc(DictSerializableObject):
    """This struct describes an ROI(region of interest).

    An ROI is composed of 3 points. The first two points describe the
    scale of ROI, and the third one describe the ROI center.

    Args:
        json_str : The dict to initialize data.
    """

    def __init__(self, json_str: Optional[Dict] = None):
        self.pt1 = [-1, -1]
        self.pt2 = [-1, -1]
        self.center = [-1, -1]
        if json_str is not None:
            self.pt1 = json_str["pt1"]
            self.pt2 = json_str["pt2"]
            self.center = json_str["center"]

    def check_element(self):
        assert isinstance(self.pt1, list)
        assert len(self.pt1) == 2
        assert isinstance(self.pt2, list)
        assert len(self.pt2) == 2
        assert isinstance(self.center, list)
        assert len(self.center) == 2


class ImageRois(DictSerializableObject):
    """This struct contains all ROIs in one image.

    Args:
        json_str : The dict to initialize data.
        force_utf8: Whether force use of utf8 format.
    """

    def __init__(
        self,
        json_str: Optional[Dict] = None,
        force_utf8: Optional[bool] = True,
    ):
        self.img_url = ""
        self.rois = []
        if json_str is not None:
            if force_utf8:
                json_str = _unicode2str(json_str)
            self.img_url = json_str["img_url"]
            for roi in json_str["rois"]:
                self.rois.append(RoiDesc(roi))

    def check_element(self):
        assert isinstance(self.rois, list)
        for roi in self.rois:
            assert isinstance(roi, RoiDesc)

    def add_roi(self, pt1, pt2, center):
        assert len(pt1) == 2
        assert len(pt2) == 2
        assert len(center) == 2
        self.rois.append(RoiDesc({"pt1": pt1, "pt2": pt2, "center": center}))


class RecordInstance(DictSerializableObject):
    """The struct to describe an object instance.

    Args:
        points_data: All the points of the instance. The size of list should
            be the same as point_num defined in ClassInfo.
        class_id: The class id. Just the index of ClassInfo that
            you defined at begining.
        attribute: The attribute value. It should match the attribute_num in
            ClassInfo
        is_hard: Whether to ignore this instance.
            If set to [1], this instance is ignored.
        is_point_hard: Whether each point in instance is hard. If it is
            provided, the size of is_point_hard should be the same as
            len(points_data). This parameter might be used in heatmap channel.
        weight_value: The weight_value of current instance.
        mask_poly: The mask poly for instance seg. Polygon stored as
                [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
        init_dict: The dict to initialize data. It will override all
            previous arguments.
    """

    def __init__(
        self,
        points_data: Optional[List[List[int]]] = None,
        class_id: Optional[int] = None,
        attribute: Optional[List] = None,
        is_hard: Optional[List[int]] = None,
        is_point_hard: Optional[List[int]] = None,
        weight_value: Optional[List[float]] = None,
        mask_poly: Optional[List[List[float]]] = None,
        init_dict: Optional[Dict] = None,
    ):
        super(RecordInstance, self).__init__()
        self.points_data = [] if points_data is None else points_data
        self.class_id = [0] if class_id is None else class_id
        self.attribute = [] if attribute is None else attribute
        self.is_hard = [0] if is_hard is None else is_hard
        self.is_point_hard = [] if is_point_hard is None else is_point_hard
        self.weight_value = [] if weight_value is None else weight_value
        self.mask_poly = [] if mask_poly is None else mask_poly
        if init_dict is not None:
            for key in self.__dict__:
                if key == "points_data":
                    if "points" in init_dict:
                        self.__dict__[key] = init_dict["points"]
                    else:
                        assert "points_data" in init_dict
                        self.__dict__[key] = init_dict[key]
                elif key in init_dict:
                    self.__dict__[key] = init_dict[key]
                else:
                    pass
        if len(self.is_point_hard) == 0:
            self.is_point_hard = [0 for i in range(len(self.points_data))]

    def __eq__(self, other):
        ret = self.points_data == other.points_data
        ret &= self.class_id == other.class_id
        ret &= self.attribute == other.attribute
        ret &= self.is_hard == other.is_hard
        ret &= self.is_point_hard == other.is_point_hard
        ret &= self.weight_value == other.weight_value
        ret &= self.mask_poly == other.mask_poly
        return ret

    def add_point(self, x, y, is_point_hard=0):
        ptr = [x, y]
        self.points_data.append(ptr)
        self.is_point_hard.append(is_point_hard)


class IgnoreRegion(DictSerializableObject):
    """The region description that should be ignored in an image.

    Args:
        contour: The contour of ignore region. If it contains two points,
            the contour will be treated as lt and rb point of bbox.
            Otherwise it will be treated as polygon contour.
        class_id: The class id of ignore region.
        point_idx: The point idx of ignored region. The region only will be
            activated for those point idx
        init_dict: The dict to initialize data. It will override all
            previous arguments.
    """

    def __init__(
        self,
        contour: List[List[int]] = None,
        class_id: List[int] = None,
        point_idx: List[int] = None,
        init_dict: Dict = None,
    ):
        super(IgnoreRegion, self).__init__()
        if init_dict is None:
            self.contour = [[-1, -1], [-1, -1]] if contour is None else contour
            self.class_id = [0] if class_id is None else class_id
            self.point_idx = [] if point_idx is None else point_idx
        else:
            self.contour = []
            if "left_top" in init_dict and "right_bottom" in init_dict:
                self.contour.append(init_dict["left_top"])
                self.contour.append(init_dict["right_bottom"])
            else:
                assert "contour" in init_dict
                self.contour = init_dict["contour"]

            self.class_id = init_dict["class_id"]
            self.point_idx = []
            if "point_idx" in init_dict:
                self.point_idx = init_dict["point_idx"]

    def __eq__(self, other):
        ret = self.contour == other.contour
        ret &= self.class_id == other.class_id
        ret &= self.point_idx == other.point_idx
        return ret

    @property
    def left_top(self):
        assert (
            len(self.contour) == 2
        ), "You only can access left_top when len(contour) == 2"
        return self.contour[0]

    @property
    def right_bottom(self):
        assert (
            len(self.contour) == 2
        ), "You only can access right_bottom when len(contour) == 2"
        return self.contour[1]

    @left_top.setter
    def left_top(self, value):
        assert (
            len(self.contour) == 2
        ), "You only can access left_top when len(contour) == 2"
        self.contour[0] = value

    @right_bottom.setter
    def right_bottom(self, value):
        assert (
            len(self.contour) == 2
        ), "You only can access right_bottom when len(contour) == 2"
        self.contour[1] = value


class ImageRecord(DictSerializableObject):
    """The image record to describe all information of one image.

    Args:
        instances: The instance annotation
        ignore_regions: The ignored region
        img_url: The image url
        parsing_map_urls: the parsing map urls
        img_h: The image height
        img_w: The image width
        img_c: The channel number of image
        idx: The index of this image. It should be unique in a dataset.
        img_attribute: The attribute to describe this image
    """

    def __init__(
        self,
        instances: Optional[
            List["gluon_densebox.anno.RecordInstance"]  # noqa: F821
        ] = None,
        ignore_regions: Optional[
            List["gluon_densebox.anno.RecordInstance"]  # noqa: F821
        ] = None,
        img_url: Optional[str] = "",
        parsing_map_urls: Optional[str] = None,
        img_h: Optional[int] = -1,
        img_w: Optional[int] = -1,
        img_c: Optional[int] = -1,
        idx: Optional[int] = -1,
        img_attribute: Optional[Dict] = None,
        init_dict: Optional[Dict] = None,
        force_utf8: Optional[bool] = False,
    ):
        super(ImageRecord, self).__init__()
        self.instances = [] if instances is None else instances
        self.ignore_regions = [] if ignore_regions is None else ignore_regions
        self.parsing_map_urls = (
            [] if parsing_map_urls is None else parsing_map_urls
        )
        self.img_url = img_url
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.idx = idx
        self.img_attribute = {} if img_attribute is None else img_attribute

        if init_dict is not None:
            if force_utf8:
                init_dict = _unicode2str(init_dict)
            self.img_url = init_dict["img_url"]
            self.img_h = init_dict["img_h"]
            self.img_w = init_dict["img_w"]
            self.img_c = init_dict["img_c"]
            if "instances" in init_dict:
                for inst in init_dict["instances"]:
                    self.instances.append(RecordInstance(init_dict=inst))
            if "parsing_map_urls" in init_dict:
                for urls in init_dict["parsing_map_urls"]:
                    self.parsing_map_urls.append(urls)
            if "ignore_regions" in init_dict:
                for ignore_region in init_dict["ignore_regions"]:
                    self.ignore_regions.append(
                        IgnoreRegion(init_dict=ignore_region)
                    )
            if "idx" in init_dict:
                self.idx = init_dict["idx"]
            if "img_attribute" in init_dict:
                self.img_attribute = init_dict["img_attribute"]

        assert isinstance(self.img_attribute, dict)
        for k, v in self.img_attribute.items():
            assert isinstance(k, str)
            assert isinstance(v, numeric_types)

    def __eq__(self, other):
        ret = self.instances == other.instances
        ret &= self.ignore_regions == other.ignore_regions
        ret &= self.parsing_map_urls == other.parsing_map_urls
        ret &= self.img_url == other.img_url
        ret &= self.img_h == other.img_h
        ret &= self.img_w == other.img_w
        ret &= self.img_c == other.img_c
        ret &= self.idx == other.idx
        ret &= self.img_attribute == other.img_attribute
        return ret

    def add_instance(self, instance):
        self.instances.append(instance)


# obj_filter
def get_bbox_x1(obj: Dict):
    """Get bounding box Left coordinate (x1).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1 = float(obj["data"][0])
    return x1


def get_bbox_y1(obj: Dict):
    """Get bounding box top coordinate (y1).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    y1 = float(obj["data"][1])
    return y1


def get_bbox_x2(obj: Dict):
    """Get bounding box right coordinate (x2).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x2 = float(obj["data"][2])
    return x2


def get_bbox_y2(obj: Dict):
    """Get bounding box bottom coordinate (y2).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    y2 = float(obj["data"][3])
    return y2


def get_bbox_height(obj: Dict):
    """Get bounding box height (y2 - y1).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1, y1, x2, y2 = map(float, obj["data"])
    height = y2 - y1
    return height


def get_bbox_width(obj: Dict):
    """Get bounding box width (x2 - x1).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    x1, y1, x2, y2 = map(float, obj["data"])
    width = x2 - x1
    return width


def get_bbox_shortside(obj: Dict):
    """Get bounding box shortside length (min(width, height)).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    height = get_bbox_height(obj)
    width = get_bbox_width(obj)
    return min(height, width)


def get_bbox_aspect_ratio(obj: Dict):
    """Get bounding box aspect ratio (height / width).

    Args:
        obj: Bouding box object, {"data": [x1, y1, x2, y2]}
    """
    return get_bbox_height(obj) / float(get_bbox_width(obj))


def get_value(obj: Dict, field: str):
    """Get object attribute value.

    Args:
        obj: Bouding box object
        field: attribute name
    """
    value = obj
    for key in field.split("."):
        value = value[key]
    return value


def _contains(lhs_dict: Dict, rhs_dict: Dict):
    """Check if the left dict contains the right dict."""
    if isinstance(lhs_dict, dict) and isinstance(rhs_dict, dict):
        for key in rhs_dict:
            if key not in lhs_dict:
                return False
            if not _contains(lhs_dict[key], rhs_dict[key]):
                return False
    else:
        return lhs_dict == rhs_dict
    return True


def _has_path(lhs_dict: Dict, path: Dict):
    """Check if the dict has a specific path."""
    if isinstance(path, str):
        if isinstance(lhs_dict, dict):
            pos = path.find(".")
            if pos > 0:
                key = path[:pos]
                subpath = path[pos + 1 :]
                return _has_path(lhs_dict.get(key), subpath)
            else:
                key = path
                return key in lhs_dict
        elif isinstance(lhs_dict, str):
            if path.find(".") > 0:
                return False
            else:
                return lhs_dict == path
    return False


def _range(obj: Dict, range_condiction: Dict):
    """Check whether the object meets the range condition."""
    field = range_condiction["field"]
    if field.startswith("$"):
        if field == "$BBOX_HEIGHT":
            value = get_bbox_height(obj)
        elif field == "$BBOX_WIDTH":
            value = get_bbox_width(obj)
        elif field == "$BBOX_ASPECT_RATIO":
            value = get_bbox_aspect_ratio(obj)
        elif field == "$BBOX_SHORTSIDE":
            value = get_bbox_shortside(obj)
        elif field == "$BBOX_X1":
            value = get_bbox_x1(obj)
        elif field == "$BBOX_Y1":
            value = get_bbox_y1(obj)
        elif field == "$BBOX_X2":
            value = get_bbox_x2(obj)
        elif field == "$BBOX_Y2":
            value = get_bbox_y2(obj)
        else:
            raise Exception("Invalid range condiction %s" % field)
    else:
        value = get_value(obj, field)

    if "lt" in range_condiction and not value < range_condiction["lt"]:
        return False
    elif "le" in range_condiction and not value <= range_condiction["le"]:
        return False
    elif "gt" in range_condiction and not value > range_condiction["gt"]:
        return False
    elif "ge" in range_condiction and not value >= range_condiction["ge"]:
        return False
    else:
        return True


def _and(obj: Dict, condictions: List[Dict]):
    """Check whether the object meets all conditions."""
    for condiction in condictions:
        if not check_obj(obj, condiction):
            return False
    return True


def _or(obj: Dict, condictions: List[Dict]):
    """Check whether the object meets any conditions."""
    for condiction in condictions:
        if check_obj(obj, condiction):
            return True
    return False


def _not(obj: Dict, condiction: List[Dict]):
    """Check whether the object does not meets condition."""
    return not check_obj(obj, condiction)


_ops = {
    "or": _or,
    "and": _and,
    "not": _not,
    "contains": _contains,
    "range": _range,
    "has_path": _has_path,
}


def check_obj(obj: Dict, condiction: Dict):
    """Check whether the object meets the condition."""
    if condiction is None:
        return False
    for op_name in _ops:
        if op_name in condiction:
            return _ops[op_name](obj, condiction[op_name])
    raise Exception("Invalid condiction keys: %s" % condiction.keys())


def drawcontour(contour: np.ndarray, value: np.ndarray, J: np.ndarray):
    """Draw contour.

    Parameters
    ----------
    contour: The input image.
    value: The segmentation label map.
    J: label colors.
    """
    if value > 255:
        return J
    M = np.zeros(J.shape, dtype=np.uint8)
    cv2.drawContours(M, [contour], -1, (1), -1)
    J[(M == 1)] = value
    return J


def imread(filename: str):
    """Read image.

    Args: The image path.
    """
    try:
        im = cv2.imread(filename, -1)
    except Exception:
        im = decode_img(filename, is_color=cv2.IMREAD_UNCHANGED)
    return im


def imwrite(filename: str, im: np.ndarray):
    """Write image.

    Args:
        filename: The image path.
        im: The image.
    """
    try:
        im = cv2.imwrite(filename, im)
    except Exception:
        im = cv2.imencode(os.path.splitext(filename)[1], im)[1].tofile(
            filename
        )


def fuse_mask(img: np.ndarray, label: np.ndarray, colors: np.ndarray):
    """Fuse label mask.

    Args:
        img: The input image.
        label: The segmentation label map.
        colors: label colors.
    """
    bgr = cv2.cvtColor(label.astype("uint8"), cv2.COLOR_GRAY2BGR)
    show = np.array(cv2.LUT(bgr, colors))
    fusion = (
        show.astype("float32") * 0.5 + np.array(img).astype("float32") * 0.5
    ).astype("uint8")
    return fusion


def draw_mask(img: np.ndarray, label: np.ndarray, colors: np.ndarray):
    """Draw label mask.

    Args:
        img: The input image.
        label: The segmentation label map.
        colors: label colors.
    """
    fusion = fuse_mask(img, label, colors)
    return np.vstack((img, fusion)).astype("uint8")


def check_parsing_ignore(label_path, ignore_thres=0.8):
    """Verify annotation of a label file for segmentation.

       If ratio of ignore region > ignore_thres, skip it.

    Args:
        label_path: str
            label path.

    Returns:
        flag: bool
            True for valid label, otherwise False.
    """
    if not os.path.exists(label_path):
        warnings.warn("cannot find label %s, ignoring..." % label_path)
        return False
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    ignore_size = np.sum(label == 255)
    if ignore_size > label.shape[0] * label.shape[1] * ignore_thres:
        warnings.warn(
            "Too many pixels are ignored in label file %s, ignoring..."
            % label_path
        )
        return False
    return True


def verify_image(image_path: str, check_image_invalid: bool = True) -> bool:
    """Verify validity of an image.

    Args:
        image_path: Image path.
        check_image_invalid: Whether to check the image invalid.

    Args:
        flag: True for valid image, otherwise False.
    """
    image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        warnings.warn("cannot find image %s, ignoring..." % image_path)
        return False
    if check_image_invalid:
        img = cv2.imread(image_path)
        if img is None:
            warnings.warn("invalid image %s, ignoring..." % image_path)
            return False
    return True


def get_image_shape(image):
    img_h, img_w = image.shape[:2]
    img_c = 1 if len(image.shape) == 2 else image.shape[2]
    return img_h, img_w, img_c


def get_img_idx_to_img_record_map(anno_path):
    assert os.path.exists(anno_path), "%s does not exists" % anno_path
    img_idx_map = {}
    with open(anno_path, mode="r") as fread:
        annos = json.load(fread)
    for img_record_i in annos["image_recs"]:
        img_record_i = ImageRecord(init_dict=img_record_i, force_utf8=False)
        img_idx = img_record_i.idx
        if img_idx < 0:
            assert not img_idx_map, "Some image record has no idx"
        else:
            assert img_idx not in img_idx_map, "duplicate img_idx %d" % img_idx
            img_idx_map[img_idx] = img_record_i
    return img_idx_map


def get_img_idx_to_img_rois_map(roi_list_path, anno_path):
    assert os.path.exists(anno_path), "%s does not exists" % anno_path
    assert os.path.exists(roi_list_path), "%s does not exists" % roi_list_path
    img_idx_img_rois_map = {}
    with open(roi_list_path) as fread:
        roi_lists = json.load(fread)

    with open(anno_path, mode="r") as fread:
        annos = json.load(fread)
    assert len(roi_lists["image_rois"]) == len(
        annos["image_recs"]
    ), f'record size unmatched, \
        image_recs size: {len(annos["image_recs"])}, \
        roi list size: {len(roi_lists["image_rois"])}'

    for anno_idx, img_record_i in enumerate(annos["image_recs"]):
        img_record_i = ImageRecord(init_dict=img_record_i, force_utf8=False)
        img_idx = img_record_i.idx

        if img_idx < 0:
            assert not img_idx_img_rois_map, "Some image record has no idx"
        else:
            assert img_idx not in img_idx_img_rois_map, (
                "duplicate img_idx %d" % img_idx
            )

        image_roi = ImageRois(
            json_str=roi_lists["image_rois"][anno_idx], force_utf8=False
        )
        assert (
            image_roi.img_url == img_record_i.img_url
        ), f"image_url unmatched, \
            image_recs[{anno_idx}].img_url: {img_record_i.img_url},  \
            image_rois[{anno_idx}].img_url: {image_roi.img_url}"
        img_idx_img_rois_map[img_idx] = image_roi
    return img_idx_img_rois_map
