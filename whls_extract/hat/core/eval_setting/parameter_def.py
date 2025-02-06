# Copyright (c) Horizon Robotics. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
@dataclass
class DetectionSettingConfig:
    classname: Optional[str] = None
    clip_hard_thresh: float = 1.0
    clip_hard_overlap: float = 0.5
    clip_to_image: bool = False
    clip_region: Optional[List] = None
    dynamic_clip: bool = False
    lower_overlap_for_small: bool = False
    score_thresh: float = -100.0
    cutin: Optional[Dict] = None
    attrs: Optional[Dict] = None
    keyPartOcclusion: Optional[Dict] = None
    criticalRegions: Optional[List] = None
    default_type: str = "normal"
    focus_gt: bool = False
    focus_gt_iou: float = 0.1
    default_overlap: float = 0.5
    normal_iou_type: str = "iou"
    hard_iou_type: str = "iou"
    ignore_iou_type: str = "iod"
    crowd_mode: bool = False
    crowd_iou_thresh: float = 1e-6
    min_crowd_group_size: int = 2
    det_cls: Optional[List] = None
    gt_cls_attr_name: str = ""
    det_cls_attr_name: str = ""
    eval_types: Optional[List] = None
    gt2eval: Optional[Dict] = None
    det2eval: Optional[Dict] = None
    image_tags: Optional[List] = None
    rois: Optional[List] = None
    det_min_ages: Optional[List] = None
    bbox_undistort: bool = False
    error_types: Optional[List] = None
    target_recalls: Optional[List] = None
    target_precisions: Optional[List] = None
    target_thresholds: Optional[List] = None
    write_eval_stats_sep_num_workers: Optional[int] = 0
    keep_ignore_data: Optional[bool] = True
    negative_key: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return self.__getattribute__(key)

    def __post_init__(self) -> None:
        if self.clip_region is None:
            self.clip_region = []
        if self.cutin is None:
            self.cutin = {}
        if self.attrs is None:
            self.attrs = {}
        if self.keyPartOcclusion is None:
            self.keyPartOcclusion = {}
        if self.criticalRegions is None:
            self.criticalRegions = []
        if self.det_cls is None:
            self.det_cls = []
        if self.eval_types is None:
            self.eval_types = []
        if self.gt2eval is None:
            self.gt2eval = {}
        if self.det2eval is None:
            self.det2eval = {}
        if self.image_tags is None:
            self.image_tags = []
        if self.rois is None:
            self.rois = []
        if self.det_min_ages is None:
            self.det_min_ages = []
        if self.error_types is None:
            self.error_types = []
        if self.target_recalls is None:
            self.target_recalls = []
        if self.target_precisions is None:
            self.target_precisions = []
        if self.target_thresholds is None:
            self.target_thresholds = []


@dataclass
class ClassificationSettingConfig:
    annokey_first_layer: str = ""
    annokey_second_layer: str = ""
    eval_categorys: Optional[List] = None
    full_image_as_bbox: bool = False
    pred_to_eval_category: Optional[Dict] = None
    gt_to_eval_category: Optional[Dict] = None
    ignores: Optional[List] = None
    image_tags: Optional[List] = None
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    score_process: Optional[str] = None
    with_scores: bool = True
    pred_prefix: str = ""
    pred_to_gt_allow_confusion: Optional[Dict] = None
    min_gt_num: int = 0
    use_valid_categorys: bool = False
    cm_x_label_rot: int = 45
    write_eval_stats_sep_num_workers: Optional[int] = 0
    show_confusion_matrix_samples: bool = True
    output_json: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.__getattribute__(key)

    def __post_init__(self) -> None:
        if self.gt_to_eval_category is None:
            self.gt_to_eval_category = {}
        if self.ignores is None:
            self.ignores = []
