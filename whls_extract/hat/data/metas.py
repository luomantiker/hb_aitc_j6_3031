# Copyright (c) Horizon Robotics. All rights reserved.

__all__ = ["img_metas"]


img_metas = [
    # img info
    "data_desc",
    "image_name",
    "image_id",
    "img_name",
    "img_height",
    "img_width",
    "img_id",
    "img",
    "layout",
    "ori_img",
    "orig_hw",
    "color_space",
    "img_shape",
    "scale_factor",
    "crop_offset",
    "before_crop_shape",
    "pad_shape",
    "scale",
    "scale_idx",
    "before_pad_shape",
    "resized_shape",
    "resized_ori_img",
    "keep_ratio",
    "before_pad_shape",
    # cls info
    "labels",
    "gt_labels",
    # bbox info
    "gt_bboxes",
    "ig_bboxes",
    "gt_classes",
    "gt_difficult",
    # seg info
    "gt_seg",
    "gt_seg_weight",
    # flow info
    "gt_flow",
    "gt_ori_flow",
    # ldmk
    "gt_ldmk",
    "ldmk_pairs",
    # track
    "frame_index",
    # faster-rcnn keys
    "im_hw",
    "gt_boxes",
    "gt_boxes_num",
    "ig_regions",
    "ig_regions_num",
    # raw-info
    "bit_nums_upper",
    "bit_nums_lower",
    "channels",
    "raw_pattern",
    "cur_pattern",
    # 2pe crop
    "crop_roi",
]
