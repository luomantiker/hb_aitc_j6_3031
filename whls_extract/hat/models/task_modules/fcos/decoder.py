# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection

from collections import OrderedDict
from math import pi as PI
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from hat.core.data_struct.base_struct import DetBoxes2D
from hat.models.base_modules.postprocess import PostProcessorBase
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .target import distance2bbox, get_points

try:
    from torchvision.ops.boxes import batched_nms
except ImportError:
    batched_nms = None


__all__ = [
    "FCOSDecoder",
    "FCOSDecoderWithConeInvasion",
    "FCOSDecoder4RCNN",
    "VehicleSideFCOSDecoder",
    "multiclass_nms",
    "FCOSDecoderForFilter",
]


def _get_bbox_with_tanalpha(
    det_bboxes: torch.Tensor, bbox_preds: torch.Tensor, has=True
):
    new_bbox_list = []
    for det_bbox in det_bboxes:
        bbox_pred = bbox_preds[
            (bbox_preds[..., :4] - det_bbox[:4]).sum(1).abs() < 1e-4
        ]
        if has:
            assert len(bbox_pred) == 1
        bbox_pred = bbox_pred[0]
        new_bbox = torch.cat((bbox_pred[:4], det_bbox[4:6], bbox_pred[4:]))
        new_bbox_list.append(new_bbox[None, :])
    return torch.cat(new_bbox_list)


@OBJECT_REGISTRY.register
class FCOSDecoder(PostProcessorBase):  # noqa: D205,D400
    """

    Args:
        num_classes: Number of categories excluding the background category.
        strides: A list contains the strides of fcos_head output.
        transforms: A list contains the transform config.
        inverse_transform_key: A list contains the inverse transform info key.
        nms_use_centerness: If True, use centerness as a factor in nms
            post-processing.
        nms_sqrt: If True, sqrt(score_thr * score_factors).
        test_cfg: Cfg dict, including some configurations of nms.
        input_resize_scale: The scale to resize bbox.
        truncate_bbox: If True, truncate the predictive bbox out of image
            boundary. Default True.
        filter_score_mul_centerness: If True, filter out bbox by score multiply
            centerness, else filter out bbox by score. Default False.
        meta_data_bool: Whether get shape info from meta data.
        label_offset: label offset.
        upscale_bbox_pred: Whether upscale bbox preds.
        bbox_relu: Whether apply relu to bbox preds.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int],
        transforms: Optional[Sequence[dict]] = None,
        inverse_transform_key: Optional[Sequence[str]] = None,
        nms_use_centerness: bool = True,
        nms_sqrt: bool = True,
        test_cfg: Optional[dict] = None,
        input_resize_scale: Optional[Union[float, torch.Tensor]] = None,
        truncate_bbox: bool = True,
        filter_score_mul_centerness: bool = False,
        meta_data_bool: bool = True,
        label_offset: int = 0,
        upscale_bbox_pred: bool = False,
        bbox_relu: bool = False,
        to_cpu: bool = False,
    ):
        super(FCOSDecoder, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.transforms = transforms
        self.inverse_transform_key = inverse_transform_key
        self.nms_use_centerness = nms_use_centerness
        self.nms_sqrt = nms_sqrt
        self.test_cfg = test_cfg
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        self.truncate_bbox = truncate_bbox
        self.filter_score_mul_centerness = filter_score_mul_centerness
        self.meta_data_bool = meta_data_bool
        self.label_offset = label_offset
        self.upscale_bbox_pred = upscale_bbox_pred
        self.bbox_relu = bbox_relu
        self.to_cpu = to_cpu

    def forward(self, pred: Sequence[torch.Tensor], meta_data: Dict[str, Any]):
        if isinstance(pred, dict):
            # dict is used for the output of onnx/trt ir.
            # assert order is cls_scores, bbox_preds, centernesses.
            pred = [pred[key] for key in sorted(pred)]
            step = int(len(pred) / 3)
            cls_scores = pred[:step]
            bbox_preds = pred[step : 2 * step]
            centernesses = pred[2 * step :]
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, centernesses = pred

        if self.bbox_relu:
            bbox_preds = [torch.nn.functional.relu(i) for i in bbox_preds]
        if self.upscale_bbox_pred:
            bbox_preds = list(bbox_preds)
            for i in range(len(bbox_preds)):
                bbox_preds[i] *= self.strides[i]
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # [cxh1xw1, cxh2xw2, cxh3xw3, ...]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [4xh1xw1, 4xh2xw2, 4xh3xw3, ...]
            if centernesses is not None:
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            else:
                centerness_pred_list = [None] * len(cls_score_list)

            if self.meta_data_bool:
                h_index = meta_data["layout"][img_id].index("h")
                w_index = meta_data["layout"][img_id].index("w")
                if "pad_shape" in meta_data:
                    max_shape = (
                        meta_data["pad_shape"][img_id][h_index],
                        meta_data["pad_shape"][img_id][w_index],
                    )
                elif "img_shape" in meta_data:
                    max_shape = (
                        meta_data["img_shape"][img_id][h_index],
                        meta_data["img_shape"][img_id][w_index],
                    )
                else:
                    max_shape = (
                        meta_data["img_height"][img_id],
                        meta_data["img_width"][img_id],
                    )
                max_shape = max_shape if self.truncate_bbox else None
                inverse_info = {}
                for key, value in meta_data.items():
                    if (
                        self.inverse_transform_key
                        and key in self.inverse_transform_key
                    ):
                        inverse_info[key] = value[img_id]
            else:
                max_shape = None
                inverse_info = {}

            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_bboxes)
        if self.to_cpu:
            det_results = [i.cpu() for i in det_results]
            results["img_id"] = meta_data["img_id"].cpu()
        if self.meta_data_bool:
            results["pred_bboxes"] = det_results
            results["img_name"] = meta_data["img_name"]
            results["img_id"] = meta_data["img_id"]
            return results
        else:
            results = []
            for det_bboxes in det_results:
                boxes = DetBoxes2D(
                    boxes=det_bboxes[:, :4],
                    scores=det_bboxes[:, -2],
                    cls_idxs=det_bboxes[:, -1],
                )
                results.append(boxes)

            return {"pred_boxes": results}

    def _check_shape(self, data, pred):
        flag = True
        for i in data:
            if len(i) != pred.shape[0]:
                flag = False
        return flag

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        centerness_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        """Decode the output of a single picture into a prediction result.

        Args:
            cls_score_list (list[torch.Tensor]): List of all levels' cls_score,
                each has shape (N, num_points * num_classes, H, W).
            bbox_pred_list (list[torch.Tensor]): List of all levels' bbox_pred,
                each has shape (N, num_points * 4, H, W).
            centerness_pred_list (list[torch.Tensor]): List of all levels'
                centerness_pred, each has shape (N, num_points * 1, H, W).
            mlvl_points (list[torch.Tensor]): List of all levels' points.
            max_shape (Sequence): Maximum allowable shape of the decoded bbox.

        Returns:
            det_bboxes (torch.Tensor): Decoded bbox, with shape (N, 6),
                represents x1, y1, x2, y2, cls_score, cls_id (0-based).
        """
        cfg = self.test_cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
            cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.num_classes)
                .sigmoid()
            )
            if has_centerness:
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                if centerness is not None:
                    centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=max_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.input_resize_scale is not None:
            mlvl_bboxes[:, :5] /= self.input_resize_scale
        # inverse transform for mapping to the original image
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels to [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if has_centerness:
            mlvl_centerness = torch.cat(mlvl_centerness)
        else:
            mlvl_centerness = None  # score_factors wile be set as None
        score_thr = cfg.get("score_thr", "0.05")
        nms = cfg.get("nms").get("name", "nms")
        iou_threshold = cfg.get("nms").get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
            label_offset=self.label_offset,
        )
        return det_bboxes


@OBJECT_REGISTRY.register
class FCOSDecoderWithConeInvasion(FCOSDecoder):  # noqa: D205,D400
    """

    Args:
        num_classes: Number of categories excluding the background category.
        strides: A list contains the strides of fcos_head output.
        transforms: A list contains the transform config.
        inverse_transform_key: A list contains the inverse transform info key.
        nms_use_centerness: If True, use centerness as a factor in nms
            post-processing.
        nms_sqrt: If True, sqrt(score_thr * score_factors).
        test_cfg: Cfg dict, including some configurations of nms.
        input_resize_scale: The scale to resize bbox.
        truncate_bbox: If True, truncate the predictive bbox out of image
            boundary. Default True.
        filter_score_mul_centerness: If True, filter out bbox by score multiply
            centerness, else filter out bbox by score. Default False.
        meta_data_bool: Whether get shape info from meta data.
        label_offset: label offset.
        upscale_bbox_pred: Whether upscale bbox preds.
        bbox_relu: Whether apply relu to bbox preds.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int],
        transforms: Optional[Sequence[dict]] = None,
        inverse_transform_key: Optional[Sequence[str]] = None,
        nms_use_centerness: bool = True,
        nms_sqrt: bool = True,
        test_cfg: Optional[dict] = None,
        input_resize_scale: Optional[Union[float, torch.Tensor]] = None,
        truncate_bbox: bool = True,
        filter_score_mul_centerness: bool = False,
        meta_data_bool: bool = True,
        label_offset: int = 0,
        upscale_bbox_pred: bool = False,
        bbox_relu: bool = False,
    ):
        super(FCOSDecoderWithConeInvasion, self).__init__(
            num_classes,
            strides,
            transforms,
            inverse_transform_key,
            nms_use_centerness,
            nms_sqrt,
            test_cfg,
            input_resize_scale,
            truncate_bbox,
            filter_score_mul_centerness,
            meta_data_bool,
            label_offset,
            upscale_bbox_pred,
            bbox_relu,
        )

    def forward(self, pred: Sequence[torch.Tensor], meta_data: Dict[str, Any]):
        assert len(pred) == 6, (
            "pred must be a tuple containing cls_scores,"
            "bbox_preds, centernesses, invasion_state, "
            "beside_valid_conf, invasion_scale"
        )
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            (
                cls_scores,
                bbox_preds,
                centernesses,
                invasion_state,
                beside_valid_conf,
                invasion_scale,
            ) = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            (
                cls_scores,
                bbox_preds,
                centernesses,
                invasion_state,
                beside_valid_conf,
                invasion_scale,
            ) = pred

        if self.bbox_relu:
            bbox_preds = [torch.nn.functional.relu(i) for i in bbox_preds]
        if self.upscale_bbox_pred:
            bbox_preds = list(bbox_preds)
            for i in range(len(bbox_preds)):
                bbox_preds[i] *= self.strides[i]
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # [cxh1xw1, cxh2xw2, cxh3xw3, ...]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [4xh1xw1, 4xh2xw2, 4xh3xw3, ...]
            if centernesses is not None:
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            else:
                centerness_pred_list = [None] * len(cls_score_list)
            invasion_state_pred_list = [
                invasion_state[i][img_id].detach() for i in range(num_levels)
            ]
            beside_valid_conf_pred_list = [
                beside_valid_conf[i][img_id].detach()
                for i in range(num_levels)
            ]
            invasion_scale_pred_list = [
                invasion_scale[i][img_id].detach() for i in range(num_levels)
            ]

            if self.meta_data_bool:
                h_index = meta_data["layout"][img_id].index("h")
                w_index = meta_data["layout"][img_id].index("w")
                if "pad_shape" in meta_data:
                    max_shape = (
                        meta_data["pad_shape"][img_id][h_index],
                        meta_data["pad_shape"][img_id][w_index],
                    )
                elif "img_shape" in meta_data:
                    max_shape = (
                        meta_data["img_shape"][img_id][h_index],
                        meta_data["img_shape"][img_id][w_index],
                    )
                else:
                    max_shape = (
                        meta_data["img_height"][img_id],
                        meta_data["img_width"][img_id],
                    )
                max_shape = max_shape if self.truncate_bbox else None
                inverse_info = {}
                for key, value in meta_data.items():
                    if (
                        self.inverse_transform_key
                        and key in self.inverse_transform_key
                    ):
                        inverse_info[key] = value[img_id]
            else:
                max_shape = None
                inverse_info = {}

            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                invasion_state_pred_list,
                beside_valid_conf_pred_list,
                invasion_scale_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_bboxes)
        if self.meta_data_bool:
            results["pred_bboxes"] = det_results
            results["img_name"] = meta_data["img_name"]
            results["img_id"] = meta_data["img_id"]
            return results
        else:
            results = []
            for det_bboxes in det_results:
                boxes = DetBoxes2D(
                    boxes=det_bboxes[:, :4],
                    scores=det_bboxes[:, -2],
                    cls_idxs=det_bboxes[:, -1],
                )
                results.append(boxes)

            return {"pred_boxes": results}

    def _check_shape(self, data, pred):
        flag = True
        for i in data:
            if len(i) != pred.shape[0]:
                flag = False
        return flag

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        centerness_pred_list,
        invasion_state_pred_list,
        beside_valid_conf_pred_list,
        invasion_scale_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        """Decode the output of a single picture into a prediction result.

        Args:
            cls_score_list: List of all levels' cls_score,
                each has shape (N, num_points * num_classes, H, W).
            bbox_pred_list: List of all levels' bbox_pred,
                each has shape (N, num_points * 4, H, W).
            centerness_pred_list: List of all levels'
                centerness_pred, each has shape (N, num_points * 1, H, W).
            mlvl_points: List of all levels' points.
            max_shape: Maximum allowable shape of the decoded bbox.

        Returns:
            det_bboxes (torch.Tensor): Decoded bbox, with shape (N, 6),
                represents x1, y1, x2, y2, cls_score, cls_id (0-based).
        """
        cfg = self.test_cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_invasion_state = []
        mlvl_left_beside_valid_conf = []
        mlvl_right_beside_valid_conf = []
        mlvl_invasion_scale = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for (
            cls_score,
            bbox_pred,
            centerness,
            invasion_state,
            beside_valid_conf,
            invasion_scale,
            points,
        ) in zip(
            cls_score_list,
            bbox_pred_list,
            centerness_pred_list,
            invasion_state_pred_list,
            beside_valid_conf_pred_list,
            invasion_scale_pred_list,
            mlvl_points,
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.num_classes)
                .sigmoid()
            )
            if has_centerness:
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            invasion_state = (
                invasion_state.permute(1, 2, 0).reshape(-1).sigmoid()
            )
            left_beside_valid_conf, right_beside_valid_conf = torch.split(
                beside_valid_conf, 1, dim=0
            )
            left_beside_valid_conf = (
                left_beside_valid_conf.permute(1, 2, 0).reshape(-1).sigmoid()
            )
            right_beside_valid_conf = (
                right_beside_valid_conf.permute(1, 2, 0).reshape(-1).sigmoid()
            )
            invasion_scale = invasion_scale.permute(1, 2, 0).reshape(-1, 2)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                invasion_state = invasion_state[topk_inds, :]
                left_beside_valid_conf = left_beside_valid_conf[topk_inds, :]
                right_beside_valid_conf = right_beside_valid_conf[topk_inds, :]
                invasion_scale = invasion_scale[topk_inds, :]
                if centerness is not None:
                    centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=max_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_invasion_state.append(invasion_state)
            mlvl_left_beside_valid_conf.append(left_beside_valid_conf)
            mlvl_right_beside_valid_conf.append(right_beside_valid_conf)
            mlvl_invasion_scale.append(invasion_scale)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_invasion_scale = torch.cat(mlvl_invasion_scale)
        if self.input_resize_scale is not None:
            mlvl_bboxes[:, :5] /= self.input_resize_scale
        # inverse transform for mapping to the original image
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
                    mlvl_invasion_scale = transform.inverse_transform(
                        inputs=mlvl_invasion_scale,
                        task_type="cone_invasion",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels to [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if has_centerness:
            mlvl_centerness = torch.cat(mlvl_centerness)
        else:
            mlvl_centerness = None  # score_factors wile be set as None
        mlvl_left_beside_valid_conf = torch.cat(mlvl_left_beside_valid_conf)
        mlvl_right_beside_valid_conf = torch.cat(mlvl_right_beside_valid_conf)
        mlvl_invasion_state = torch.cat(mlvl_invasion_state)
        score_thr = cfg.get("score_thr", "0.05")
        nms = cfg.get("nms").get("name", "nms")
        iou_threshold = cfg.get("nms").get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_with_ps_nms(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_invasion_state,
            mlvl_left_beside_valid_conf,
            mlvl_right_beside_valid_conf,
            mlvl_invasion_scale,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
            label_offset=self.label_offset,
        )
        return det_bboxes


@OBJECT_REGISTRY.register
class FCOSDecoder4RCNN(FCOSDecoder):
    """Decoder for FCOS+RCNN Architecture.

    Args:
        num_classes: Number of categories excluding the background category.
        strides: A list contains the strides of fcos_head output.
        input_shape: The shape of input_image.
        nms_use_centerness: If True, use centerness as a
            factor in nms post-processing.
        nms_sqrt: If True, sqrt(score_thr * score_factors).
        rescale: Whether to map the prediction result to the
            orig img.
        test_cfg: Cfg dict, including some configurations of
            nms.
        input_resize_scale: The scale to resize bbox.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int],
        input_shape: Tuple[int],
        nms_use_centerness: bool = True,
        nms_sqrt: bool = True,
        test_cfg: Optional[Dict] = None,
        input_resize_scale: Optional[Union[float, torch.Tensor]] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            strides=strides,
            nms_use_centerness=nms_use_centerness,
            nms_sqrt=nms_sqrt,
            test_cfg=test_cfg,
            input_resize_scale=input_resize_scale,
        )

        self.input_shape = input_shape
        self.post_nms_top_k = test_cfg.get("nms_post_nms_top_k", 15)
        self.nms_padding_mode = test_cfg.get("nms_padding_mode", None)

    def pad_data(self, data):
        if data.numel() == 0:
            return data.new_zeros((self.post_nms_top_k, data.shape[1]))
        else:
            if self.nms_padding_mode == "rollover":
                data = torch.cat([data] * (self.post_nms_top_k // len(data)))
                num_padded = self.post_nms_top_k - data.shape[0]
                return torch.cat([data, data[:num_padded]])
            elif self.nms_padding_mode == "pad_zero":
                num_padded = self.post_nms_top_k - data.shape[0]
                return torch.cat(
                    [data, data.new_zeros((num_padded, *data.shape[1:]))]
                )

    def forward(self, pred: OrderedDict):
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            if len(pred) == 3:
                cls_scores, bbox_preds, centernesses = pred.values()
            else:
                cls_scores, bbox_preds, centernesses = (
                    pred["cls"],
                    pred["offset_2d_reg"],
                    pred["ctrness_2d_reg"],
                )
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, centernesses = pred

        # when training multitask bbox_preds should mul strides
        if self.training:
            bbox_preds = [
                bbox_preds[i] * stride for i, stride in enumerate(self.strides)
            ]

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        inverse_info = {}
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # [cxh1xw1, cxh2xw2, cxh3xw3, ...]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [4xh1xw1, 4xh2xw2, 4xh3xw3, ...]
            if centernesses is not None:
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            else:
                centerness_pred_list = [None] * len(cls_score_list)

            max_shape = self.input_shape
            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            data = det_bboxes[:, :4]
            if self.nms_padding_mode is not None:
                data = self.pad_data(data)
            det_results.append(data)

        results["pred_bboxes"] = det_results

        return results


@require_packages("torchvision")
def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms,
    iou_threshold,
    max_per_img=-1,
    score_factors=None,
    nms_sqrt=False,
    filter_score_mul_centerness=False,
    label_offset=0,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (str): nms type, candidate values are ['nms', 'soft_nms'].
        iou_threshold (float): NMS IoU threshold
        max_per_img (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        nms_sqrt (bool): If True, sqrt(score_thr * score_factors)
    Return:
        torch.cat((dets, labels[keep].view(-1, 1).float()), dim=-1):
            shape (n_nms, #class*4 + 2) or (n_nms, 4 + 2)
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4
        )  # NC4
    scores = multi_scores[:, :-1]  # NC
    if score_factors is not None:
        if not nms_sqrt:
            scores_new = scores * score_factors[:, None]
        else:
            scores_new = torch.sqrt(scores * score_factors[:, None])
    # filter out boxes with low scores
    if filter_score_mul_centerness and score_factors is not None:
        valid_mask = scores_new > score_thr
        scores = scores_new
    elif score_factors is not None:
        valid_mask = scores > score_thr
        scores = scores_new
    else:
        valid_mask = scores > score_thr
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1),
    ).view(-1, 4)
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1] + label_offset

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )

        return torch.cat((bboxes, labels), dim=-1)

    if nms == "nms":
        keep = batched_nms(bboxes, scores, labels, iou_threshold)
    else:
        raise NotImplementedError

    dets = torch.cat([bboxes[keep], scores[keep][:, None]], -1)

    if max_per_img > 0:
        dets = dets[:max_per_img]
        keep = keep[:max_per_img]

    return torch.cat((dets, labels[keep].view(-1, 1).float()), dim=-1)


def multiclass_with_ps_nms(
    multi_bboxes: Tensor,
    multi_scores: Tensor,
    mlvl_invasion_state: Tensor,
    mlvl_left_beside_valid_conf: Tensor,
    mlvl_right_beside_valid_conf: Tensor,
    multi_invasion_scale: Tensor,
    score_thr: float,
    nms: str,
    iou_threshold: float,
    max_per_img: int = -1,
    score_factors: Tensor = None,
    nms_sqrt: bool = False,
    filter_score_mul_centerness: bool = False,
    label_offset: int = 0,
):
    """NMS for multi-class bboxes with cone invasion.

    Args:
        multi_bboxes: shape (n, #class*4) or (n, 4)
        multi_scores: shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr: bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms: nms type, candidate values are ['nms', 'soft_nms'].
        iou_threshold: NMS IoU threshold
        max_per_img: if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors: The factors multiplied to scores before
            applying NMS
        nms_sqrt: If True, sqrt(score_thr * score_factors)
    Return:
        torch.cat((dets, labels[keep].view(-1, 1).float()), dim=-1):
            shape (n_nms, #class*4 + 2) or (n_nms, 4 + 2)
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4
        )  # NC4
    scores = multi_scores[:, :-1]  # NC
    if score_factors is not None:
        if not nms_sqrt:
            scores_new = scores * score_factors[:, None]
        else:
            scores_new = torch.sqrt(scores * score_factors[:, None])
    # filter out boxes with low scores
    if filter_score_mul_centerness and score_factors is not None:
        valid_mask = scores_new > score_thr
        scores = scores_new
    elif score_factors is not None:
        valid_mask = scores > score_thr
        scores = scores_new
    else:
        valid_mask = scores > score_thr

    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1),
    ).view(-1, 4)
    scores = torch.masked_select(scores, valid_mask)
    invasion_state = torch.masked_select(
        mlvl_invasion_state[:, None], valid_mask
    )
    l_beside_valid = torch.masked_select(
        mlvl_left_beside_valid_conf[:, None], valid_mask
    )
    r_beside_valid = torch.masked_select(
        mlvl_right_beside_valid_conf[:, None], valid_mask
    )
    invasion_scale = torch.masked_select(
        multi_invasion_scale, torch.cat((valid_mask, valid_mask), -1)
    ).view(-1, 2)
    labels = valid_mask.nonzero()[:, 1] + label_offset

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)
        invasion_state = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)
        l_beside_valid = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)
        r_beside_valid = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)
        invasion_scale = multi_bboxes.new_zeros((0, 2), dtype=torch.float32)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )

        return torch.cat(
            (
                bboxes,
                labels,
                invasion_state,
                l_beside_valid,
                r_beside_valid,
                invasion_scale,
            ),
            dim=-1,
        )

    if nms == "nms":
        keep = batched_nms(bboxes, scores, labels, iou_threshold)
    else:
        raise NotImplementedError

    dets = torch.cat([bboxes[keep], scores[keep][:, None]], -1)

    if max_per_img > 0:
        dets = dets[:max_per_img]
        keep = keep[:max_per_img]

    return torch.cat(
        (
            dets,
            labels[keep].view(-1, 1).float(),
            invasion_state[keep].view(-1, 1),
            l_beside_valid[keep].view(-1, 1),
            r_beside_valid[keep].view(-1, 1),
            invasion_scale[keep].view(-1, 2),
        ),
        dim=-1,
    )


@OBJECT_REGISTRY.register
class VehicleSideFCOSDecoder(PostProcessorBase):  # noqa: D205,D400
    """

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        strides (Sequence[int]): A list contains the strides of fcos_head
            output.
        transforms (Sequence[dict]): A list contains the transform config.
        inverse_transform_key (Sequence[str]): A list contains the inverse
            transform info key.
        nms_use_centerness (bool, optional): If True, use centerness as a
            factor in nms post-processing.
        nms_sqrt (bool, optional): If True, sqrt(score_thr * score_factors).
        test_cfg (dict, optional): Cfg dict, including some configurations of
            nms.
        truncate_bbox (bool, optional): If True, truncate the predictive bbox
            out of image boundary. Default True.
        filter_score_mul_centerness (bool, optional): If True, filter out bbox
            by score multiply centerness, else filter out bbox by score.
            Default False.
    """

    def __init__(
        self,
        num_classes,
        strides,
        transforms=None,
        inverse_transform_key=None,
        nms_use_centerness=True,
        nms_sqrt=True,
        test_cfg=None,
        input_resize_scale=None,
        truncate_bbox=True,
        filter_score_mul_centerness=False,
        int8_output=True,
        decouple_h=False,
    ):
        super(VehicleSideFCOSDecoder, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.transforms = transforms
        self.inverse_transform_key = inverse_transform_key
        self.nms_use_centerness = nms_use_centerness
        self.nms_sqrt = nms_sqrt
        self.test_cfg = test_cfg
        self.input_resize_scale = input_resize_scale
        if self.input_resize_scale is not None:
            assert self.input_resize_scale > 0
        self.truncate_bbox = truncate_bbox
        self.filter_score_mul_centerness = filter_score_mul_centerness
        self.int8_output = int8_output
        self.decouple_h = decouple_h

    def forward(self, pred: Sequence[torch.Tensor], meta_data: Dict[str, Any]):
        assert len(pred) == 4, (
            "pred must be a tuple containing cls_scores,"
            "bbox_preds, alpha_preds and centernesses"
        )
        cls_scores, bbox_preds, alpha_preds, centernesses = pred
        if not self.int8_output:
            bbox_preds = list(
                map(lambda x: torch.nn.functional.relu(x), bbox_preds)
            )
            alpha_preds = list(
                map(lambda x: torch.nn.functional.tanh(x), alpha_preds)
            )
        assert len(cls_scores) == len(bbox_preds) == len(alpha_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # [cxh1xw1, cxh2xw2, cxh3xw3, ...]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [4xh1xw1, 4xh2xw2, 4xh3xw3, ...]
            alpha_pred_list = [
                alpha_preds[i][img_id].detach() for i in range(num_levels)
            ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            if centernesses is not None:
                centerness_pred_list = [
                    centernesses[i][img_id].detach() for i in range(num_levels)
                ]  # [1xh1xw1, 1xh2xw2, 1xh3xw3, ...]
            else:
                centerness_pred_list = [None] * len(cls_score_list)

            h_index = meta_data["layout"][img_id].index("h")
            w_index = meta_data["layout"][img_id].index("w")
            if "pad_shape" in meta_data:
                max_shape = (
                    meta_data["pad_shape"][img_id][h_index],
                    meta_data["pad_shape"][img_id][w_index],
                )
            elif "img_shape" in meta_data:
                max_shape = (
                    meta_data["img_shape"][img_id][h_index],
                    meta_data["img_shape"][img_id][w_index],
                )
            else:
                max_shape = (
                    meta_data["img_height"][img_id],
                    meta_data["img_width"][img_id],
                )
            max_shape = max_shape if self.truncate_bbox else None
            inverse_info = {}
            for key, value in meta_data.items():
                if (
                    self.inverse_transform_key
                    and key in self.inverse_transform_key
                ):
                    inverse_info[key] = value[img_id]
            det_polygons = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                alpha_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_polygons)

        results["pred_polygons"] = det_results
        results["img_name"] = meta_data["img_name"]
        results["img_id"] = meta_data["img_id"]
        return results

    def _distancealpha2bbox(self, points, distance, tanalpha, max_shape=None):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        if self.decouple_h:
            y2 = points[..., 1] + distance[..., 3]
        else:
            y2_temp = points[..., 1] + distance[..., 3]
            delta_left = tanalpha[..., 0] * distance[..., 0]
            delta_right = tanalpha[..., 0] * distance[..., 2]  #
            y2 = y2_temp + (delta_right - delta_left) / 2
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return torch.stack([x1, y1, x2, y2], -1)

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        alpha_pred_list,
        centerness_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        """Decode the output of a single picture into a prediction result.

        Args:
            cls_score_list (list[torch.Tensor]): List of all levels' cls_score,
                each has shape (N, num_points * num_classes, H, W).
            bbox_pred_list (list[torch.Tensor]): List of all levels' bbox_pred,
                each has shape (N, num_points * 4, H, W).
            centerness_pred_list (list[torch.Tensor]): List of all levels'
                centerness_pred, each has shape (N, num_points * 1, H, W).
            mlvl_points (list[torch.Tensor]): List of all levels' points.
            max_shape (Sequence): Maximum allowable shape of the decoded bbox.

        Returns:
            det_bboxes (torch.Tensor): Decoded bbox, with shape (N, 6),
                represents x1, y1, x2, y2, cls_score, cls_id (0-based).
        """
        cfg = self.test_cfg
        assert (
            len(cls_score_list)
            == len(bbox_pred_list)
            == len(alpha_pred_list)
            == len(mlvl_points)
        )  # noqa
        mlvl_bboxes = []
        mlvl_tanalphas = []
        mlvl_scores = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for cls_score, bbox_pred, tanalpha_pred, centerness, points in zip(
            cls_score_list,
            bbox_pred_list,
            alpha_pred_list,
            centerness_pred_list,
            mlvl_points,
        ):
            assert (
                cls_score.size()[-2:]
                == bbox_pred.size()[-2:]
                == tanalpha_pred.size()[-2:]
            )
            tanalpha_pred.mul_(PI / 2)
            tanalpha_pred.tan_()
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.num_classes)
                .sigmoid()
            )
            if has_centerness:
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            tanalpha_pred = tanalpha_pred.permute(1, 2, 0).reshape(-1, 1)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                tanalpha_pred = tanalpha_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                if centerness is not None:
                    centerness = centerness[topk_inds]
            bboxes = self._distancealpha2bbox(
                points, bbox_pred, tanalpha_pred, max_shape=None
            )
            mlvl_bboxes.append(bboxes)
            mlvl_tanalphas.append(tanalpha_pred)
            mlvl_scores.append(scores)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_tanalphas = torch.cat(mlvl_tanalphas)
        if self.input_resize_scale is not None:
            mlvl_bboxes[:, :5] /= self.input_resize_scale
        # inverse transform for mapping to the original image
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels to [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_bboxes_tanalphas = torch.cat([mlvl_bboxes, mlvl_tanalphas], -1)
        if has_centerness:
            mlvl_centerness = torch.cat(mlvl_centerness)
        else:
            mlvl_centerness = None  # score_factors wile be set as None
        score_thr = cfg.get("score_thr", "0.05")
        nms = cfg.get("nms").get("name", "nms")
        iou_threshold = cfg.get("nms").get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
        )
        det_polygons = self._get_det_polygon(det_bboxes, mlvl_bboxes_tanalphas)
        return det_polygons

    def _get_det_polygon(
        self, det_bboxes: torch.Tensor, mlvl_bboxes_tanalphas: torch.Tensor
    ):
        if det_bboxes.size()[-1] == 0:
            return det_bboxes
        det_polygons = det_bboxes.new_empty((det_bboxes.size(0), 10))
        for i in range(det_bboxes.size(0)):
            det_bbox = det_bboxes[i]
            where_is = (
                mlvl_bboxes_tanalphas[..., :4] - det_bbox[..., :4]
            ).abs().sum(-1) < 1e-4
            assert (
                where_is.sum() >= 1
            ), f"{det_bbox}\n{mlvl_bboxes_tanalphas[where_is]}"  # 有能够匹配上
            if where_is.sum() > 1:
                print(
                    "Warning: got same pred bbox on different level features"
                )
                print(f"{det_bbox}\n{mlvl_bboxes_tanalphas[where_is]}")
            bbox_tanalpha = mlvl_bboxes_tanalphas[where_is][0]  # shape: 5
            delta_y = (
                (bbox_tanalpha[2] - bbox_tanalpha[0]) / 2 * bbox_tanalpha[-1]
            )

            det_polygons[i][0] = det_bbox[0]
            det_polygons[i][1] = det_bbox[1]

            det_polygons[i][2] = det_bbox[2]
            det_polygons[i][3] = det_bbox[1]

            det_polygons[i][4] = det_bbox[2]
            det_polygons[i][5] = det_bbox[3] + delta_y

            det_polygons[i][6] = det_bbox[0]
            det_polygons[i][7] = det_bbox[3] - delta_y

            det_polygons[i][8] = det_bbox[4]

            det_polygons[i][9] = det_bbox[5]
        return det_polygons


@OBJECT_REGISTRY.register
class FCOSDecoderForFilter(FCOSDecoder):
    """
    The basic structure of FCOSDecoderForFilter.

    Args:
        kwargs: Same as FCOSDecoder.
    """

    def __init__(self, **kwargs):
        super(FCOSDecoderForFilter, self).__init__(**kwargs)

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        centerness_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        cfg = self.test_cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
            cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points
        ):
            scores = cls_score.sigmoid()
            if has_centerness:
                centerness = centerness.sigmoid()

            nms_pre = cfg.get("nms_pre", -1)
            centerness = centerness.view(-1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                if centerness is not None:
                    centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=max_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.input_resize_scale is not None:
            mlvl_bboxes[:, :5] /= self.input_resize_scale
        # inverse transform for mapping to the original image
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels to [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if has_centerness:
            mlvl_centerness = torch.cat(mlvl_centerness)
        else:
            mlvl_centerness = None  # score_factors wile be set as None
        score_thr = cfg.get("score_thr", "0.05")
        nms = cfg.get("nms").get("name", "nms")
        iou_threshold = cfg.get("nms").get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
            label_offset=self.label_offset,
        )
        return det_bboxes

    def forward(self, preds, meta_data):
        bs = len(preds[0])
        results = {}
        det_results = []
        for img_id in range(bs):
            cls_score_list = []
            bbox_pred_list = []
            centerness_pred_list = []
            mlvl_points = []

            for level_id, pred_level in enumerate(preds):
                (
                    _,
                    _,
                    per_img_coord,
                    per_img_score,
                    per_img_bbox_pred,
                    per_img_centerness,
                ) = pred_level[img_id]
                cls_score_list.append(per_img_score)

                if self.bbox_relu:
                    per_img_bbox_pred = torch.nn.functional.relu(
                        per_img_bbox_pred
                    )
                if self.upscale_bbox_pred:
                    per_img_bbox_pred = (
                        per_img_bbox_pred * self.strides[level_id]
                    )
                bbox_pred_list.append(per_img_bbox_pred)

                centerness_pred_list.append(per_img_centerness)
                per_img_coord = torch.stack(
                    [per_img_coord[..., 1], per_img_coord[..., 0]], dim=-1
                )
                per_img_coord = (
                    per_img_coord * self.strides[level_id]
                    + self.strides[level_id] // 2
                )
                mlvl_points.append(per_img_coord)

            h_index = meta_data["layout"][img_id].index("h")
            w_index = meta_data["layout"][img_id].index("w")
            if "pad_shape" in meta_data:
                max_shape = (
                    meta_data["pad_shape"][img_id][h_index],
                    meta_data["pad_shape"][img_id][w_index],
                )
            elif "img_shape" in meta_data:
                max_shape = (
                    meta_data["img_shape"][img_id][h_index],
                    meta_data["img_shape"][img_id][w_index],
                )
            else:
                max_shape = (
                    meta_data["img_height"][img_id],
                    meta_data["img_width"][img_id],
                )
            max_shape = max_shape if self.truncate_bbox else None
            inverse_info = {}
            for key, value in meta_data.items():
                if (
                    self.inverse_transform_key
                    and key in self.inverse_transform_key
                ):
                    inverse_info[key] = value[img_id]
            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_bboxes)
        results["pred_bboxes"] = det_results
        results["img_name"] = meta_data["img_name"]
        results["img_id"] = meta_data["img_id"]
        return results


@OBJECT_REGISTRY.register
class FCOSDecoderForFilterHbir(FCOSDecoderForFilter):
    """
    The basic structure of FCOSDecoderForFilterHbir.

    Args:
        kwargs: Same as FCOSDecoder.
    """

    def __init__(self, **kwargs):
        super(FCOSDecoderForFilterHbir, self).__init__(**kwargs)

    def forward(self, outputs, meta_data):

        preds = []

        for out in outputs:
            max_vlaue = out[0]
            max_id = out[1]
            coords = out[2]
            cls = out[3][..., : self.num_classes]
            bbox = out[4][..., :4]
            center = out[5][..., :1]
            pred = [[max_vlaue, max_id, coords, cls, bbox, center]]
            preds.append(pred)
        results = super().forward(preds, meta_data)
        return results
