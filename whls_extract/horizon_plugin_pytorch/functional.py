from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.overrides import handle_torch_function, has_torch_function_unary

from horizon_plugin_pytorch._compat import node_get
from horizon_plugin_pytorch.fx import fx_helper
from . import nn
from .dtype import qint8, qint32
from .march import March, with_march
from .qtensor import QTensor
from .utils.auto_cast import handle_autocast
from .utils.script_helper import get_script_subgraph
from .utils.typeguard import typechecked

__all__ = [
    "max",
    "argmax",
    "round",
    "sort",
    "set_annotation",
    "get_output_annotation",
    "nms",
    "nms_rotated",
    "batched_nms",
    "box3d_iou_bev",
    "box3d_overlap_bev",
    "box3d_iou",
    "nms3d",
    "nms3d_normal",
    "multi_tensor_legacynadamex",
    "batched_nms_with_padding",
    "om_ogc",
    "om_extract",
    "om_confusion_matrix",
    "om_chamfer_distance",
    "om_pt2lineseg",
    "bgr2centered_gray",
    "bgr2centered_yuv",
    "bgr2gray",
    "bgr2rgb",
    "bgr2yuv",
    "rgb2bgr",
    "rgb2centered_gray",
    "rgb2centered_yuv",
    "rgb2gray",
    "rgb2yuv",
    "centered_yuv2bgr",
    "centered_yuv2rgb",
    "raycast",
    "abs",
    "quantized_conv2d",
]


def max(
    input: Union[Tensor, QTensor], dim: int = 1, keepdim: bool = True
) -> Union[Tuple[Tensor, Tensor], Tuple[QTensor, Tensor]]:
    """
    Please refer to torch.max for detailed info.

    Args:
        input (Union[Tensor, QTensor]): The input tensor in NCHW format.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.

    Returns:
        Union[Tuple[Tensor, Tensor], Tuple[QTensor, Tensor]]:
            value: Max values in the shape of (N, 1 / None, H, W).
            idx: Index of max values in its own group
                in the shape of (N, 1 / None, H, W)
    """
    return input.max(dim, keepdim)


def argmax(
    input: Union[Tensor, QTensor], dim: int = 1, keepdim: bool = True
) -> Tensor:
    """
    Please refer to torch.argmax for detailed info.

    Args:
        input (Union[Tensor, QTensor]): The input tensor in NCHW format.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.

    Returns:
        Tensor: Index of max values in its own group in the shape of
            (N, 1 / None, H, W)
    """
    return input.argmax(dim, keepdim)


@torch.jit.script
def _stable_sort(
    input: Tensor, dim: int = -1, descending: bool = False
) -> Tuple[Tensor, Tensor]:
    if input.numel() == 0 or input.dim() == 0:
        return torch.sort(input, dim, descending)
    return torch.ops.horizon.sort(input, dim, descending)


@torch.jit.script
def sort(
    input: Tensor,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
):
    """Please refer to torch.sort for detailed info.

    Args:
        input (Tensor): the input tensor.
        dim (int, optional): the dimension to sort along. Defaults to -1.
        descending (bool, optional): controls the sorting order (ascending or
        descending). Defaults to False.
        stable (bool, optional):  makes the sorting routine stable, which
        guarantees that the order of equivalent elements is preserved.
        Defaults to False.

    Returns:
        tuple: A namedtuple of (values, indices) is returned, where the values
        are the sorted values and indices are the indices of the elements in
        the original input tensor.
    """
    # stable_sort is fully supported after torch 1.10
    return torch.sort(input, dim=dim, descending=descending, stable=stable)


@torch.jit.script
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Perform non-maximum suppression.

    Perform non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with
        IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been
        kept by NMS, sorted in decreasing order of scores
    """
    return torch.ops.horizon.nms(boxes, scores, iou_threshold)


def nms_rotated(
    dets: Tensor,
    scores: Tensor,
    iou_threshold: float,
    labels: Optional[Tensor] = None,
    clockwise: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Perform non-maximum suppression (NMS) on the rotated boxes.

    Performs non-maximum suppression (NMS) on the rotated boxes according to
    their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Args:
        dets (torch.Tensor):  Rotated boxes in shape (N, 5).
            They are expected to be in
            (x_ctr, y_ctr, width, height, angle_radian) format.
        scores (torch.Tensor): scores in shape (N, ).
        iou_threshold (float): IoU thresh for NMS.
        labels (torch.Tensor, optional): boxes' label in shape (N,).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the
        same data type as the input.
    """
    if dets.shape[0] == 0:
        return dets, None
    if not clockwise:
        flip_mat = dets.new_ones(dets.shape[-1])
        flip_mat[-1] = -1
        dets_cw = dets * flip_mat
    else:
        dets_cw = dets

    multi_label = labels is not None
    if multi_label:
        dets_wl = torch.cat((dets_cw, labels.unsqueeze(1)), 1)
    else:
        dets_wl = dets_cw
    _, order = sort(scores, 0, descending=True, stable=True)
    dets_sorted = dets_wl.index_select(0, order)

    keep_inds = torch.ops.horizon.nms_rotated(
        dets_wl,
        scores,
        order,
        dets_sorted,
        iou_threshold,
        multi_label,
    )
    dets = torch.cat(
        (dets[keep_inds], scores[keep_inds].reshape(-1, 1)), dim=1
    )
    return dets, keep_inds


@torch.jit.script
def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Perform non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float):
            discards all overlapping boxes with IoU > iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of
            the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep


@handle_autocast(
    supported_cpu_dtype=(torch.float16,),
    supported_gpu_dtype=(torch.float16,),
)
def round(data: Tensor) -> Tensor:
    """Round that round middle value away from zero.

    This behaviour is same as std::round.
    """
    return torch.ops.horizon.round(data)


@torch.jit.script
def _set_annotation(data: Tensor, annotation: str) -> Tensor:
    """Set tensor annotation internal."""
    return data


@fx_helper.wrap()
def set_annotation(data: Tensor, annotation: str) -> Tensor:
    """Set tensor annotation."""
    r = _set_annotation(data, annotation)
    r.annotation = annotation
    return r


def get_output_annotation(script_model):
    """Get output annotation from scripted model.

    the output of sctipt_model must be tensor or tuple of tensor or
    tuple of tuple of tensor.
    """
    assert isinstance(
        script_model, torch.jit.ScriptModule
    ), "please input script model use jit.trace or jit.script"
    anno_list = []
    passed_node = set()
    for out in script_model.graph.outputs():
        node = out.node()
        # tuple or function or forward
        if (
            node.kind() == "prim::TupleConstruct"
            or node.kind() == "prim::CallFunction"
            or node.kind() == "prim::CallMethod"
        ):
            anno_list.extend(
                _get_node_annotation(out, script_model, passed_node)
            )
        else:
            anno_list.append(None)
    return anno_list


def _get_node_annotation(node, model, passed_node):
    anno_list = []
    if node.debugName() in passed_node:
        return anno_list
    passed_node.add(node.debugName())
    node = node.node()
    if (
        node.kind() == "prim::TupleConstruct"
        or node.kind() == "prim::TupleUnpack"
    ):
        # tuple
        for i in node.inputs():
            anno_list.extend(_get_node_annotation(i, model, passed_node))
    elif node.kind() == "prim::CallMethod":
        # submodule forward
        subgraph = get_script_subgraph(model, node)
        if subgraph is not None:
            anno_list.extend(get_output_annotation(subgraph))
        else:
            anno_list.extend(_get_node_annotation(node, model, passed_node))
    elif node.kind() == "prim::CallFunction":
        # function of _set_annotation
        inputs = list(node.inputs())
        if (
            inputs[0].type().kind() == "FunctionType"
            and inputs[0].node().kind() == "prim::Constant"
            and inputs[0].node().hasAttribute("name")
            # and inputs[0].node()["name"] == "_set_annotation"
            and node_get(inputs[0].node(), "name") == "_set_annotation"
        ):
            # anno_list.append(inputs[2].node()["value"])
            anno_list.append(node_get(inputs[2].node(), "value"))
    else:
        anno_list.append(None)
    return anno_list


def box3d_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """Calculate 3d boxes IoU based on bev overlap.

    Args:
        boxes_a (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading].
        boxes_b (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading].

    Returns:
        ans_iou: (N, M) torch.Tensor object.
    """
    ans_iou = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )

    torch.ops.horizon.box3d_iou_bev(boxes_a, boxes_b, ans_iou)

    return ans_iou


def box3d_overlap_bev(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Calculate 3d boxes overlap under BEV view.

    This is a direct function call to CUDA overlap function.

    Args:
        boxes_a (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_overlap: (N, M) torch.Tensor object, where
        ans_overlap[i, j] = overlap(boxes_a[i], boxes_b[j]).
    """
    ans_overlap = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )

    torch.ops.horizon.box3d_overlap_bev(boxes_a, boxes_b, ans_overlap)

    return ans_overlap


def box3d_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """Calculate 3d boxes IoU based on 3d volumetric overlap.

    Args:
        boxes_a: (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M) torch.Tensor object.
    """

    # transform back to pcdet's coordinate
    boxes_a = boxes_a[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes_a[:, -1] = -boxes_a[:, -1] - np.pi / 2
    boxes_b = boxes_b[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes_b[:, -1] = -boxes_b[:, -1] - np.pi / 2

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )
    torch.ops.horizon.box3d_overlap_bev(boxes_a, boxes_b, overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms3d(boxes: torch.Tensor, scores: torch.Tensor, thresh: float, **kwargs):
    """Perform 3d bounding box non-max suppression.

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading] 3d boxes.
        scores: (N) confidence of each box.
        thresh: nms overlap threshold.

    Return:
        Indices of boxes that survived the selection.
    """
    out = torch.ops.horizon.nms3d(boxes, scores, thresh)
    return out


def nms3d_normal(boxes, scores, thresh, **kwargs):
    """Perform 3d bounding box non-max suppression. Boxes are not rotated.

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading] 3d boxes.
        scores: (N) confidence of each box.
        thresh: nms overlap threshold.

    Return:
        Indices of boxes that survived the selection.
    """
    out = torch.ops.horizon.nms3d_normal(boxes, scores, thresh)
    return out


def multi_tensor_legacynadamex(
    tensor_lists,
    step_list,
    lr,
    weight_decay,
    beta1,
    beta2,
    eps,
    schedule_decay,
    m_schedule,
    rescale_grad,
):
    """Perform fused LegacyNadamEx optimizer function.

    Args:
        tensor_lists: format is [param_list, grad_list, mean_list, var_list]
                      and each list must have same number of tensors.
                      Tensors will be modified same as origin optimizer.
        step_list: a list of step w.r.t each param.
        lr, weight_decay and all others is same as LegacyNadamEx optimizer.
    Return:
        m_schedule that has been updated. Must be saved for next step.
    """
    return torch.ops.horizon.multi_tensor_legacynadamex(
        tensor_lists,
        step_list,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
        schedule_decay,
        m_schedule,
        rescale_grad,
    )


def batched_nms_with_padding(
    boxes: Tensor,
    scores: Tensor,
    class_idxs: Optional[Tensor],
    iou_threshold: float,
    pre_top_n: int,
    post_top_n: int,
    legacy_bbox: bool,
    pad_mode: str = "pad_zero",
):
    """Batched Non-Maximum Supression.

    Output the index of preserved post_top_n boxes.
    Insufficient output will be padded to target number.

    Args:
        boxes (Tensor[N, box_num, 4]): Boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N, box_num]): Scores for each one of the boxes.
        class_idxs (Optional[Tensor[N, box_num]]):
            indices of the categories for each one of the boxes.
        iou_threshold (float):
            Discards all overlapping boxes with IoU > iou_threshold.
        pre_top_n (int): The top n bbox to apply nms on.
        post_top_n (int): The top n bbox to keep after nms.
        legacy_bbox (bool):
            Whether to add 1 when computing bounding box border.
        pad_mode (str, optional):
            The way to pad bbox to match the number of post_top_n.
            Defaults to "pad_zero".

    Returns:
        Tensor[N, box_num]: Preserved box index padded to target number.
    """
    if class_idxs is None:
        multi_class = False
        class_idxs = scores
    else:
        multi_class = True

    return torch.ops.horizon.batched_nms(
        boxes,
        scores,
        class_idxs,
        iou_threshold,
        pre_top_n,
        post_top_n,
        multi_class,
        legacy_bbox,
        pad_mode,
    )


def om_ogc(
    pred_cls: Tensor,
    pred_prob: Tensor,
    pred_r: Tensor,
    pred_sin: Tensor,
    pred_cos: Tensor,
    pred_embedding: Tensor,
    cls_num: int,
    cls_thr: float = 0.5,
    radius_l: int = 9,
    radius_t: int = 2,
    min_num: int = 1,
    pose_weight: float = 0.1,
    cluster_thr: float = 0.95,
    merge: bool = False,
    split_channel: bool = False,
) -> Tensor:
    """
    OGC (Offset Growth Cluster) for Online Mapping Post Process.

    Notice: Generally, the channel C means set num, the default value is 2

    Args:
        pred_cls: tensor [C, H, W], class label
        pred_prob: tensor [C, H, W], class confidence
        pred_r: tensor [C, H, W], offset radius
        pred_sin: tensor [C, H, W], offset sin value
        pred_cos: tensor [C, H, W], offset cos value
        pred_embedding: tensor [C, H, W, D], embedding features for instance
        cls_num: int, number of classes
        cls_thr: float, used to select valid pixels from pred_probs
        radius_l: int, the radius of longitudinal searching
        radius_t: int, the radius of transverse searching
        min_num: int, the minimum number of clustering points
        pose_weights: float, the weight of computing pose diversity
        cluster_thr: float, the threshold of point similarly
        merge: bool, if merge clusters
        split_channel: bool, if running cluster in different channel
    Returns:
        cluster_result: tensor [C, H, W], clsuter ids
    """
    return torch.ops.horizon.om_ogc(
        pred_cls.contiguous(),
        pred_prob.contiguous(),
        pred_r.contiguous(),
        pred_sin.contiguous(),
        pred_cos.contiguous(),
        pred_embedding.contiguous(),
        cls_num,
        cls_thr,
        radius_l,
        radius_t,
        min_num,
        pose_weight,
        cluster_thr,
        merge,
        split_channel,
    )


def om_extract(
    pred_cls: np.ndarray,
    pred_prob: np.ndarray,
    pred_offset: np.ndarray,
    pred_instance: np.ndarray,
    top: float,
    left: float,
    res_h: float,
    res_w: float,
    cls_thr: float = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract foreground pixels for Online Mapping Post Process.

    Args:
        pred_cls: numpy [C, H, W], class label
        pred_prob: numpy [C, H, W], class confidence
        pred_offset: numpy [C, H, W], position offset values
        pred_instance: numpy [C, H, W], instance id
        top: top range
        left: left range
        res_h: the resolution of height
        res_w: the resolution of wideth
        cls_thr: float, used to select valid pixels from pred_probs
    Returns:
        valid_pixels: numpy [N, 11]
            (instance_id, c, h, w, cls, prob, x_vcs, y_vcs,
             x_vcs_origin, y_vcs_origin, offset_dist)
        instance_info: numpy [M, 4]
            (instance_id, instance_num, cls, prob)

    """
    cls_tensor = torch.tensor(pred_cls, dtype=torch.int32)
    prob_tensor = torch.tensor(pred_prob, dtype=torch.float32)
    offset_tensor = torch.tensor(pred_offset, dtype=torch.float32)
    instance_tensor = torch.tensor(pred_instance, dtype=torch.int32)
    results = torch.ops.horizon.om_extract(
        cls_tensor.contiguous(),
        prob_tensor.contiguous(),
        offset_tensor.contiguous(),
        instance_tensor.contiguous(),
        top,
        left,
        res_h,
        res_w,
        cls_thr,
    )
    return results[0].numpy(), results[1].numpy()


def om_confusion_matrix(
    gt_cls: Tensor,
    pred_cls: Tensor,
    pred_prob: Tensor,
    cls_thr_list: Tensor,
    cls_num: int,
) -> Tensor:
    """
    Compute Confusion Matrix for Online Mapping Post Process.

    Args:
        gt_cls: tensor, ground truth class label
        pred_cls: tensor, prediction class label
        pred_prob: tensor, prediction class confidence
        cls_thr_list: tensor, confidence threshold list
        cls_num: int, number of classes

    Returns:
        result: tensor [len(cls_thr_list), cls_num, cls_num], confusion matrix
    """
    return torch.ops.horizon.om_confusion_matrix(
        gt_cls.contiguous(),
        pred_cls.contiguous(),
        pred_prob.contiguous(),
        cls_thr_list.contiguous(),
        cls_num,
    )


def om_chamfer_distance(
    data_x: Tensor,
    data_y: Tensor,
    direction: str = "bi",
    dist_thresh: float = 10.0,
) -> float:
    """
    Compute Chamfer Distance for online mapping.

    Args:
        data_x, cpu or gpu tensor, the first data
        data_y, cpu or gpu tensor, the second data
        direction, one of {"x_to_y", "y_to_x", "bi"}
        dist_thresh: default dist, if data is empty

    Returns:
        dist: float, chamfer distance value
    """
    if data_x.numel() == 0 or data_y.numel() == 0:
        if direction == "bi":
            return dist_thresh * 2.0
        else:
            return dist_thresh

    return torch.ops.horizon.om_chamfer_distance(
        data_x.contiguous(),
        data_y.contiguous(),
        direction,
    )


def om_pt2lineseg(
    pts: Tensor,
    lines: Tensor,
    mask: Optional[Tensor] = None,
    dist_thr: float = -1,
) -> List[Tensor]:
    """Compute the distance between points and line segments.

    Args:
        pts, cpu or gpu tensor, input points, the last dimension must be 2
        lines, cpu or gpu tensor, line segments, the last dimension must be 4
        mask, cpu or gpu tensor, indicates whether the point is valid
        dist_thr, float, if > 0, enable output select indexes

    Returns:
        dists: output distance between points and line segments
        directions: output directions
        select_indexes: output select indexes of points
                        which satisfying the distance requirement

    Example:
        pts: [3, 3, 2]
        lines: [8, 4]
        mask: [3, 3]
        dists: [3, 3, 8]
        directions: [3, 3, 8, 2]
        select_indexes: [3, 3, 8+1], thr fist value is the select num
    """
    pts_shape = pts.shape
    line_num = lines.shape[0]
    dists_shape = pts_shape[:-1] + (line_num,)
    direction_shape = dists_shape + (2,)
    dists = torch.zeros(dists_shape, dtype=torch.float32, device=pts.device)
    directions = torch.zeros(
        direction_shape, dtype=torch.float32, device=pts.device
    )
    select_shape = pts_shape[:-1] + (line_num + 1,)
    select_indexes = torch.zeros(
        select_shape, dtype=torch.int32, device=pts.device
    )
    if pts.numel() == 0 or lines.numel() == 0:
        return (dists, directions, select_indexes)

    if not isinstance(mask, Tensor):
        mask = torch.ones(pts_shape[:-1], dtype=torch.uint8, device=pts.device)
    else:
        assert mask.numel() == pts.numel() // 2, "mask size not equal pts num"

    torch.ops.horizon.om_pt2lineseg(
        pts, lines, mask, dists, directions, select_indexes, dist_thr
    )
    return (dists, directions, select_indexes)


def _reverse_channel(input: Tensor) -> Tensor:
    """Reverse channel orders of RGB/BGR images.

    Args:
        input: image with shape [N, C, H, W]

    Returns:
        Tensor: image with shape [N, C, H, W]
    """
    return input.flip(1)


@typechecked
def rgb2bgr(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from RGB format to BGR

    Args:
        input: image in RGB format with shape [N, 3, H, W]

    Returns:
        Tensor: image in BGR format with shape [N, 3, H, W]
    """
    return _reverse_channel(input)


@typechecked
def bgr2rgb(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from BGR format to RGB

    Args:
        input: image in BGR format with shape [N, 3, H, W]

    Returns:
        Tensor: image in RGB format with shape [N, 3, H, W]
    """
    return _reverse_channel(input)


def _convert_color(input: Tensor, weight: Tensor, offset: Tensor) -> Tensor:
    """Convert RGB to YUV BT.601.

    (https://en.wikipedia.org/wiki/YUV#Studio_swing_for_YCbCr_BT.601)
    """
    input = input.to(torch.float32)
    weight = weight.to(torch.float32).to(input.device)
    bias = (torch.ones(3) * 128).to(input.device)
    weight = weight.unsqueeze(2).unsqueeze(3)  # weight shape: [3,3,1,1]
    res = torch.nn.functional.conv2d(input, weight, bias) / 256
    offset = offset.to(torch.float32).to(input.device)
    offset = offset.reshape(1, 3, 1, 1)
    res += offset
    return res.to(torch.int32)


@typechecked
def bgr2yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from BGR format to YUV444 BT.601

    Args:
        input: input image in BGR format, ranging 0 to 255
        swing: "studio" for YUV studio swing (Y: 16 to 235, U, V: 16 to 240).
                "full" for YUV full swing (Y, U, V: 0 to 255).
                default is "studio"

    Returns:
        Tensor: YUV image
    """
    weight_map = {
        "studio": [[25, 129, 66], [112, -74, -38], [-18, -94, 112]],
        "full": [[29, 150, 77], [127, -84, -43], [-21, -106, 127]],
    }
    offset_map = {
        "studio": [16, 128, 128],
        "full": [0, 128, 128],
    }
    assert (
        swing in weight_map
    ), '`swing` is not valid! must be "full" or "studio"!'
    return _convert_color(
        input,
        torch.tensor(weight_map[swing]),
        torch.tensor(offset_map[swing]),
    )


@typechecked
def rgb2yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from RGB format to YUV444 BT.601

    Args:
        input: input image in RGB format, ranging 0 to 255
        swing: "studio" for YUV studio swing (Y: 16 to 235, U, V: 16 to 240).
                "full" for YUV full swing (Y, U, V: 0 to 255).
                default is "studio"

    Returns:
        Tensor: YUV image
    """
    weight_map = {
        "studio": [[66, 129, 25], [-38, -74, 112], [112, -94, -18]],
        "full": [[77, 150, 29], [-43, -84, 127], [127, -106, -21]],
    }
    offset_map = {
        "studio": [16, 128, 128],
        "full": [0, 128, 128],
    }
    assert (
        swing in weight_map
    ), '`swing` is not valid! must be "full" or "studio"!'
    return _convert_color(
        input,
        torch.tensor(weight_map[swing]),
        torch.tensor(offset_map[swing]),
    )


@typechecked
def bgr2centered_yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from BGR format to centered YUV444 BT.601

    Args:
        input: input image in BGR format, ranging 0 to 255
        swing: "studio" for YUV studio swing
                (Y: -112 to 107, U, V: -112 to 112).
                "full" for YUV full swing (Y, U, V: -128 to 127).
                default is "studio"

    Returns:
        Tensor: centered YUV image
    """
    return bgr2yuv(input, swing) - 128


@typechecked
def rgb2centered_yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from RGB format to centered YUV444 BT.601

    Args:
        input: input image in RGB format, ranging 0 to 255
        swing: "studio" for YUV studio swing
                (Y: -112 to 107, U, V: -112 to 112).
                "full" for YUV full swing (Y, U, V: -128 to 127).
                default is "studio"

    Returns:
        Tensor: centered YUV image
    """
    return rgb2yuv(input, swing) - 128


@typechecked
def bgr2gray(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from BGR format to gray

    Args:
        input: input image in BGR format of shape [N, 3, H, W],
               ranging 0 to 255

    Returns:
        Tensor: gray image of shape [N, 1, H, W],
        ranging 0 to 255
    """
    return bgr2yuv(input, "full")[:, :1]


@typechecked
def rgb2gray(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from RGB format to gray

    Args:
        input: input image in RGB format of shape [N, 3, H, W],
               ranging 0 to 255

    Returns:
        Tensor: gray image of shape [N, 1, H, W],
        ranging 0 to 255
    """
    return rgb2yuv(input, "full")[:, :1]


@typechecked
def bgr2centered_gray(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from BGR format to centered gray

    Args:
        input: input image in BGR format of shape [N, 3, H, W],
               ranging 0 to 255

    Returns:
        Tensor: centered gray image of shape [N, 1, H, W],
        ranging -128 to 127
    """
    return bgr2gray(input) - 128


@typechecked
def rgb2centered_gray(input: Tensor) -> Tensor:
    """Convert color space.

    Convert images from RGB format to centered gray

    Args:
        input: input image in RGB format of shape [N, 3, H, W],
               ranging 0 to 255

    Returns:
        Tensor: centered gray image of shape [N, 1, H, W],
        ranging -128 to 127
    """
    return rgb2gray(input) - 128


@with_march
@typechecked
def _quantized_convert_color(
    input: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    offset: Tensor,
    mean: Tensor,
    std: Tensor,
    q_scale: Tensor,
    march: str,
) -> Tensor:
    assert march in (
        March.BAYES,
        March.BAYES_E,
    ), "only implemented for {} and {}".format(March.BAYES, March.BAYES_E)
    assert (
        mean.numel() == 1 or mean.numel() == 3
    ), "`mean` must be scalar or size [3] !"
    assert (
        std.numel() == 1 or std.numel() == 3
    ), "`std` must be scalar or size [3] !"
    assert q_scale.numel() == 1, "`q_scale` must be scalar !"

    # broadcast size [1] tensor to size [3]
    mean, std = mean.expand(3).to(input.device), std.expand(3).to(input.device)

    weight_scale = weight_scale / std
    bias = (offset - mean) / std

    # broadcast
    std = std.reshape(3, 1, 1, 1)

    # weight shape: [3, 3, 1, 1]
    weight = weight / std

    return nn.quantized.functional.conv2d(
        input.as_subclass(Tensor),
        weight,
        bias,
        None,
        [1, 1],
        [0, 0],
        [1, 1],
        1,
        "zero",
        "",
        torch.tensor([1.0], device=input.device),
        torch.tensor([0.0], device=input.device),
        "qint8",
        weight_scale,
        torch.tensor([0.0, 0.0, 0.0], device=input.device),
        "qint8",
        torch.tensor([1.0], device=input.device),
        torch.tensor([0.0], device=input.device),
        "qint32",
        None,
        None,
        None,
        q_scale,
        torch.tensor([0.0], device=input.device),
        "qint8",
        march,
    )[0]


@typechecked
def centered_yuv2bgr(
    input: QTensor,
    swing: str = "studio",
    mean: Union[List[float], Tensor] = (128.0,),
    std: Union[List[float], Tensor] = (128.0,),
    q_scale: Union[float, Tensor] = 1.0 / 128.0,
) -> QTensor:
    """Convert color space.

    Convert images from centered YUV444 BT.601 format to transformed
    and quantized BGR. Only use this operator in the quantized model.
    Insert it after `QuantStub`. Pass the scale of `QuantStub` to the
    `q_scale` argument and set scale of `QuantStub` to 1 afterwards.

    Args:
        input: Input images in centered YUV444 BT.601 format,
               centered by the pyramid with -128.
        swing: "studio" for YUV studio swing (Y: -112 to 107,
                U, V: -112 to 112).
                "full" for YUV full swing (Y, U, V: -128 to 127).
                default is "studio"
        mean: BGR mean, a list of float,
                or torch.Tensor, can be a scalar [float],
                or [float, float, float] for per-channel mean.
        std: BGR standard deviation, a list of float,
                or torch.Tensor, can be a scalar [float],
                or [float, float, float] for per-channel std.
        q_scale: BGR quantization scale.

    Returns:
        QTensor: Transformed and quantized image in BGR color,
        `dtype` is qint8.
    """
    weight_map = {
        "studio": [
            [[[1.164]], [[2.016]], [[0.0000]]],
            [[[1.164]], [[-0.392]], [[-0.812]]],
            [[[1.164]], [[0.0000]], [[1.596]]],
        ],
        "full": [
            [[[1.0000]], [[1.7720]], [[0.0000]]],
            [[[1.0000]], [[-0.3441]], [[-0.7141]]],
            [[[1.0000]], [[0.0000]], [[1.4020]]],
        ],
    }
    weight_scale_map = {
        "studio": [0.0158, 0.0091, 0.0125],
        "full": [0.0139, 0.0078, 0.0110],
    }
    offset_map = {
        "studio": [112 * 1.164],
        "full": [128],
    }
    assert (
        swing in weight_map
    ), '`swing` is not valid! must be "full" or "studio"!'

    if isinstance(mean, Tensor):
        mean = mean.float().tolist()
    if isinstance(std, Tensor):
        std = std.float().tolist()
    if isinstance(q_scale, Tensor):
        q_scale = q_scale.float().item()
    assert len(mean) in (1, 3), "len(mean) must be 1 or 3!"
    assert len(std) in (1, 3), "len(mean) must be 1 or 3!"

    out = _quantized_convert_color(
        input.as_subclass(Tensor),
        weight=torch.tensor(weight_map[swing], device=input.device),
        weight_scale=torch.tensor(
            weight_scale_map[swing], device=input.device
        ),
        offset=torch.tensor(offset_map[swing], device=input.device),
        mean=torch.tensor(mean, device=input.device),
        std=torch.tensor(std, device=input.device),
        q_scale=torch.tensor([q_scale], device=input.device),
    )
    return QTensor(out, torch.tensor([q_scale], device=out.device), "qint8")


@typechecked
def centered_yuv2rgb(
    input: QTensor,
    swing: str = "studio",
    mean: Union[List[float], Tensor] = (128.0,),
    std: Union[List[float], Tensor] = (128.0,),
    q_scale: Union[float, Tensor] = 1.0 / 128.0,
) -> QTensor:
    """Convert color space.

    Convert images from centered YUV444 BT.601 format to transformed
    and quantized RGB. Only use this operator in the quantized model.
    Insert it after `QuantStub`. Pass the scale of `QuantStub` to the
    `q_scale` argument and set scale of `QuantStub` to 1 afterwards.

    Args:
        input: Input images in centered YUV444 BT.601 format,
               centered by the pyramid with -128.
        swing: "studio" for YUV studio swing (Y: -112 to 107,
                U, V: -112 to 112).
                "full" for YUV full swing (Y, U, V: -128 to 127).
                default is "studio"
        mean: RGB mean, a list of float,
                or torch.Tensor, can be a scalar [float],
                or [float, float, float] for per-channel mean.
        std: RGB standard deviation, a list of float,
                or torch.Tensor, can be a scalar [float],
                or [float, float, float] for per-channel std.
        q_scale: RGB quantization scale.

    Returns:
        QTensor: Transformed and quantized image in RGB color,
        `dtype` is qint8.
    """
    weight_map = {
        "studio": [
            [[[1.164]], [[0.0000]], [[1.596]]],
            [[[1.164]], [[-0.392]], [[-0.812]]],
            [[[1.164]], [[2.016]], [[0.0000]]],
        ],
        "full": [
            [[[1.0000]], [[0.0000]], [[1.4020]]],
            [[[1.0000]], [[-0.3441]], [[-0.7141]]],
            [[[1.0000]], [[1.7720]], [[0.0000]]],
        ],
    }
    weight_scale_map = {
        "studio": [0.0125, 0.0091, 0.0158],
        "full": [0.0110, 0.0078, 0.0139],
    }
    offset_map = {
        "studio": [112 * 1.164],
        "full": [128],
    }
    assert (
        swing in weight_map
    ), '`swing` is not valid! must be "full" or "studio"!'

    if isinstance(mean, Tensor):
        mean = mean.float().tolist()
    if isinstance(std, Tensor):
        std = std.float().tolist()
    if isinstance(q_scale, Tensor):
        q_scale = q_scale.float().item()
    assert len(mean) in (1, 3), "len(mean) must be 1 or 3!"
    assert len(std) in (1, 3), "len(mean) must be 1 or 3!"

    out = _quantized_convert_color(
        input.as_subclass(Tensor),
        weight=torch.tensor(weight_map[swing], device=input.device),
        weight_scale=torch.tensor(
            weight_scale_map[swing], device=input.device
        ),
        offset=torch.tensor(offset_map[swing], device=input.device),
        mean=torch.tensor(mean, device=input.device),
        std=torch.tensor(std, device=input.device),
        q_scale=torch.tensor([q_scale], device=input.device),
    )
    return QTensor(out, torch.tensor([q_scale], device=out.device), "qint8")


def raycast(
    origins: torch.Tensor, points: torch.Tensor, output_grid: List[int]
) -> torch.Tensor:
    """Compute 2D-bev freespace following with 3D-pointcloud.

    Returns from the ground via a robust ground segmentation algorithm, discard
    ground returns, compute 2D freespace via a 2D visibility algorithm known
    as wall tracking.

    Args:
        origins: tensor of emitter's center points.
        points: points cloud with 4 dims x, y, z, t.
        output_grid: shape of 2D-bev freespace.

    Returns:
        A tensor of specified output size, representing casted rays.
    """
    assert len(output_grid) == 3, "Output_grid should have a length of 3."
    return torch.ops.horizon.raycast(origins, points, output_grid)


@typechecked
@fx_helper.wrap()
def abs(
    input: Union[Tensor, QTensor], overflow_mode: str = "saturate"
) -> Union[Tensor, QTensor]:
    """Horizon version of torch.abs.

    Args:
        input: input tensor.
        overflow_mode: because the quantized data are in int dtype,
        for example, [-128, 127] for qint8, abs(-128) = 128 would cause
        overflow. set overflow_mode
        = "saturate" to clamp the input when overflow occurs, in which
        case abs(-128) = 127. set overflow_mode = "trunc" to truncate
        higher bits, in which case abs(-128) = -128.

    Returns:
        The output tensor.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(abs, (input,), input, overflow_mode)

    return torch.abs(input)


QTensor.patch_torch_func(torch.abs, abs)


@typechecked
def quantized_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Quantized version of torch.nn.functional.conv2d.

    Same as torch.nn.functional.conv2d, but has quantization information
    so that it can be compiled by hbdk. Quantization scales for input, weight
    and bias are all 1.0.
    Only use this operator in the quantized model.

    Args:
        input: Tensor of int dtype such as int8, int16, int32, int64.
        weight: Weight tensor. Must be constant tensor. dtype must be int8.
        bias: Bias tensor. Must be constant tensor.
        dtype could be int8, int16, int32.

    Returns:
        Tensor: Output with dtype int32.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    device = input.device
    assert not input.is_floating_point(), "input must be int dtype!"
    assert weight.dtype == torch.int8, "weight must be int8!"
    assert bias is None or bias.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
    ], "bias must be int8/int16/int32!"
    return nn.quantized.functional.conv2d(
        input=input,
        weight=weight.to(torch.float32).to(device),
        bias=torch.zeros([weight.size(0)]).to(device)
        if bias is None
        else bias.to(torch.float32).to(device),
        sumin=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode="zeros",
        activation="",
        input_scale=torch.ones([1]).to(device),
        input_zero_point=torch.zeros([1]).to(device),
        input_dtype=qint32,
        weight_scale=torch.ones([1]).to(device),
        weight_zero_point=torch.zeros([1]).to(device),
        weight_dtype=qint8,
        bias_scale=torch.ones([1]).to(device),
        bias_zero_point=torch.zeros([1]).to(device),
        bias_dtype=qint32,
        sumin_scale=None,
        sumin_zero_point=None,
        sumin_dtype=None,
        scale=torch.ones([1]).to(device),
        zero_point=torch.zeros([1]).to(device),
        dtype=qint32,
    )[0]
