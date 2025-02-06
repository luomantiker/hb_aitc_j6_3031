from numbers import Integral
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.utils import _pair

# from torchvision.ops import RoIAlign
from horizon_plugin_pytorch._torchvision_wrapper.ops import RoIAlign
from horizon_plugin_pytorch.dtype import QuantDType
from horizon_plugin_pytorch.nn.qat.functional import scale_quanti
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.auto_cast import handle_autocast
from horizon_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)

# RoIAlign raises Exception with cpu amp
# TODO: remove this after torchvision support cpu amp
RoIAlign.forward = handle_autocast()(RoIAlign.forward)


def bbox_clip_with_qtensor(boxes, clip_ratio):
    from horizon_plugin_pytorch.nn.quantized.functional import bbox_clip

    roi_quantized = False
    if isinstance(boxes[0], QTensor):
        roi_quantized = True
        roi_scale = boxes[0].q_scale()
        roi_dtype: QuantDType = boxes[0].dtype
        boxes = [box.as_subclass(torch.Tensor) for box in boxes]

    boxes = bbox_clip(boxes, clip_ratio)

    if roi_quantized:
        boxes = [
            QTensor(
                scale_quanti(
                    box,
                    roi_scale,
                    torch.zeros_like(roi_scale, dtype=torch.long),
                    -1,
                    roi_dtype.min,
                    roi_dtype.max,
                    False,
                    False,
                ),
                roi_scale,
                roi_dtype,
            )
            for box in boxes
        ]

    return boxes


def convert_boxes_to_roi_format(boxes):
    """Convert boxes from List[Tensor[N, 4]] to Tensor[M, 5]."""
    roi_quantized = False
    if isinstance(boxes[0], QTensor):
        roi_quantized = True
        roi_scale = boxes[0].q_scale()
        roi_dtype = boxes[0].dtype
        boxes = [box.as_subclass(torch.Tensor) for box in boxes]
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full_like(b[:, :1], i, dtype=dtype, device=device)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    if roi_quantized:
        rois = QTensor(rois, roi_scale, roi_dtype)
    return rois


def assign_boxes_to_levels(
    box_lists,
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in box_lists to a feature map level index and.

    return the assignment vector.

    Args:
        box_lists(List[Tensor[N,4]]): a list of N boxes.
        min_level(int): smallest feature map level index
        max_level(int): largest feature map level index
        canonical_box_size(int): a canonical box size in pixels(sqrt(box area))
        canonical_level(int): the feature map level index on which a
            canonically-sized box should be placed

    Returns:
        A tensor of length M, where M is the total number of boxes
        aggregated over all N batch images. Each element is the
        feature map index, as an offset from 'min_level', so value i
        means the box is at 'min_level' + i.
    """
    if isinstance(box_lists[0], QTensor):
        box_lists = [box.as_subclass(torch.Tensor) for box in box_lists]
    box_sizes = torch.sqrt(
        torch.cat(
            [
                ((each_box[:, 2] - each_box[:, 0]) + 1)
                * ((each_box[:, 3] - each_box[:, 1]) + 1)
                for each_box in box_lists
            ]
        )
    )

    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size) + 1e-8
    )
    level_assignments = torch.clamp(
        level_assignments,
        min=min_level,
        max=max_level,
    )
    return level_assignments.to(torch.int64) - min_level


class MultiScaleRoIAlign(nn.Module):
    """Multi-scale RoIAlign pooling described in FPN.

    Note that we use different parameters compared with
    torchvision.ops.MultiScaleRoIAlign but similar to the
    implements in detectron2.

    Args:
        output_size: output size of the pooled region
        feature_strides: a stride list of a feature pyramid.
            The strides must be power of 2. scale = 1 / stride.
            Default in ascending order, such as 2, 4, 8, 16, ...
        sampling_ratio: the sampling ratio for the RoIAlign op.
            Only support sampling_ratio = 1 now.
        interpolate_mode: the interpolate mode for the RoIAlign op.
            Only support 'bilinear' now.
        canonical_box_size: A canonical box size in pixels (sqrt(box area)).
            The default is heuristically defined as 224 pixels
            in the FPN paper.
        canonical_level: The feature map level index from which a
            canonically-sized box should be placed.
            The default is defined as level 4 (stride=16) in the FPN paper.
        aligned: if False, use the legacy implementation. If True, align the
            results more perfectly.(This parameter is only supported on BAYES.
            On BERNOULLI2, 'aligned' should be None)
        box_clip_ratio: Cilp the input rois by given ratio of format
            (ratio_x1, ratio_y1, ratio_x2, ratio_y2).
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        feature_strides: List[int],
        sampling_ratio: int = 1,
        interpolate_mode: str = "bilinear",
        canonical_box_size: int = 224,
        canonical_level: int = 4,
        aligned: Optional[bool] = False,
        box_clip_ratio: Optional[Tuple[float, float, float, float]] = None,
    ):
        super(MultiScaleRoIAlign, self).__init__()
        self.output_size = output_size
        self.feature_strides = feature_strides
        self.sampling_ratio = sampling_ratio
        self.mode = interpolate_mode
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.aligned = aligned
        self.box_clip_ratio = box_clip_ratio
        self._check_init()
        self.aligners = self._get_aligners()

    def _check_init(self):
        # check output_size
        if isinstance(self.output_size, Integral):
            self.output_size = _pair(self.output_size)
        assert (
            len(self.output_size) == 2
        ), "output size must have exactly two elements h*w."
        assert isinstance(self.output_size[0], Integral) and isinstance(
            self.output_size[1], Integral
        ), "output size must be 'int'."
        out_height, out_width = _pair(self.output_size)
        assert out_height > 0 and out_width > 0, "output size must be positive"

        # check feature_strides. Must be a power of 2.
        levels = torch.log2(torch.tensor(self.feature_strides))
        assert torch.isclose(
            levels, levels.round()
        ).all(), "feature strides must be a power of 2"

        # assuming feature_strides in ascending order, no checking or reorder

        self.min_level = levels[0].item()
        self.max_level = levels[-1].item()

        assert self.sampling_ratio == 1, "only support sampling_ratio = 1"
        assert self.mode == "bilinear", "only support 'bilinear' mode now"
        assert isinstance(
            self.canonical_box_size, Integral
        ), "canonical_box_size must be int"
        assert isinstance(
            self.canonical_level, Integral
        ), "canonical_level must be int"
        assert isinstance(
            self.aligned, (bool, type(None))
        ), "aligned must be bool or None"
        if self.box_clip_ratio is not None:
            assert (
                isinstance(self.box_clip_ratio, tuple)
                and len(self.box_clip_ratio) == 4
                and all([type(x) == float for x in self.box_clip_ratio])
            ), "box_clip_ratio must be a tuple of 4 float elements"
            msg = "total clip ratio in the same direction must be less than 1"
            assert self.box_clip_ratio[0] + self.box_clip_ratio[2] < 1.0, msg
            assert self.box_clip_ratio[1] + self.box_clip_ratio[3] < 1.0, msg

    @script_quantized_fn
    def forward(
        self,
        x: List[torch.Tensor],
        box_lists: Union[torch.Tensor, List[torch.Tensor]],
    ):
        """Refine this docsting in the future.

        Args:
            x(List[Tensor]): a list of feature maps of NCHW shape, with
                feature srides matching those used to construct this modules.
            box_lists(List[Tensor[L, 4]]): a list of N boxes, where N is the
                number of images in the batch. The box coordinates are defined
                on the original image.

        Returns:
            Tensor: A tensor of shape (M, C, output_size, output_size) where M
            is the total number of boxes aggregated over all N batch images
            and C is the number of channels in 'x'.
        """
        num_level_assignments = len(self.aligners)

        # for rcnn_post_process output compatability
        # convert Tensor[B, N, 6] to [Tensor[N, 4] * B]
        if isinstance(box_lists, torch.Tensor):
            assert (
                box_lists.ndim == 3
            ), "rcnn_post_process output should be Tensor[B, N, 6], but got {}".format(  # noqa
                box_lists.shape
            )
            box_lists = list(torch.unbind(box_lists[:, :, :4], dim=0))

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "arguments to forward must be lists"
        assert len(x) == num_level_assignments, (
            "the number of feature maps must equal to "
            + "the number of feature strides"
        )

        assert len(box_lists) == self._data(x).size(
            0
        ), "the length of box_lists must equal to the batch size N"

        if len(box_lists) == 0:
            return torch.empty(
                (0, self._data(x).shape[1]) + self.output_size,
                device=self._data(x).device,
                dtype=self._data(x).dtype,
            )

        if self.box_clip_ratio is not None:
            box_lists = bbox_clip_with_qtensor(box_lists, self.box_clip_ratio)

        rois = convert_boxes_to_roi_format(box_lists)

        if num_level_assignments == 1:
            return self.aligners[0](x[0], rois)

        mapped_levels = assign_boxes_to_levels(
            box_lists,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level,
        )
        num_boxes = rois.size(0)
        num_channels = self._data(x).shape[1]

        dtype, device = self._data(x).dtype, self._data(x).device
        result = torch.zeros(
            (num_boxes, num_channels) + self.output_size,
            dtype=dtype,
            device=device,
        )

        for level, aligner in enumerate(self.aligners):
            indexs = torch.where(mapped_levels == level)[0]
            rois_per_level = rois[indexs]
            result.index_put_(
                (indexs,),
                self._per_level_align_result(
                    aligner(x[level], rois_per_level)
                ).to(dtype=result.dtype),
            )

        return result

    def _get_aligners(self):
        return nn.ModuleList(
            RoIAlign(
                self.output_size,
                1 / stride,
                self.sampling_ratio,
                self.aligned,
            )
            for stride in self.feature_strides
        )

    def _per_level_align_result(self, x):
        return x

    def _data(self, x):
        return x[0]
