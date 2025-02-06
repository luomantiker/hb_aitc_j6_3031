import math
import warnings

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.jit.annotations import (
    BroadcastingList2,
    BroadcastingList3,
    List,
    Optional,
    Tuple,
)
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.dtype import qinfo
from ...utils.script_quantized_fn import script_quantized_fn
from .activation_function_fit_utils import (
    _get_multi_table_params,
    _lut_int8_to_int8,
    _multi_table_fit,
)


@script_quantized_fn
def _quantize(
    input: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    ch_axis: int,
    dtype: str,
    round_mode: str,
    march: str,
) -> Tensor:
    info = qinfo(dtype)

    return torch.ops.horizon.bpu_scale_quantization(
        input,
        scale,
        zero_point,
        -1 if scale.numel() == 1 else ch_axis,
        info.min,
        info.max,
        dtype,
        round_mode,
        march,
    )


def _dequantize(
    input: Tensor, scale: Tensor, zero_point: Tensor, ch_axis: int, march: str
) -> Tensor:
    return torch.ops.horizon.bpu_scale_dequantization(input, scale, ch_axis)


def _requantize(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if torch.all(torch.eq(scale, input_scale)):
        if dtype == input_dtype:
            return input
        else:
            in_info = qinfo(input_dtype)
            out_info = qinfo(dtype)
            # clip the boundary value
            # if int32 128 requantize to the same scale int8
            # directly change dtype will return -128 !!
            return input.clip(
                max(in_info.min, out_info.min), min(in_info.max, out_info.max)
            ).to(dtype=out_info._storage_type)

    if march == "meta":
        return torch.ops.horizon.meta_requantization(
            input,
            input_scale,
            scale,
            input_dtype,
            dtype,
            march,
        )
    return torch.ops.horizon.bpu_scale_requantization(
        input,
        input_scale,
        scale,
        input_dtype,
        dtype,
        False,
        march,
    )


def _conv_convert_int_params(
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    weight_dtype: str,
    bias: Tensor,
    bias_scale: Tensor,
    bias_dtype: str,
    out_scale: Tensor,
    out_dtype: str,
    input2_scale: Optional[Tensor],
    is_conv_transpose2d: bool,
    groups: int,
    march: str,
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    assert weight_dtype == "qint8", "Only support qint8 weight!"
    return torch.ops.horizon.convert_conv_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        out_scale,
        out_dtype,
        input2_scale
        if input2_scale is not None
        else torch.zeros_like(out_scale),
        is_conv_transpose2d,
        groups,
        march,
    )


def _conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    convert_ret = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        False,
        1,
        march,
    )
    filters = weight.size()[0]
    kernel_size = (weight.size()[2], weight.size(3))
    if march == "bernoulli":
        (
            bpu_weight,
            bpu_weight_shift,
            bpu_bias,
            bpu_bias_shift,
            bpu_input_shift,
            bpu_output_shift,
            bpu_sumin_shift,
            dequant_output_scale,
            _,
        ) = convert_ret
        if not torch.all(
            bpu_input_shift + bpu_weight_shift - bpu_output_shift >= 0
        ):
            warnings.warn(
                "Not support output left shift on Bernoulli hardware, "
                "which may cause unexpected result or accuracy mismatch here."
            )
        conv_ret = torch.ops.horizon.bpu_scale_quanti_convolution_with_shift(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_input_shift.item(),
            bpu_output_shift.item(),
            bpu_bias_shift,
            bpu_weight_shift,
            bpu_sumin_shift.item(),
            True,  # use_bias
            filters,  # filters
            kernel_size,  # kernel_size
            stride,
            padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,
            march,
        )
    else:
        (
            bpu_weight,
            bpu_bias,
            bpu_bias_lshift,
            bpu_escale,
            bpu_escale_lshift,
            bpu_oscale,
            bpu_accu_rshift,
            bpu_output_rshift,
            dequant_output_scale,
        ) = convert_ret
        conv_ret = (
            torch.ops.horizon.meta_quant_convolution(
                input,
                bpu_weight,
                bpu_bias,
                sumin if sumin is not None else torch.zeros(1).to(input),
                bpu_oscale,
                bpu_accu_rshift,
                bpu_bias_lshift,
                bpu_output_rshift,
                bpu_escale,
                bpu_escale_lshift,
                True,  # use_bias,
                filters,
                kernel_size,
                stride,
                padding,
                dilation,
                activation,
                groups,
                True if sumin is not None else False,  # elementwise_input,
                True
                if dtype == "qint32"
                else False,  # disable_output_quantization,
                dtype,  # out_quanti_type,
                march,
            )
            if march == "meta"
            else torch.ops.horizon.bpu_scale_quanti_convolution(
                input,
                bpu_weight,
                bpu_bias,
                sumin if sumin is not None else torch.zeros(1).to(input),
                bpu_oscale,
                bpu_accu_rshift,
                bpu_bias_lshift,
                bpu_output_rshift,
                bpu_escale,
                bpu_escale_lshift,
                True,  # use_bias,
                filters,
                kernel_size,
                stride,
                padding,
                dilation,
                activation,
                groups,
                True if sumin is not None else False,  # elementwise_input,
                True
                if dtype == "qint32"
                else False,  # disable_output_quantization,
                dtype,  # out_quanti_type,
                march,
            )
        )
    return conv_ret, dequant_output_scale


def _conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList3[int],
    padding: BroadcastingList3[int],
    dilation: BroadcastingList3[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    (
        bpu_weight,
        bpu_bias,
        bpu_bias_lshift,
        bpu_escale,
        bpu_escale_lshift,
        bpu_oscale,
        bpu_accu_rshift,
        bpu_output_rshift,
        dequant_output_scale,
    ) = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        False,
        1,
        march,
    )
    filters = weight.size(0)
    kernel_size = (weight.size(2), weight.size(3), weight.size(4))
    conv_ret = torch.ops.horizon.bpu_scale_quanti_convolution3d(
        input,
        bpu_weight,
        bpu_bias,
        sumin if sumin is not None else torch.zeros(1).to(input),
        bpu_oscale,
        bpu_accu_rshift,
        bpu_bias_lshift,
        bpu_output_rshift,
        bpu_escale,
        bpu_escale_lshift,
        True,  # use_bias,
        filters,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        groups,
        True if sumin is not None else False,  # elementwise_input,
        True if dtype == "qint32" else False,  # disable_output_quantization,
        dtype,  # out_quanti_type,
        march,
    )
    return conv_ret, dequant_output_scale


def _max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    return_indices: bool,
    ceil_mode: bool,
    march: str,
) -> Tensor:
    return F.max_pool2d(
        input.to(dtype=torch.float32),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        # directly pass `return_indices` will cause
        # 'boolean dispatch not constant error' when trace
        False,
    ).to(input.dtype)


def _avg_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: None,
    input_scale: Tensor,
    input_zero_point: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    unsqueeze_input = False
    if input.dim() == 3:
        warnings.warn(
            "Average pool 2d with 3-dimensional input is not"
            "supported, will turn input into [1, C, H, W]."
        )
        input = input.unsqueeze(0)
        unsqueeze_input = True

    assert count_include_pad, "count_include_pad must be True"
    assert divisor_override is None, "divisor_override must be None"

    if march == "bernoulli":
        assert dtype == "qint8", "bernoulli only support int8 output"
        res = torch.ops.horizon.bpu_scale_quanti_pooling(
            input,
            "avg",
            kernel_size,
            padding,
            stride,
            ceil_mode,
            "qint8",
            march,
        )
        if unsqueeze_input:
            res = res.squeeze(0)
        return res, input_scale

    accu = torch.ops.horizon.bpu_scale_quanti_pooling(
        input, "sum", kernel_size, padding, stride, ceil_mode, "qint32", march
    )

    intermediate_scale = input_scale * (1 / kernel_size[0] / kernel_size[1])

    if dtype == "qint32":
        if unsqueeze_input:
            accu = accu.squeeze(0)
        return accu, intermediate_scale

    if march == "meta":
        res = torch.ops.horizon.meta_requantization(
            accu,
            intermediate_scale,
            scale,
            "qint32",
            dtype,
            march,
        )
    else:
        res = torch.ops.horizon.bpu_scale_requantization(
            accu,
            intermediate_scale,
            scale,
            "qint32",
            dtype,
            True if march in ("bayes", "bayes-e") else False,
            march,
        )

    if unsqueeze_input:
        res = res.squeeze(0)

    return (res, scale)


def _interpolate(
    input: Tensor,
    size: Optional[BroadcastingList2[int]],
    scale_factor: Optional[BroadcastingList2[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    march: str,
) -> Tensor:
    if input.dim() != 4:
        warnings.warn(
            "interpolate with non-four-dimensional input is a BPU op."
            "Please ensure it is used in hybrid qat mode."
        )
        return F.interpolate(
            input,
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
        )

    if size is not None:
        out_height, out_width = size
    else:
        out_height, out_width = -1, -1
    if scale_factor is not None:
        ratio_height, ratio_width = scale_factor
    else:
        ratio_height, ratio_width = -1.0, -1.0

    # Note!!!
    # We use center mode when implementing nearest interpolate
    # Result of torch 'nearest' interpolate shifts to the bottom right
    # https://github.com/pytorch/pytorch/issues/34808
    if align_corners is None:
        align_corners = False
    return torch.ops.horizon.bpu_quanti_resize(
        input,
        mode,
        align_corners,
        out_height,
        out_width,
        ratio_height,
        ratio_width,
        march,
    )


def _pad(
    input: Tensor,
    pad: List[int],
    mode: str,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if mode == "constant":
        padding_value = float(
            _quantize(
                torch.tensor([float(value)], device=input.device),
                scale,
                zero_point,
                -1,
                dtype,
                "bpu_round",
                march,
            )[0],
        )
    else:
        padding_value = float(value if value else 0.0)

    return torch.nn.functional.pad(
        input.to(dtype=torch.float32), pad, mode, value=padding_value
    ).to(dtype=input.dtype)


def _masked_fill(
    input: Tensor,
    mask: Tensor,
    value: float,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    filled_value = _quantize(
        torch.tensor([float(value)], device=input.device),
        scale,
        zero_point,
        -1,
        dtype,
        "bpu_round",
        march,
    )[0]

    return torch.masked_fill(
        input,
        mask,
        filled_value,
    )


def _roi_align_list(
    input: Tensor,
    boxes: List[Tensor],
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    if isinstance(boxes, (list, tuple)):
        assert len(boxes) == input.size(
            0
        ), "The length of roi list should be equal to batch size"
        for _tensor in boxes:
            assert _tensor.size(1) == 4, (
                "The shape of the tensor in the boxes list is"
                + " not correct as List[Tensor[L, 4]]"
            )
    else:
        raise AssertionError(
            "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
        )

    if march == "bernoulli2" or march == "bernoulli":
        assert not boxes[
            0
        ].is_floating_point(), (
            "roi_align of bernoulli2 only support fixed point rois"
        )

    roi_quantized = not boxes[0].is_floating_point()

    output_size = _pair(output_size)

    rois = boxes

    device = input.device
    roi_dtype = rois[0].dtype

    max_roi_num = max([roi.size(0) for roi in rois])

    aligned_rois: List[Tensor] = []
    valid_mask = torch.empty(0, dtype=torch.bool, device=device)

    # check if illegal roi
    illegal_roi_mask = torch.empty(0, dtype=torch.bool, device=device)

    for roi_per_image in rois:
        append_length = max_roi_num - roi_per_image.size(0)
        aligned_rois.append(
            torch.cat(
                (
                    roi_per_image,
                    torch.zeros(
                        size=(append_length, 4), dtype=roi_dtype, device=device
                    ),
                ),
                dim=0,
            )
        )
        valid_mask = torch.cat(
            (
                valid_mask,
                torch.ones(
                    roi_per_image.size(0), dtype=torch.bool, device=device
                ),
                torch.zeros(append_length, dtype=torch.bool, device=device),
            ),
            dim=0,
        )
        # roi is on the left or on the top of the feature
        if_left_top = torch.logical_or(
            roi_per_image[:, 2] < 0, roi_per_image[:, 3] < 0
        )
        # roi is on the right or on the bottom of the feature
        if_right_bottom = torch.logical_or(
            roi_per_image[:, 0] * spatial_scale
            > (input.size(3) * 4 if roi_quantized else input.size(3)),
            roi_per_image[:, 1] * spatial_scale
            > (input.size(2) * 4 if roi_quantized else input.size(2)),
        )
        roi_out_feature = torch.logical_or(if_left_top, if_right_bottom)
        if_negative_roi = torch.logical_or(
            (roi_per_image[:, 2] - roi_per_image[:, 0] <= 0),
            (roi_per_image[:, 3] - roi_per_image[:, 1] <= 0),
        )

        illegal_roi_mask = torch.cat(
            (
                illegal_roi_mask,
                torch.logical_or(roi_out_feature, if_negative_roi),
                torch.zeros(append_length, dtype=torch.bool, device=device),
            ),
            dim=0,
        )

    batched_rois = torch.stack(aligned_rois, dim=0)

    ret = torch.ops.horizon.bpu_quanti_roi_resize(
        input,
        batched_rois,
        spatial_scale,
        output_size[0] * sampling_ratio,
        output_size[1] * sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )

    ret[illegal_roi_mask] = 0
    return ret[valid_mask]


def _roi_align_tensor(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    march: str,
) -> Tensor:
    if isinstance(boxes, torch.Tensor):
        assert (
            boxes.size(1) == 5
        ), "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        raise AssertionError(
            "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
        )

    rois = boxes

    rois_list: List[Tensor] = []
    forward_mapping = torch.empty(0, dtype=torch.int, device=input.device)

    for batch_idx in range(input.size(0)):
        if not boxes.is_floating_point():
            batch_idx *= 4
        rois_list.append(rois[rois[:, 0] == batch_idx, 1:])
        forward_mapping = torch.cat(
            (forward_mapping, (rois[:, 0] == batch_idx).nonzero()), dim=0
        )

    reverse_mapping = torch.argsort(
        forward_mapping.flatten(), descending=False
    )

    return _roi_align_list(
        input,
        rois_list,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
        interpolate_mode,
        march,
    )[reverse_mapping]


def _cat(
    input_list: List[Tensor],
    dim: int,
    input_scale_list: List[Tensor],
    input_zero_point_list: List[Tensor],
    input_dtype_list: List[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    rescaled_input_list: List[Tensor] = []
    for input, in_scale, in_zero_point, in_dtype in zip(
        input_list, input_scale_list, input_zero_point_list, input_dtype_list
    ):
        rescaled_input_list.append(
            _requantize(
                input,
                in_scale,
                in_zero_point,
                in_dtype,
                scale,
                zero_point,
                dtype,
                march,
            )
        )

    return torch.cat(rescaled_input_list, dim)


def _add(
    x: Tensor,
    y: Tensor,
    x_scale: Tensor,
    y_scale: Tensor,
    x_zero_point: Tensor,
    y_zero_point: Tensor,
    x_dtype: str,
    y_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march in ["bayes", "bayes-e", "meta"]:
        # Hardware computation process:
        #   (x << shift + y * qscale) >> out_shift
        #   -32767 <= qscale <= 32767, 0 <= out_shift <= 31
        #   -31 <= shift <= 23 if x_dtype == qint8
        #                <= 15 if x_dtype == qint16
        # So:
        #   x * sx + y * sy
        #   = (x * sx / inter_scale + y * sy / inter_scale) * inter_scale
        #   1 / 2 ** shift = sx / inter_scale
        #   qscale = sy / inter_scale = sy * 2 ** shift / sx <= 31
        # => shift <= log2(32767 * sx / sy)
        # So we get shift <= min{23 or 15, log2(32767 * sx / sy)}
        # If x/y is per channel quantized, each scale must satisfy the limit.
        # To reduce the precision loss, `shift` should be as large as possible.
        if torch.min(y_scale / x_scale) > torch.min(x_scale / y_scale):
            x, y = y, x
            x_scale, y_scale = y_scale, x_scale
            x_dtype, y_dtype = y_dtype, x_dtype
        xshift = 23 if x_dtype == "qint8" else 15
        yshift = (
            torch.floor(torch.log2(32767 * x_scale / y_scale))
            .to(torch.int32)
            .min()
        )
        shift = yshift.clamp_max(xshift)
        x = x.to(dtype=torch.int32) << shift
        origin_y_shape = y.shape
        y = (
            y.to(dtype=torch.int32)
            * torch.floor(y_scale * (2 ** shift) / x_scale + 0.5)
            .clamp(-32767, 32767)
            .to(torch.int32)
            .reshape(-1, 1, 1)
        ).reshape(origin_y_shape)
        intermediate_scale = x_scale / (2 ** shift)
    elif march == "bernoulli2":
        if x_scale.prod() > y_scale.prod():
            feature = y
            feature_scale = y_scale
            sumin = x
            sumin_scale = x_scale
        else:
            feature = x
            feature_scale = x_scale
            sumin = y
            sumin_scale = y_scale

        intermediate_scale = torch.max(
            feature_scale / 127, sumin_scale / (1 << 25)
        )

        weight = (
            torch.ops.horizon.round(feature_scale / intermediate_scale)
            .clamp(-128, 127)
            .to(dtype=torch.int8)
        )
        feature = feature.to(dtype=torch.int32) * weight.reshape(1, -1, 1, 1)

        sumin_weight = torch.ops.horizon.round(
            sumin_scale / intermediate_scale
        )
        m, e = torch.ops.horizon.frexp(sumin_weight)
        qm = (
            (2 ** e.clamp_max(15) * m)
            .clamp_max((1 << 15) - 1)
            .to(dtype=torch.int)
        )
        left_shift = (e - 15).clamp_min(0)
        sumin = (
            sumin.to(dtype=torch.int32) << left_shift.reshape(1, -1, 1, 1)
        ) * qm.reshape(1, -1, 1, 1)

        x = feature
        y = sumin

    else:
        shift = 14
        intermediate_scale = torch.max(x_scale, y_scale) / (2 ** shift)
        if x_scale.numel() > 1:
            x = x.to(dtype=torch.int32) * torch.ops.horizon.round(
                x_scale / intermediate_scale
            ).reshape(1, -1, 1, 1).to(dtype=torch.int32)
            y = y.to(dtype=torch.int32) * torch.ops.horizon.round(
                y_scale / intermediate_scale
            ).reshape(1, -1, 1, 1).to(dtype=torch.int32)
        else:
            x = x.to(dtype=torch.int32) * torch.ops.horizon.round(
                x_scale / intermediate_scale
            ).to(dtype=torch.int32)
            y = y.to(dtype=torch.int32) * torch.ops.horizon.round(
                y_scale / intermediate_scale
            ).to(dtype=torch.int32)

    add_res = torch.add(x, y)
    add_res = _requantize(
        add_res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )
    return add_res


def _grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    grid_scale: Tensor,
    grid_zero_point: Tensor,
    grid_dtype: str,
    march: str,
) -> Tensor:
    # Convert from xy to yx.
    grid_yx = torch.stack((grid[..., 1], grid[..., 0]), dim=-1)

    # Compute coord_shift.
    h, w = input.size(2), input.size(3)
    grid_h, grid_w = grid.size(1), grid.size(2)

    h = h if h > grid_h else grid_h
    w = w if w > grid_w else grid_w

    max_coord = h if h > w else w
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = min(coord_shift, 8)
    coord_shift = coord_shift if coord_shift > 0 else 0

    # Coord int16 quantization.
    grid_out_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)
    grid_yx = _requantize(
        grid_yx,
        grid_scale,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )

    # Convert to absolute grid.
    n, h, w, _ = grid.shape
    base_coord = (
        torch.stack(
            [
                torch.arange(h, dtype=torch.int32, device=grid.device)
                .reshape(1, h, 1)
                .expand(n, h, w),
                torch.arange(w, dtype=torch.int32, device=grid.device)
                .reshape(1, 1, w)
                .expand(n, h, w),
            ],
            dim=-1,
        )
        * (1 << coord_shift)
    )
    absolute_grid = grid_yx + base_coord
    # Convert grid format from [n, h, w, (y, x)] to [n, 1, (y, x), h, w].
    absolute_grid = absolute_grid.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.horizon.bpu_quanti_grid_sample(
        input,
        absolute_grid,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


def _grid_sample_norm_grid(
    input: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
    align_corners: bool,
    grid_scale: Tensor,
    grid_zero_point: Tensor,
    grid_dtype: str,
    march: str,
) -> Tensor:
    # Compute coord_shift.
    h, w = input.size(2), input.size(3)
    grid_h, grid_w = grid.size(1), grid.size(2)

    max_coord = max_coord = max(max(h, w), max(grid_h, grid_w))
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)

    # convert grid from -1 ~ 1 to -(size - 1) / 2 ~ (size - 1) / 2
    # and same out scale
    grid_x = grid[..., :1]
    grid_y = grid[..., 1:]

    grid_out_scale = torch.tensor(
        1.0 / (1 << coord_shift), dtype=torch.float, device=grid.device
    ).reshape(1)

    rescaled_grid_x = _requantize(
        grid_x,
        grid_scale * (w - 1) / 2,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )
    rescaled_grid_y = _requantize(
        grid_y,
        grid_scale * (h - 1) / 2,
        grid_zero_point,
        grid_dtype,
        grid_out_scale,
        grid_zero_point,
        "qint16",
        march,
    )

    # add (size - 1) / 2 to grid
    rescaled_grid_x += int((1 << coord_shift) * (w - 1) / 2)
    rescaled_grid_y += int((1 << coord_shift) * (h - 1) / 2)

    absolute_grid_yx = torch.cat((rescaled_grid_y, rescaled_grid_x), dim=3)
    # Convert grid format from [n, h, w, (y, x)] to [n, 1, (y, x), h, w].
    absolute_grid_yx = absolute_grid_yx.permute(0, 3, 1, 2).unsqueeze(1)

    return torch.ops.horizon.bpu_quanti_grid_sample(
        input,
        absolute_grid_yx,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march,
    )


@script_quantized_fn
def _horizon_nn_filter(
    inputs: List[Tensor],
    threshold: float,
    idx_range: Tuple[int, int],
    march: str,
) -> List[List[Tensor]]:
    score = inputs[0]

    max_value, max_idx = score[:, idx_range[0] : idx_range[1], :, :].max(
        dim=1, keepdim=False
    )
    max_idx = max_idx + idx_range[0]

    if march in ["bayes", "bayes-e", "meta"]:
        mask = max_value >= threshold
    else:
        mask = max_value > threshold

    batch_size, c, h, w = score.shape
    otype = score.dtype
    h_index = (
        torch.arange(h, device=score.device, dtype=otype)
        .reshape(1, 1, -1, 1)
        .expand(batch_size, 1, h, w)
    )
    w_index = (
        torch.arange(w, device=score.device, dtype=otype)
        .reshape(1, 1, 1, -1)
        .expand(batch_size, 1, h, w)
    )
    coord = torch.cat([h_index, w_index], dim=1)

    mask = mask.flatten(1, 2)
    max_value = max_value.flatten(1, 2)
    max_idx = max_idx.flatten(1, 2)
    coord = coord.permute(0, 2, 3, 1).flatten(1, 2)
    inputs = [input.permute(0, 2, 3, 1).flatten(1, 2) for input in inputs]

    batch_size, _, h, w = score.shape

    ret: List[List[Tensor]] = []
    for i in range(batch_size):
        m = mask[i]
        per_image_ret = [max_value[i][m], max_idx[i][m], coord[i][m]]
        per_image_ret += [data[i][m] for data in inputs]
        ret.append(per_image_ret)

    return ret


def _filter(
    inputs: List[Tensor],
    scales: List[Tensor],
    zero_points: List[Tensor],
    dtypes: List[str],
    threshold: float,
    idx_range: Tuple[int, int],
    march: str,
) -> List[List[Tensor]]:
    return filter(
        inputs, scales, zero_points, dtypes, threshold, idx_range, march
    )


@script_quantized_fn
def filter(
    inputs: List[Tensor],
    scales: List[Tensor],
    zero_points: List[Tensor],
    dtypes: List[str],
    threshold: float,
    idx_range: Tuple[int, int],
    march: str,
) -> List[List[Tensor]]:
    if inputs[0].dtype in (torch.int8, torch.int16, torch.int32):
        is_bpu_inference = True
        inputs = [
            _dequantize(data, scale, zero_point, -1, march)
            for data, scale, zero_point in zip(inputs, scales, zero_points)
        ]
    else:
        is_bpu_inference = False

    score = inputs[0]

    _qtype_limit = {
        "qint4": (-8, 7),
        "quint4": (0, 15),
        "qint8": (-128, 127),
        "qint16": (-32768, 32767),
    }

    if is_bpu_inference:
        if march in [
            "bayes",
            "bayes-e",
            "nash",
            "nash-e",
            "nash-m",
            "nash-p",
            "meta",
        ]:
            working_threshold = (
                (
                    (
                        (threshold / scales[0] + zero_points[0])
                        .ceil()
                        .clamp(*_qtype_limit[dtypes[0]])
                        - zero_points[0]
                    )
                    * scales[0]
                )
                .to(dtype=score.dtype)
                .item()
            )
        else:
            working_threshold = (
                (
                    (
                        (threshold / scales[0] + zero_points[0])
                        .floor()
                        .clamp(*_qtype_limit[dtypes[0]])
                        - zero_points[0]
                    )
                    * scales[0]
                )
                .to(dtype=score.dtype)
                .item()
            )
    else:
        working_threshold = threshold

    max_value, max_idx = score[:, idx_range[0] : idx_range[1], :, :].max(
        dim=1, keepdim=False
    )
    max_idx = max_idx + idx_range[0]

    if march in ["bernoulli", "bernoulli2"]:
        mask = max_value > working_threshold
    else:
        mask = max_value >= working_threshold

    batch_size, c, h, w = score.shape
    otype = torch.int16 if is_bpu_inference else score.dtype
    h_index = (
        torch.arange(h, device=score.device, dtype=otype)
        .reshape(1, 1, -1, 1)
        .expand(batch_size, 1, h, w)
    )
    w_index = (
        torch.arange(w, device=score.device, dtype=otype)
        .reshape(1, 1, 1, -1)
        .expand(batch_size, 1, h, w)
    )
    coord = torch.cat([h_index, w_index], dim=1)

    if is_bpu_inference:
        max_idx = max_idx.to(dtype=torch.int16)
        coord = coord.to(dtype=torch.int16)

    mask = mask.flatten(1, 2)
    max_value = max_value.flatten(1, 2)
    max_idx = max_idx.flatten(1, 2)
    coord = coord.permute(0, 2, 3, 1).flatten(1, 2)
    inputs = [input.permute(0, 2, 3, 1).flatten(1, 2) for input in inputs]

    batch_size, _, h, w = score.shape

    ret: List[List[Tensor]] = []
    for i in range(batch_size):
        m = mask[i]
        per_image_ret = [max_value[i][m], max_idx[i][m], coord[i][m]]
        per_image_ret += [data[i][m] for data in inputs]
        ret.append(per_image_ret)

    return ret


def _max(
    input: Tensor,
    dim: int,
    keepdim: bool,
    group: int,
    march: str,
) -> Tuple[Tensor, Tensor]:
    idx, value = torch.ops.horizon.bpu_post_process_channel_argmax(
        input, group, march
    )
    return value, idx


def _sub(
    x: Tensor,
    y: Tensor,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    y_scale: Tensor,
    y_zero_point: Tensor,
    y_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march in ["bayes", "bayes-e", "meta"]:
        negative_inter_scale = 1
        if torch.min(y_scale / x_scale) > torch.min(x_scale / y_scale):
            x, y = y, x
            x_scale, y_scale = y_scale, x_scale
            x_dtype, y_dtype = y_dtype, x_dtype
            negative_inter_scale = -1
        xshift = 23 if x_dtype == "qint8" else 15
        yshift = (
            torch.floor(torch.log2(32767 * x_scale / y_scale))
            .to(torch.int32)
            .min()
        )
        shift = yshift.clamp_max(xshift)
        x = x.to(dtype=torch.int32) << shift
        origin_y_shape = y.shape
        y = (
            y.to(dtype=torch.int32)
            * torch.floor(-y_scale * (2 ** shift) / x_scale + 0.5)
            .clamp(-32767, 32767)
            .to(torch.int32)
            .reshape(-1, 1, 1)
        ).reshape(origin_y_shape)
        intermediate_scale = x_scale / (2 ** shift) * negative_inter_scale
        ret = torch.add(x, y)
        return _requantize(
            ret,
            intermediate_scale,
            torch.zeros_like(intermediate_scale).to(dtype=torch.long),
            "qint32",
            scale,
            zero_point,
            dtype,
            march,
        )
    else:
        info = qinfo(y_dtype)
        return _add(
            x,
            (y.to(torch.int32) * -1).clamp(info.min, info.max),
            x_scale,
            y_scale,
            x_zero_point,
            y_zero_point,
            x_dtype,
            y_dtype,
            scale,
            zero_point,
            dtype,
            march,
        )


def _lut(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    table: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    assert data_type == "qint8" and dtype == "qint8"
    return _lut_int8_to_int8(table, data, march)


def _get_multi_table_params_impl(
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    left_line_xmin: Tensor,
    left_line_ymin: Tensor,
    left_line_xmax: Tensor,
    left_line_ymax: Tensor,
    right_line_xmin: Tensor,
    right_line_ymin: Tensor,
    right_line_xmax: Tensor,
    right_line_ymax: Tensor,
    left_constant_fit_y: Tensor,
    right_constant_fit_y: Tensor,
    qint_dense_xmin: Tensor,
    qint_dense_xmax: Tensor,
    qint_sparse_xmin: Tensor,
    qint_sparse_xmax: Tensor,
    march: str,
):
    return _get_multi_table_params(
        data_scale,
        data_zero_point,
        data_type,
        scale,
        zero_point,
        dtype,
        left_line_xmin,
        left_line_ymin,
        left_line_xmax,
        left_line_ymax,
        right_line_xmin,
        right_line_ymin,
        right_line_xmax,
        right_line_ymax,
        left_constant_fit_y,
        right_constant_fit_y,
        qint_dense_xmin,
        qint_dense_xmax,
        qint_sparse_xmin,
        qint_sparse_xmax,
    )


def _multi_table_fit_impl(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    dense_table: Tensor,
    qint_dense_xmin: Tensor,
    qint_dense_xmax: Tensor,
    sparse_table: Tensor,
    qint_sparse_xmin: Tensor,
    qint_sparse_xmax: Tensor,
    left_line_xmin: Tensor,
    left_line_ymin: Tensor,
    left_line_xmax: Tensor,
    left_line_ymax: Tensor,
    right_line_xmin: Tensor,
    right_line_ymin: Tensor,
    right_line_xmax: Tensor,
    right_line_ymax: Tensor,
    qint_left_constant_xmin: Tensor,
    qint_left_constant_xmax: Tensor,
    qint_right_constant_xmin: Tensor,
    qint_right_constant_xmax: Tensor,
    left_constant_fit_y: Tensor,
    right_constant_fit_y: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    is_symmetric: bool,
    symmetric_k: int,
    symmetric_b: Tensor,
    march: str,
) -> Tensor:
    out = torch.zeros_like(data, dtype=torch.int32)
    if is_symmetric:
        assert data_type != "qint8", "input int8 cannot use symmetric mode"
        # if use symmetric mode, compiler calculation:
        # (right_out * k + b) >> 8 if out_type is int8
        # so if out_type is int8, transform out_scale and out_type to int16
        # and do right shfit use int16 result to be consistent with compiler
        mask = torch.logical_and(
            data.to(torch.int32) >= 0, data.to(torch.int32) <= 32767
        )
        right_data = torch.masked_select(data, mask)
        right_out = _multi_table_fit(
            right_data,
            data_scale,
            data_zero_point,
            data_type,
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            scale,
            zero_point,
            "qint16",
            march,
            is_symmetric,
        )
        out.masked_scatter_(mask, right_out)
        mask = torch.logical_and(
            data.to(torch.int32) < 0, data.to(torch.int32) >= -32768
        )
        left_data = (-1) * torch.masked_select(data.to(torch.int32), mask)
        left_out = (
            _multi_table_fit(
                left_data,
                data_scale,
                data_zero_point,
                data_type,
                dense_table,
                qint_dense_xmin,
                qint_dense_xmax,
                sparse_table,
                qint_sparse_xmin,
                qint_sparse_xmax,
                left_line_xmin,
                left_line_ymin,
                left_line_xmax,
                left_line_ymax,
                right_line_xmin,
                right_line_ymin,
                right_line_xmax,
                right_line_ymax,
                qint_left_constant_xmin,
                qint_left_constant_xmax,
                qint_right_constant_xmin,
                qint_right_constant_xmax + 1,
                left_constant_fit_y,
                right_constant_fit_y,
                scale,
                zero_point,
                "qint16",
                march,
                is_symmetric,
            )
            * symmetric_k
            + symmetric_b.to(torch.int32)
        )
        out.masked_scatter_(mask, left_out)
    else:
        out = _multi_table_fit(
            data,
            data_scale,
            data_zero_point,
            data_type,
            dense_table,
            qint_dense_xmin,
            qint_dense_xmax,
            sparse_table,
            qint_sparse_xmin,
            qint_sparse_xmax,
            left_line_xmin,
            left_line_ymin,
            left_line_xmax,
            left_line_ymax,
            right_line_xmin,
            right_line_ymin,
            right_line_xmax,
            right_line_ymax,
            qint_left_constant_xmin,
            qint_left_constant_xmax,
            qint_right_constant_xmin,
            qint_right_constant_xmax,
            left_constant_fit_y,
            right_constant_fit_y,
            scale,
            zero_point,
            "qint16",
            march,
        )
    if dtype == "qint8":
        out = ((out.to(torch.int32) + 128) >> 8).to(torch.int16)
        out = torch.clamp(out, -128, 127).to(torch.int8)
    else:
        out = torch.clamp(out, -32768, 32767).to(torch.int16)
    return out


def _matmul(
    input: Tensor,
    other: Tensor,
    input_trans: bool,
    other_trans: bool,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    other_scale: Tensor,
    other_zero_point: Tensor,
    other_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if input_trans:
        input = input.transpose(-1, -2)
    if other_trans:
        other = other.transpose(-1, -2)

    intermediate_scale = input_scale * other_scale

    if march == "meta":
        res = torch.matmul(
            input.to(dtype=torch.float64), other.to(dtype=torch.float64)
        ).to(dtype=torch.int64)
        return _requantize(
            res,
            intermediate_scale,
            torch.zeros_like(intermediate_scale).to(dtype=torch.long),
            "qint64",
            scale,
            zero_point,
            dtype,
            march,
        )

    if input_dtype == "qint16":
        # We need to consider two constraints:
        #  1. BPU input range is limited to [-32768, 32767 - 128]
        #  2. Avoid overflow of sum operation
        #     For [M, K] [K, N] matmul, there are K values to be sumed, and
        #     the result is of int32 type, so each value is limited to
        #     [INT32_MIN / K, INT32_MAX / K], input value is limited to
        #     [-sqrt(INT32_MAX / K), sqrt(INT32_MAX / K)] ~=
        #     [INT16_MIN * sqrt(2 / K), INT16_MAX * sqrt(2 / K)]
        #     Input type is int16, and the value range is multiplied with
        #     sqrt(2 / K), so the scale should multiply with sqrt(K / 2)
        scale_scale_1 = 32767 / (32767 - 128)
        scale_scale_2 = torch.sqrt(torch.tensor(input.size(-1) / 2)).item()
        scale_scale = (
            scale_scale_1 if scale_scale_1 > scale_scale_2 else scale_scale_2
        )

        intermediate_scale *= scale_scale ** 2

        input = _requantize(
            input,
            input_scale,
            input_zero_point,
            input_dtype,
            input_scale * scale_scale,
            torch.zeros_like(input_scale).to(dtype=torch.long),
            "qint16",
            march,
        )
        other = _requantize(
            other,
            other_scale,
            other_zero_point,
            other_dtype,
            other_scale * scale_scale,
            torch.zeros_like(other_scale).to(dtype=torch.long),
            "qint16",
            march,
        )

    if input.is_cuda:
        res = torch.matmul(
            input.to(dtype=torch.float64), other.to(dtype=torch.float64)
        ).to(dtype=torch.int32)
    else:
        res = torch.matmul(
            input.to(dtype=torch.int32), other.to(dtype=torch.int32)
        )

    res = _requantize(
        res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )

    return res


def _base_grid_generator(
    size: BroadcastingList2[int],
    with_third_channel: bool,
    device: torch.device,
    march: str,
) -> Tensor:
    size = _pair(size)

    x = (
        torch.arange(size[1], dtype=torch.int16, device=device)
        .unsqueeze(0)
        .expand(size)
    )
    y = (
        torch.arange(size[0], dtype=torch.int16, device=device)
        .unsqueeze(-1)
        .expand(size)
    )

    tensor_list = [x, y]

    if with_third_channel:
        ones = torch.ones(size, dtype=torch.int16, device=device)
        tensor_list.append(ones)

    return torch.stack(tensor_list, dim=-1)


def _mul(
    input: Tensor,
    other: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    other_scale: Tensor,
    other_zero_point: Tensor,
    other_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if "qbool" in [input_dtype, other_dtype]:
        res = torch.mul(input, other)
        return res
    else:
        if march == "bernoulli":
            assert input_dtype == other_dtype
            assert input_dtype == "qint8"
            assert input.dtype == torch.int8
            assert other.dtype == torch.int8
            input_shift = (-1) * torch.log2(input_scale).to(torch.int8)
            other_shift = (-1) * torch.log2(other_scale).to(torch.int8)
            out_shift = (-1) * torch.log2(scale).to(torch.int8)
            data_x, data_y = torch.broadcast_tensors(input, other)
            res = torch.ops.horizon.bpu_quanti_mul(
                data_x, data_y, input_shift, other_shift, out_shift, march
            )
            return res
        else:
            oscale = input_scale * other_scale
            res = torch.mul(
                input.to(dtype=torch.int32), other.to(dtype=torch.int32)
            )
            res = _requantize(
                res,
                oscale,
                torch.zeros_like(oscale).to(dtype=torch.long),
                "qint32",
                scale,
                zero_point,
                dtype,
                march,
            )
            return res


def _sum(
    x: Tensor,
    dim: int,
    keepdim: bool,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march == "meta":
        r = torch.sum(x, dim, keepdim, dtype=torch.int64)
        return _requantize(
            r, x_scale, x_zero_point, "qint64", scale, zero_point, dtype, march
        )
    r = torch.sum(x, dim, keepdim, dtype=torch.int32)
    return _requantize(
        r, x_scale, x_zero_point, "qint32", scale, zero_point, dtype, march
    )


def _softmax(
    data: Tensor,
    data_scale: Tensor,
    data_zero_point: Tensor,
    data_type: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    exp_out_scale: Tensor,
    exp_out_zero_point: Tensor,
    exp_out_type: str,
    reciprocal_out_scale: Tensor,
    reciprocal_out_zero_point: Tensor,
    reciprocal_out_type: str,
    exp_dense_table: Tensor,
    exp_qint_dense_xmin: Tensor,
    exp_qint_dense_xmax: Tensor,
    exp_sparse_table: Tensor,
    exp_qint_sparse_xmin: Tensor,
    exp_qint_sparse_xmax: Tensor,
    exp_left_line_xmin: Tensor,
    exp_left_line_ymin: Tensor,
    exp_left_line_xmax: Tensor,
    exp_left_line_ymax: Tensor,
    exp_right_line_xmin: Tensor,
    exp_right_line_ymin: Tensor,
    exp_right_line_xmax: Tensor,
    exp_right_line_ymax: Tensor,
    exp_qint_left_constant_xmin: Tensor,
    exp_qint_left_constant_xmax: Tensor,
    exp_qint_right_constant_xmin: Tensor,
    exp_qint_right_constant_xmax: Tensor,
    exp_left_constant_fit_y: Tensor,
    exp_right_constant_fit_y: Tensor,
    rescale_shift: Tensor,
    reciprocal_dense_table: Tensor,
    reciprocal_qint_dense_xmin: Tensor,
    reciprocal_qint_dense_xmax: Tensor,
    reciprocal_sparse_table: Tensor,
    reciprocal_qint_sparse_xmin: Tensor,
    reciprocal_qint_sparse_xmax: Tensor,
    reciprocal_left_line_xmin: Tensor,
    reciprocal_left_line_ymin: Tensor,
    reciprocal_left_line_xmax: Tensor,
    reciprocal_left_line_ymax: Tensor,
    reciprocal_right_line_xmin: Tensor,
    reciprocal_right_line_ymin: Tensor,
    reciprocal_right_line_xmax: Tensor,
    reciprocal_right_line_ymax: Tensor,
    reciprocal_qint_left_constant_xmin: Tensor,
    reciprocal_qint_left_constant_xmax: Tensor,
    reciprocal_qint_right_constant_xmin: Tensor,
    reciprocal_qint_right_constant_xmax: Tensor,
    reciprocal_left_constant_fit_y: Tensor,
    reciprocal_right_constant_fit_y: Tensor,
    march: str,
) -> Tensor:
    data = data.to(torch.int16) - torch.max(
        data, dim=1, keepdim=True
    ).values.to(torch.int16)
    exp_out = _multi_table_fit(
        data,
        data_scale,
        data_zero_point,
        "qint16",
        exp_dense_table,
        exp_qint_dense_xmin,
        exp_qint_dense_xmax,
        exp_sparse_table,
        exp_qint_sparse_xmin,
        exp_qint_sparse_xmax,
        exp_left_line_xmin,
        exp_left_line_ymin,
        exp_left_line_xmax,
        exp_left_line_ymax,
        exp_right_line_xmin,
        exp_right_line_ymin,
        exp_right_line_xmax,
        exp_right_line_ymax,
        exp_qint_left_constant_xmin,
        exp_qint_left_constant_xmax,
        exp_qint_right_constant_xmin,
        exp_qint_right_constant_xmax,
        exp_left_constant_fit_y,
        exp_right_constant_fit_y,
        exp_out_scale,
        exp_out_zero_point,
        exp_out_type,
        march,
        False,
    )
    exp_out = torch.clamp(exp_out, qinfo("qint16").min, qinfo("qint16").max)
    exp_sum = torch.sum(exp_out, 1, True)
    exp_sum = torch.clamp(
        (exp_sum / (2 ** rescale_shift)),
        qinfo("qint16").min,
        qinfo("qint16").max,
    ).to(torch.int16)
    exp_sum_scale = exp_out_scale * 2 ** rescale_shift
    reciprocal_out = _multi_table_fit(
        exp_sum,
        exp_sum_scale,
        torch.zeros_like(exp_sum_scale).to(dtype=torch.long),
        "qint16",
        reciprocal_dense_table,
        reciprocal_qint_dense_xmin,
        reciprocal_qint_dense_xmax,
        reciprocal_sparse_table,
        reciprocal_qint_sparse_xmin,
        reciprocal_qint_sparse_xmax,
        reciprocal_left_line_xmin,
        reciprocal_left_line_ymin,
        reciprocal_left_line_xmax,
        reciprocal_left_line_ymax,
        reciprocal_right_line_xmin,
        reciprocal_right_line_ymin,
        reciprocal_right_line_xmax,
        reciprocal_right_line_ymax,
        reciprocal_qint_left_constant_xmin,
        reciprocal_qint_left_constant_xmax,
        reciprocal_qint_right_constant_xmin,
        reciprocal_qint_right_constant_xmax,
        reciprocal_left_constant_fit_y,
        reciprocal_right_constant_fit_y,
        reciprocal_out_scale,
        reciprocal_out_zero_point,
        reciprocal_out_type,
        march,
        False,
    )
    reciprocal_out = torch.clamp(
        reciprocal_out, qinfo("qint16").min, qinfo("qint16").max
    )
    intermediate_scale = exp_out_scale * reciprocal_out_scale
    intermediate_res = exp_out.to(torch.int32) * reciprocal_out.to(torch.int32)
    softmax_out = _requantize(
        intermediate_res,
        intermediate_scale,
        torch.zeros_like(intermediate_scale).to(dtype=torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )
    return softmax_out


def _detection_post_process_v1(
    # Tensors
    data: List[Tensor],
    anchor: List[Tensor],
    exp_table: Tensor,
    image_sizes: Tensor,
    # task params
    num_classes: int,
    # shifts
    input_shifts: List[int],
    exp_shift: int,
    # filter params
    box_filter_threshold: int,
    class_offsets: List[int],
    seed: int,
    # decode params
    use_clippings: bool,
    # nms params
    nms_threshold: int,
    nms_margin: int,
    post_nms_top_k: int,
    use_stable_sort: Optional[bool],
    march: str,
) -> List[Tuple[Tensor, Tensor, Tensor]]:
    num_anchors: List[int] = [int(a.size(1) / 4) for a in anchor]
    anchor_start_idx: List[int] = [0]
    for per_branch_num_anchors in num_anchors:
        anchor_start_idx.append(anchor_start_idx[-1] + per_branch_num_anchors)
    anchor_start_idx = anchor_start_idx[:-1]

    block_sizes: List[Tuple[int, int]] = []

    if march in ["bayes", "bayes-e", "meta"]:
        for branch_data in data:
            block_sizes.append((branch_data.size(2), branch_data.size(3)))
    else:
        max_input_size = 144 * 4 * 2048
        for num_anchor in num_anchors:
            # per_anchor_size is aligned with 4
            per_anchor_size = math.ceil((4 + num_classes) / 4) * 4
            max_tile_area = math.floor(
                max_input_size
                / (math.ceil(per_anchor_size * num_anchor / 4) * 4)
            )
            max_tile_w = (
                math.ceil(math.floor(math.sqrt(max_tile_area)) / 8) * 8
            )
            max_tile_h = math.floor(max_tile_area / max_tile_w)
            block_sizes.append((max_tile_h, max_tile_w))

    stride_hw: List[Tuple[int, int]] = []
    for per_branch_anchor in anchor:
        stride_hw.append(
            (
                int(
                    (
                        per_branch_anchor[0, 1, 1, 0]
                        - per_branch_anchor[0, 1, 0, 0]
                    ).item()
                ),
                int(
                    (
                        per_branch_anchor[0, 0, 0, 1]
                        - per_branch_anchor[0, 0, 0, 0]
                    ).item()
                ),
            )
        )

    anchor = torch.cat(
        [
            per_branch_anchor[0, :, 0, 0].flatten()
            for per_branch_anchor in anchor
        ]
    ).reshape(-1, 4)

    x1 = anchor[:, 0]
    y1 = anchor[:, 1]
    x2 = anchor[:, 2]
    y2 = anchor[:, 3]

    anchor = torch.stack(
        [y2 - y1, x2 - x1, (y1 + y2) / 2, (x1 + x2) / 2], dim=-1
    )

    shifted_anchor = torch.ops.horizon.round(anchor * 4).to(dtype=torch.int32)

    assert shifted_anchor.min() >= 0 and shifted_anchor.max() <= (
        (1 << 16) - 1
    ), "anchor value out of range"

    per_class_idx: List[int] = []
    per_class_threshold: List[int] = []

    batch_size = data[0].size(0)

    if use_stable_sort is None:
        if march in ("bernoulli", "bernoulli2"):
            use_stable_sort = False
        else:
            use_stable_sort = True

    if march in ("bernoulli", "bernoulli2") and use_stable_sort:
        pow_of_2_block_sizes: List[Tuple[int, int]] = []
        for max_tile_h, max_tile_w in block_sizes:
            pow_of_2_block_sizes.append(
                (
                    int(
                        2
                        ** math.floor(
                            torch.log2(torch.tensor(max_tile_h)).item()
                        )
                    ),
                    int(
                        2
                        ** math.floor(
                            torch.log2(torch.tensor(max_tile_w)).item()
                        )
                    ),
                )
            )

        skip_nms_ret = torch.ops.horizon.bpu_quanti_proposal(
            [d.cpu() for d in data],
            shifted_anchor.cpu(),
            exp_table.cpu(),
            image_sizes.expand(batch_size, 2).float().cpu(),
            num_anchors,
            [num_classes] * len(data),
            input_shifts,
            exp_shift,
            [block_size[0] for block_size in pow_of_2_block_sizes],
            [block_size[1] for block_size in pow_of_2_block_sizes],
            box_filter_threshold,
            True,  # filter_invalid_boxes
            per_class_idx,
            per_class_threshold,
            class_offsets,
            seed,
            anchor_start_idx,
            [s[0] for s in stride_hw],
            [s[1] for s in stride_hw],
            use_clippings,
            False,  # image_size_fixed
            image_sizes[0][0].item(),
            image_sizes[0][1].item(),
            "hw",
            255,  # nms_threshold
            4095,  # post_nms_top_k,
            255,  # nms_margin
            -1,
            march,
        ).to(device=anchor.device)

        sort_helper = torch.ops.horizon.bpu_quanti_proposal(
            [d.cpu() for d in data],
            shifted_anchor.cpu(),
            exp_table.cpu(),
            image_sizes.expand(batch_size, 2).float().cpu(),
            num_anchors,
            [num_classes] * len(data),
            input_shifts,
            exp_shift,
            [block_size[0] for block_size in pow_of_2_block_sizes],
            [block_size[1] for block_size in pow_of_2_block_sizes],
            box_filter_threshold,
            False,  # filter_invalid_boxes
            per_class_idx,
            per_class_threshold,
            class_offsets,
            seed,
            anchor_start_idx,
            [s[0] for s in stride_hw],
            [s[1] for s in stride_hw],
            use_clippings,
            False,  # image_size_fixed
            image_sizes[0][0].item(),
            image_sizes[0][1].item(),
            "hw",
            255,  # nms_threshold
            4095,  # post_nms_top_k,
            255,  # nms_margin
            -1,
            march,
        ).to(device=anchor.device)

        # First sort data by location
        branch_idx = sort_helper[:, :, -4]
        anchor_idx = sort_helper[:, :, -3]
        h = sort_helper[:, :, -2]
        w = sort_helper[:, :, -1]

        overall_idx = (
            (branch_idx * anchor_idx.max() + anchor_idx) * h.max() + h
        ) * w.max() + w
        sorted_by_loc_idxs = torch.argsort(overall_idx)

        skip_nms_ret = torch.gather(
            skip_nms_ret,
            1,
            sorted_by_loc_idxs.unsqueeze(-1).expand_as(skip_nms_ret),
        )

        # Then stable sort data by score
        _, sorted_by_score_idxs = torch.ops.horizon.sort(
            skip_nms_ret[:, :, 4], 1, True
        )
        skip_nms_ret = torch.gather(
            skip_nms_ret,
            1,
            sorted_by_score_idxs.unsqueeze(-1).expand_as(skip_nms_ret),
        )

        valid_mask = torch.ops.horizon.quanti_nms(
            skip_nms_ret[:, :, :4],
            skip_nms_ret[:, :, 4],
            skip_nms_ret[:, :, 5],
            nms_threshold,
            nms_margin,
            march,
        )

        ret_list: List[Tuple[Tensor, Tensor, Tensor]] = []
        for per_image_ret, per_image_mask in zip(skip_nms_ret, valid_mask):
            valid = torch.logical_and(
                per_image_ret[:, -1] >= 0, per_image_mask
            )
            per_image_ret = per_image_ret[valid]
            splited_ret: Tuple[Tensor, Tensor, Tensor] = (
                per_image_ret[:, :4].to(dtype=torch.int16),
                per_image_ret[:, 4].to(dtype=torch.int8),
                per_image_ret[:, 5],
            )
            ret_list.append(splited_ret)
    else:
        if march in ("bernoulli", "bernoulli2"):
            assert not use_stable_sort
        else:
            assert use_stable_sort

        ret = torch.ops.horizon.bpu_quanti_proposal(
            [d.cpu() for d in data],
            shifted_anchor.cpu(),
            exp_table.cpu(),
            image_sizes.expand(batch_size, 2).float().cpu(),
            num_anchors,
            [num_classes] * len(data),
            input_shifts,
            exp_shift,
            [block_size[0] for block_size in block_sizes],
            [block_size[1] for block_size in block_sizes],
            box_filter_threshold,
            per_class_idx,
            per_class_threshold,
            class_offsets,
            seed,
            anchor_start_idx,
            [s[0] for s in stride_hw],
            [s[1] for s in stride_hw],
            use_clippings,
            False,
            image_sizes[0][0].item(),
            image_sizes[0][1].item(),
            "hw",
            nms_threshold,
            post_nms_top_k,
            nms_margin,
            -1,
            march,
        ).to(device=anchor.device)

        ret_list: List[Tuple[Tensor, Tensor, Tensor]] = []
        for per_image_ret in ret:
            valid = per_image_ret[:, -1] >= 0
            per_image_ret = per_image_ret[valid]
            splited_ret: Tuple[Tensor, Tensor, Tensor] = (
                per_image_ret[:, :4].to(dtype=torch.int16),
                per_image_ret[:, 4].to(dtype=torch.int8),
                per_image_ret[:, 5],
            )
            ret_list.append(splited_ret)

    return ret_list


def _conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    output_padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    groups: int,
    padding_mode: str,
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    convert_ret = _conv_convert_int_params(
        input_scale,
        weight,
        weight_scale,
        weight_dtype,
        bias,
        bias_scale,
        bias_dtype,
        scale,
        dtype,
        sumin_scale,
        True,
        groups,
        march,
    )

    filters = weight.size()[1] * groups
    kernel_size = (weight.size()[2], weight.size(3))
    if march == "bernoulli":
        (
            bpu_weight,
            bpu_weight_shift,
            bpu_bias,
            bpu_bias_shift,
            bpu_input_shift,
            bpu_output_shift,
            bpu_sumin_shift,
            dequant_output_scale,
            _,
        ) = convert_ret
        if not (
            torch.all(bpu_input_shift + bpu_weight_shift - bpu_bias_shift >= 0)
            and torch.all(
                bpu_input_shift + bpu_weight_shift - bpu_sumin_shift >= 0
            )
            and torch.all(
                bpu_input_shift + bpu_weight_shift - bpu_output_shift >= 0
            )
        ):
            warnings.warn(
                "Not support bias/sumin right shift or output left shift"
                "on Bernoulli hardware, which may cause accuracy mismatch"
            )
        conv_ret = torch.ops.horizon.bpu_scale_quanti_deconvolution_with_shift(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_input_shift.item(),
            bpu_output_shift.item(),
            bpu_bias_shift,
            bpu_weight_shift,
            bpu_sumin_shift.item(),
            True,  # use_bias
            filters,  # filters
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,
            march,
        )
    else:
        """
        int-conv_transpose_2d
        Calculation formula:
        * f <- convolution
        * x <- feature,  w <- weight,  e <- sumin, b <- bias
        * y = saturate8(saturate16(f(x, w) + (b << bias_left_shift) +
            truncate16(e << sumin_left_shift) * sumin_scale))
            >> accu_right_shift) * output_scale >> output_right_shift)

        """
        (
            bpu_weight,
            bpu_bias,
            bpu_bias_lshift,
            bpu_escale,
            bpu_escale_lshift,
            bpu_oscale,
            bpu_accu_rshift,
            bpu_output_rshift,
            dequant_output_scale,
        ) = convert_ret
        conv_ret = torch.ops.horizon.bpu_scale_quanti_deconvolution(
            input,
            bpu_weight,
            bpu_bias,
            sumin if sumin is not None else torch.zeros(1).to(input),
            bpu_oscale,
            bpu_accu_rshift,
            bpu_bias_lshift,
            bpu_output_rshift,
            bpu_escale,
            bpu_escale_lshift,
            True,  # use_bias,
            filters,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            activation,
            groups,
            True if sumin is not None else False,  # elementwise_input,
            True
            if dtype == "qint32"
            else False,  # disable_output_quantization
            dtype,  # out_quanti_type,
            march,
        )
    return conv_ret, dequant_output_scale


def _multi_scale_roi_align(
    # input
    features: List[Tensor],
    boxes: List[Tensor],  # Tensor shape can be [B, N, 6] or [N, 4]
    # roi_align parameters
    output_size: BroadcastingList2[int],
    spatial_scale: List[float],
    sampling_ratio: int,
    aligned: bool,
    interpolate_mode: str,
    # rois selection parameters
    canonical_box_size: int,
    canonical_level: int,
    box_clip_ratio: Optional[Tuple[float, float, float, float]],
    # march
    march: str,
) -> Tensor:
    output_size = _pair(output_size)

    # if no boxes, return empty
    if len(boxes) == 0:
        return torch.empty(
            (0, features[0].shape[1]) + output_size,
            device=features[0].device,
            dtype=features[0].dtype,
        )

    # for rcnn_post_process output compatability
    # convert [Tensor[B, N, 6]] to [Tensor[N, 4] * B]
    if len(boxes) == 1 and boxes[0].ndim == 3:
        boxes = list(torch.unbind(boxes[0][:, :, :4], dim=0))

    if box_clip_ratio is not None:
        boxes = _bbox_clip(boxes, box_clip_ratio)

    if len(spatial_scale) == 1:
        return _roi_align_list(
            features[0],
            boxes,
            output_size,
            spatial_scale[0],
            sampling_ratio,
            aligned,
            interpolate_mode,
            march,
        )

    # convert boxes from List[Tensor[L, 4]] to Tensor[M, 5]
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

    # Bernoulli2 or Bernoulli roi_align only support fixed point box input
    # Bayes roi_align only support float box input
    # No influence when invoking roi_align,
    # BUT affect feature level mapping computation!!!
    if not boxes[0].is_floating_point():
        boxes = [box * 0.25 for box in boxes]

    box_sizes = torch.sqrt(
        torch.cat(
            [
                ((each_box[:, 2] - each_box[:, 0]) + 1)
                * ((each_box[:, 3] - each_box[:, 1]) + 1)
                for each_box in boxes
            ]
        )
    )
    # Eqn.(1) in FPN paper
    mapped_levels = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size) + 1e-8
    )
    levels = -torch.log2(
        torch.tensor(spatial_scale, device=mapped_levels.device)
    )
    mapped_levels = (
        torch.clamp(mapped_levels, min=levels[0], max=levels[-1]).to(
            torch.int64
        )
        - levels[0]
    )
    num_boxes = rois.size(0)
    num_channels = features[0].shape[1]

    dtype, device = features[0].dtype, features[0].device
    n = features[0].size(0)
    result = torch.zeros(
        (num_boxes, num_channels) + output_size, dtype=dtype, device=device
    )

    for level, scale in enumerate(spatial_scale):
        indexs = torch.where(mapped_levels == level)[0]
        rois_per_level = rois[indexs]

        # convert rois:Tensor[L, 5] to box_lists: List[Tensor[M,4]]
        # to avoid jit.script error when invoking roi_align_tensor here
        box_list: List[Tensor] = []
        for i in range(n):
            box_list.append(rois_per_level[rois_per_level[:, 0] == i, 1:])

        result.index_put_(
            (indexs,),
            _roi_align_list(
                features[level],
                box_list,
                output_size,
                scale,
                sampling_ratio,
                aligned,
                interpolate_mode,
                march,
            ),
        )
    return result


def _correlation(
    data1: Tensor,
    data2: Tensor,
    kernel_size: int,
    max_displacement: int,
    stride1: int,
    stride2: int,
    pad_size: int,
    is_multiply: bool,
    scale1: Tensor,
    zero_point1: Tensor,
    dtype1: str,
    scale2: Tensor,
    zero_point2: Tensor,
    dtype2: str,
    inter_scale: Tensor,
    out_scale: Tensor,
    out_zero_point: Tensor,
    out_dtype: str,
    march: str,
) -> Tensor:
    return torch.ops.horizon.bpu_scale_quanti_correlation(
        data1,
        data2,
        scale1,
        scale2,
        inter_scale,
        out_scale,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        pad_size,
        is_multiply,
        out_dtype,
        march,
    )


def _mean(
    x: Tensor,
    dim: int,
    x_scale: Tensor,
    x_zero_point: Tensor,
    x_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    if march == "bernoulli":
        assert (
            x_dtype == "qint8" and dtype == "qint8"
        ), "only support int8 input and output type"
        c = x.shape[1]
        device = x.device
        m, e = torch.ops.horizon.frexp(torch.tensor(1 / c, device=device))
        fake_weight_value = torch.clamp(
            torch.floor(m * 128 + 0.5), -128, 127
        ) * torch.pow(2.0, e - 7)
        weight_scale = (2.0 ** (e - 7)).reshape(1)  # must guarantee dim=1
        weight = torch.full(
            (1, c, 1, 1), fake_weight_value, dtype=torch.float32, device=device
        )

        # use conv to compute
        out, _ = _conv2d(
            x,
            weight,
            torch.zeros(1, dtype=torch.float32).to(device),
            None,
            (1, 1),  # stride
            (0, 0),  # padding
            (1, 1),  # dilation
            1,
            "zeros",
            "",
            x_scale,
            x_zero_point,
            x_dtype,
            weight_scale,
            torch.zeros_like(weight_scale).to(torch.long),
            "qint8",
            torch.ones(1, dtype=torch.float32).to(device),
            torch.zeros(1, dtype=torch.long),
            "qint8",
            None,
            x_zero_point,
            None,
            scale,
            zero_point,
            dtype,
            march,
        )
        return out
    else:
        r = torch.sum(x, dim, True, dtype=torch.int32)
        return _requantize(
            r,
            x_scale,
            x_zero_point,
            "qint32",
            scale * x.shape[dim],
            zero_point,
            dtype,
            march,
        )


def _segment_lut(
    input: Tensor,
    table: Tensor,
    scales: Tensor,
    beta: Tensor,
    left_shift: Tensor,
    right_shift: Tensor,
    max: Tensor,
    is_centrosymmetric: bool,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    num_entries_per_table: int,
    march: str,
):
    device = input.device
    return torch.ops.horizon.bpu_segment_lut(
        input.to(torch.int16),
        table.to(device),
        scales.to(device),
        beta.to(device),
        left_shift.to(device),
        right_shift.to(device),
        max.to(device),
        is_centrosymmetric,
        8,
        dtype,
        num_entries_per_table,
        march,
    )


@script_quantized_fn
def _horizon_nn_point_pillars_scatter(
    voxel_features: Tensor, coords: Tensor, sizes: List[int]
) -> Tensor:
    mask = coords[:, -1] >= 0

    voxel_features = voxel_features[mask]
    coords = coords[mask]

    batch_size = sizes[0]
    channel_dim = voxel_features.size(1)

    hight = sizes[1]
    width = sizes[2]

    canvas = torch.zeros(
        batch_size * hight * width,
        channel_dim,
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )

    index = (
        coords[:, 0] * (hight * width) + coords[:, -2] * width + coords[:, -1]
    ).long()

    canvas[index] = voxel_features

    return canvas.reshape(batch_size, hight, width, channel_dim).permute(
        0, 3, 1, 2
    )


def _point_pillars_scatter(
    voxel_features: Tensor, coords: Tensor, output_shape: List[int]
) -> Tensor:
    return point_pillars_scatter(voxel_features, coords, output_shape)


@script_quantized_fn
def point_pillars_scatter(
    voxel_features: Tensor, coords: Tensor, output_shape: List[int]
) -> Tensor:
    voxel_features = voxel_features.reshape((voxel_features.size(0), -1))

    mask = coords[:, -1] >= 0

    voxel_features = voxel_features[mask]
    coords = coords[mask]

    batch_size = output_shape[0]
    channel_dim = voxel_features.size(1)

    hight = output_shape[-2]
    width = output_shape[-1]

    canvas = torch.zeros(
        batch_size * hight * width,
        channel_dim,
        dtype=voxel_features.dtype,
        device=voxel_features.device,
    )

    index = (
        coords[:, 0] * (hight * width) + coords[:, -2] * width + coords[:, -1]
    ).long()

    canvas[index] = voxel_features

    return canvas.reshape(batch_size, hight, width, channel_dim).permute(
        0, 3, 1, 2
    )


def _channel_shuffle(input: Tensor, groups: int):
    batch_size, channels, height, width = input.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    input = input.contiguous()
    input = input.view(batch_size, groups, channels_per_group, height, width)
    input = input.transpose(1, 2).contiguous()

    input = input.view(batch_size, channels, height, width)

    return input


def _prelu(
    input: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tensor:
    # prelu out = max(input, 0) + weight * min(input, 0)

    positive_out = input.clamp_min(0)
    positive_out = _requantize(
        positive_out,
        input_scale,
        input_zero_point,
        input_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )

    negative_out = weight.reshape(-1, 1, 1) * input.clamp_max(0).to(
        torch.int32
    )
    negative_out = _requantize(
        negative_out.to(torch.int32),
        input_scale * weight_scale,
        input_zero_point,
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )

    return positive_out + negative_out


def _window_partition(x: Tensor, window_size: int):
    """Partition window.

    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = x.shape
    x = x.view(
        b, h // window_size, window_size, w // window_size, window_size, c
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )
    return windows


def _window_reverse(windows: Tensor, window_size: int, h: int, w: int):
    """Reverse window.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b, h // window_size, w // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def _linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    sumin: Optional[Tensor],
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    bias_scale: Tensor,
    bias_zero_point: Tensor,
    bias_dtype: str,
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    conv_input = input.reshape(-1, input.shape[-1], 1, 1)
    conv_weight = weight.reshape(list(weight.shape) + [1, 1])
    if sumin is not None:
        sumin = sumin.reshape(-1, sumin.shape[-1], 1, 1)
    out, dequant_out_scale = _conv2d(
        input=conv_input,
        weight=conv_weight,
        bias=bias,
        sumin=sumin,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        padding_mode="zeros",
        activation=activation,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        input_dtype=input_dtype,
        weight_scale=weight_scale,
        weight_zero_point=torch.zeros_like(weight_scale).to(torch.long),
        weight_dtype=weight_dtype,
        bias_scale=bias_scale,
        bias_zero_point=bias_zero_point,
        bias_dtype=bias_dtype,
        sumin_scale=sumin_scale,
        sumin_zero_point=sumin_zero_point,
        sumin_dtype=sumin_dtype,
        scale=scale,
        zero_point=zero_point,
        dtype=dtype,
        march=march,
    )
    out_shape = list(input.shape)[:-1] + [weight.shape[0]]
    return out.reshape(out_shape), dequant_out_scale


def _rle(input: Tensor, dtype: torch.dtype) -> List[Tensor]:
    assert not input.is_floating_point(), "rle only works on int values"
    assert (
        dtype == torch.int8 or dtype == torch.int16
    ), "Only support torch.int8 or torch.int16 dtype in rle"
    flatten_input = input.flatten(start_dim=1).cpu()

    min = -128 if dtype == torch.int8 else -32768
    max = 127 if dtype == torch.int8 else 32767
    max_num = max - min
    assert (
        input.min() >= min and input.max() <= max
    ), "input data range exceeds {} range".format(dtype)
    result: List[Tensor] = []
    n, len = flatten_input.size()
    for i in range(n):
        num = 0
        src_index = 0
        per_batch_result: List[int] = []
        # process per batch
        while src_index < len:
            repeat = 1
            data = flatten_input[i][src_index].item()

            # get the repeat times of data
            src_index += 1
            while src_index < len and flatten_input[i][src_index] == data:
                src_index += 1
                repeat += 1

            # process repeat times exceed max_num limit
            while repeat > max_num:
                per_batch_result.append(data)
                per_batch_result.append(max_num)
                repeat -= max_num
                num += 1
            per_batch_result.append(data)
            per_batch_result.append(repeat)
            num += 1
        # num may larger than 255 or 65535
        result.append(torch.tensor([num] + per_batch_result))

    return result


def _generate_warp_coord_for_deform_conv(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    device: torch.device,
):
    kernel_coord = torch.arange(
        0, kernel_size * dilation, dilation, dtype=torch.int16, device=device
    )
    conv_coord = torch.arange(
        -padding,
        input_size + padding - kernel_coord[-1],
        stride,
        dtype=torch.int16,
        device=device,
    )
    abs_coord = (
        conv_coord.reshape(-1, 1) + kernel_coord.reshape(1, -1)
    ).flatten()

    return abs_coord


def _deform_conv2d(
    input: Tensor,
    offset: Tensor,
    mask: Optional[Tensor],
    sumin: Optional[Tensor],
    weight: Tensor,
    bias: Tensor,
    stride: BroadcastingList2[int],
    padding: BroadcastingList2[int],
    dilation: BroadcastingList2[int],
    activation: str,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    offset_scale: Tensor,
    offset_zero_point: Tensor,
    offset_dtype: str,
    mask_scale: Optional[Tensor],
    mask_zero_point: Optional[Tensor],
    mask_dtype: Optional[str],
    sumin_scale: Optional[Tensor],
    sumin_zero_point: Optional[Tensor],
    sumin_dtype: Optional[str],
    weight_scale: Tensor,
    weight_zero_point: Tensor,
    weight_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
) -> Tuple[Tensor, Tensor]:
    device = input.device
    batch_size, in_channel, in_h, in_w = (
        input.size(0),
        input.size(1),
        input.size(2),
        input.size(3),
    )
    out_h, out_w = offset.size(2), offset.size(3)
    kernel_size = weight.size(2), weight.size(3)
    groups = in_channel // weight.size(1)

    # [(y, x), out_h * kernel_h, out_w * kernel_w]
    base_grid = torch.stack(
        torch.meshgrid(
            _generate_warp_coord_for_deform_conv(
                in_h,
                kernel_size[0],
                stride[0],
                padding[0],
                dilation[0],
                device,
            ),
            _generate_warp_coord_for_deform_conv(
                in_w,
                kernel_size[1],
                stride[1],
                padding[1],
                dilation[1],
                device,
            ),
            # ignore indexing because 1.9.1 do not have this param
            # but in the future the default behaviour will be changed to "xy"
            # indexing="ij",
        ),
        dim=0,
    )

    # Compute coord_shift for warp.
    h = in_h if in_h > out_h else out_h
    w = in_w if in_w > out_w else out_w

    max_coord = h if h > w else w
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = min(coord_shift, 8)
    coord_shift = coord_shift if coord_shift > 0 else 0

    grid_scale = torch.tensor(
        1.0 / (1 << coord_shift),
        dtype=torch.float,
        device=base_grid.device,
    ).reshape(1)

    # [batch_size, offset_group, (y, x), out_h * kernel_h, out_w * kernel_w]
    offset = (
        offset.reshape(
            offset.size(0),
            -1,
            kernel_size[0],
            kernel_size[1],
            2,
            out_h,
            out_w,
        )
        .permute(0, 1, 4, 5, 2, 6, 3)
        .reshape(
            offset.size(0),
            -1,
            2,
            out_h * kernel_size[0],
            out_w * kernel_size[1],
        )
    )
    offset_group = offset.size(1)

    # [batch_size, offset_group, (y, x), out_h * kernel_h, out_w * kernel_w]
    grid = _add(
        base_grid,
        offset,
        torch.ones_like(offset_scale),
        offset_scale,
        offset_zero_point,
        offset_zero_point,
        "qint16",
        offset_dtype,
        grid_scale,
        offset_zero_point,
        "qint16",
        march,
    )

    # [batch_size * offset_group, in_channel,
    # out_h * kernel_h, out_w * kernel_w]
    feature = torch.ops.horizon.bpu_quanti_grid_sample(
        input,
        grid,
        "bilinear",
        "zeros",
        True,
        coord_shift,
        march,
    )

    if offset_group > 1:
        feature = feature.reshape(
            batch_size,
            offset_group,
            offset_group,
            in_channel // offset_group,
            out_h * kernel_size[0],
            out_w * kernel_size[1],
        )
        feature = torch.cat(
            [feature[:, i, i, :, :, :] for i in range(offset_group)], dim=1
        )

    output, out_scale = _conv2d(
        feature,
        weight,
        bias,
        None if sumin is None else sumin,
        kernel_size,  # stride
        (0, 0),  # padding
        (1, 1),  # dilation
        groups,
        "zeros",
        activation,
        input_scale,
        input_zero_point,
        input_dtype,
        weight_scale,
        weight_zero_point,
        weight_dtype,
        input_scale,  # bias_scale,
        input_zero_point,  # bias_zero_point,
        input_dtype,  # bias_dtype,
        sumin_scale,
        sumin_zero_point,
        sumin_dtype,
        scale,
        zero_point,
        dtype,
        march,
    )

    return output, out_scale


def _voxelization(
    points: Tensor,
    voxel_size: Tensor,
    pc_range: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert points(N, >=3) to voxels.

    Args:
        points: (N, ndim), points[:, :3] contain xyz points
            and points[:, 3:] contain other information such as reflectivity.
        voxel_size: (3,), xyz, indicate voxel size.
        pc_range: (6,), indicate voxel range, format:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points: Indicate maximum points contained in a voxel.
        max_voxels: Indicate maximum voxels this function create.
            you should shuffle points before call this function because
            max_voxels may drop some points.

    Returns:
        voxels: (M, max_points, ndim) Only contain points.
        coordinates: (M, 3) coordinates in zyx format.
        num_points_per_voxel: (M,) Number of points in per voxel.
    """

    ndim = 3

    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size  # (x, y, z)
    grid_size = torch.round(grid_size).type(torch.int32)
    voxel_map_shape: List[int] = grid_size.type(torch.int64).tolist()
    voxel_map_shape = voxel_map_shape[::-1]  # (z, y, x)

    voxels = torch.zeros(
        (max_voxels, max_points_per_voxel, points.shape[-1]),
        dtype=points.dtype,
    )
    coors = torch.full(
        (max_voxels, 3),
        -1,
        dtype=torch.int32,
    )
    num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int32)

    voxel_num = torch.zeros(1, dtype=torch.int64)
    voxel_num = torch.ops.horizon.voxelization(
        points.cpu(),
        voxel_size.cpu(),
        pc_range.cpu(),
        voxels.cpu(),
        coors.cpu(),
        num_points_per_voxel.cpu(),
        max_points_per_voxel,
        max_voxels,
        ndim,
    )

    if use_max:
        # for deploy, use max_voxels
        out_num = max_voxels
    else:
        out_num = voxel_num.item()

    coors = coors[:out_num].to(points.device)
    voxels = voxels[:out_num].to(points.device)
    num_points_per_voxel = num_points_per_voxel[:out_num].to(points.device)
    return (voxels, coors, num_points_per_voxel)


def _point_pillars_preprocess(
    points_list: List[Tensor],
    pc_range: Tensor,
    voxel_size: Tensor,
    max_voxels: int,
    max_points_per_voxel: int,
    use_max: bool,
    norm_range: Tensor,
    norm_dims: Tensor,
) -> Tuple[Tensor, Tensor]:
    voxel_lst: List[Tensor] = []
    coors_lst: List[Tensor] = []
    num_points_per_voxel_lst: List[Tensor] = []
    for points in points_list:
        # voxelize per points, for batch_size > 1
        voxels, coors, num_points_per_voxel = _voxelization(
            points,
            voxel_size=voxel_size,
            pc_range=pc_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            use_max=use_max,
        )
        voxel_lst.append(voxels)
        coors_lst.append(coors)
        num_points_per_voxel_lst.append(num_points_per_voxel)

    voxel_feature = torch.cat(voxel_lst, dim=0)
    num_points_per_voxel = torch.cat(num_points_per_voxel_lst, dim=0)

    # Pad first element of coord according the index in batch_data.
    # Example:
    #   batch_data = [data1, data2], and batch_size = 2,
    #   batch_data.index(data1) = 0, batch_data.index(data2) = 1,
    #   for data1:  coord (z, y, x) --> Pad 0 --> coord (0, z, y, x)
    #   for data2:  coord (z, y, x) --> Pad 1 --> coord (1, z, y, x)
    coors_batch: List[Tensor] = []
    for i, coor in enumerate(coors_lst):
        coor_pad = F.pad(coor, (1, 0), mode="constant", value=float(i))
        coors_batch.append(coor_pad)
    coors_batch = torch.cat(coors_batch, dim=0).int()

    features = _voxel_feature_encoder(
        features=voxel_feature,
        num_points_in_voxel=num_points_per_voxel,
        norm_range=norm_range,
        norm_dims=norm_dims,
    )

    return features, coors_batch


def _get_paddings_indicator(
    actual_num: Tensor, max_num: int, axis: int = 0
) -> Tensor:
    """Create boolean mask by actual number of a padded tensor.

    This function helps to identify pillars where there's too little data.

    Example:

    actual_num = [[3,3,3,3,3]] (5 pillars, each contains 3 lidar points)
    max_num: 4 (turns to [[0, 1, 2, 3, 4]])
    will return: [[T, T, T, F, F]]

    Args:
        actual_num: (N,M), where N is batch size and M is
            total number of pillars. In certain cases N can be omitted.
        max_num: max number of points allowed in a pillar.
        axis: axis position. Defaults to 0.

    Returns:
        paddings_indicator: indicates where the tensor should be padded.
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape: List[int] = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


def _voxel_feature_encoder(
    features: Tensor,
    num_points_in_voxel: Tensor,
    norm_range: Tensor,
    norm_dims: Tensor,
) -> Tensor:
    norm_dims_num = norm_dims.size(0)
    if norm_dims_num > 0:
        assert norm_dims_num <= len(norm_range) // 2, (
            f"`len(norm_dims)` cannot greater than `len(norm_range) // 2`,"
            f"but get len(norm_dims)={norm_dims_num}, len(norm_range):{len(norm_range)}"  # noqa E501
        )
        assert norm_dims.max() < len(norm_range) // 2, (
            f"Max value of norm_dims cannot greater than `len(norm_range) // 2`,"  # noqa E501
            f"but get max(norm_dims)={norm_dims.max()}, len(norm_range)//2:{len(norm_range)//2}"  # noqa E501
        )
        start = norm_range[norm_dims]
        end = norm_range[norm_dims + len(norm_range) // 2]
        features[:, :, norm_dims] = features[:, :, norm_dims] - start
        features[:, :, norm_dims] = features[:, :, norm_dims] / (end - start)

    # The feature decorations were calculated without regard to whether
    # pillar was empty. Need to ensure that empty pillars remain set to
    # zeros.
    voxel_count = features.shape[1]
    mask = _get_paddings_indicator(num_points_in_voxel, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask

    features = features.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

    return features


def _rcnn_post_process(
    boxes: List[Tensor],
    scores: Tensor,
    deltas: Tensor,
    image_sizes: Optional[Tensor],
    fixed_image_h: int,
    fixed_image_w: int,
    nms_threshold: float,
    box_filter_threshold: float,
    num_classes: int,
    post_nms_top_k: int,
    delta_mean: List[float],
    delta_std: List[float],
    march: str,
) -> Tuple[Tensor, Tensor]:
    """Post Process of RCNN output.

    Given bounding boxes and corresponding scores and deltas,
    decodes bounding boxes and performs NMS. In details, it consists of:
    - Argmax on multi-class scores
    - Filter out those belows the given threshold
    - Non-linear Transformation,
      convert box deltas to original image coordinates
    - Bin-sort remaining boxes on score
    - Apply class-awared NMS and return the firstnms_output_box_num of boxes

    Args:
        boxes: list of box of shape [box_num, (x1, y1, x2, y2)],
                bbox must be quantized.
        scores: shape is [num_batch * num_box, num_classes + 1, 1, 1,],
                dtype is float32
        deltas: shape is [num_batch * num_box, (num_classes + 1) * 4,
                1, 1,], dtype is float32
        image_sizes: shape is [num_batch, 2], dtype is int32, can be None.
                if None, fixed_image_h and fixed_image_w must be provided
        fixed_image_h: height of the original images, rewriten by image_sizes
        fixed_image_w: width of the original images, rewriten by image_sizes
        nms_threshold: bounding boxes of IOU greater than nms_threshold
                will be suppressed
        box_filter_threshold: bounding boxes of scores after softmax less than
                box_filter_threshold will be discarded
        num_classes: total number of classes
        post_nms_top_k: number of bounding boxes after NMS
        delta_mean: a float list of size 4
        delta_std: a float list of size 4
        march: March

    Returns:
        output: int output. shape is [num_batch, post_nms_top_k, 6],
                dtype is int16
                one bbox has 6 numbers [x1, y1, x2, y2, score, class_index]
                if the output boxes number is less than `post_nms_top_k`,
                they are padded with -1
        output_float: float output. same shape but dtype is float32.
                if the output boxes number is less than `post_nms_top_k`,
                they are padded with -1.0
    """

    # if num_batch == 0, return empty tensor
    if len(boxes) == 0:
        return (
            torch.empty(
                (0, post_nms_top_k, 6),
                device=scores.device,
                dtype=torch.int16,
            ),
            torch.empty(
                (0, post_nms_top_k, 6),
                device=scores.device,
                dtype=torch.float32,
            ),
        )
    # When boxes is DPP output during export hbir
    if boxes[0].dim() == 3:
        boxes = [b.squeeze(0) for b in boxes]
    if boxes[0].size(-1) > 4:
        boxes = [b[:, :4] for b in boxes]
    assert (
        boxes[0].dim() == 2 and boxes[0].shape[-1] == 4
    ), "box should be a 4-value Tensor of (x1, y1, x2, y2)."
    assert (
        scores.dim() == 4
        and scores.shape[1] == num_classes + 1
        and scores.shape[2] == 1
        and scores.shape[3] == 1
    ), f"scores should be shape of [num_batch * num_box, num_classes + 1, 1, 1,], but get {scores.shape}"  # noqa
    assert (
        deltas.dim() == 4
        and deltas.shape[1] == (num_classes + 1) * 4
        and deltas.shape[2] == 1
        and deltas.shape[3] == 1
    ), f"deltas should be shape of [num_batch * num_box, (num_classes + 1) * 4, 1, 1,], but get {deltas.shape}"  # noqa
    if image_sizes is not None:
        assert (
            image_sizes.dim() == 2 and image_sizes.shape[-1] == 2
        ), "image_sizes should be shape of [num_batch, 2]."
    assert len(delta_mean) == 4, "delta_mean should be a list of size 4."
    assert len(delta_std) == 4, "delta_std should be a list of size 4."
    max_box_num = 0
    for box in boxes:
        per_batch_box_num = box.size(0)
        max_box_num = max(max_box_num, per_batch_box_num)

    # if all boxes in a batch are Tensor[0, 4], return tensor filled with -1
    if max_box_num == 0:
        return (
            torch.full(
                (len(boxes), post_nms_top_k, 6),
                fill_value=-1,
                device=scores.device,
                dtype=torch.int16,
            ),
            torch.full(
                (len(boxes), post_nms_top_k, 6),
                fill_value=-1,
                device=scores.device,
                dtype=torch.float32,
            ),
        )
    batched_box = torch.stack(
        [
            # padding boxes in a batch to the max num_box
            torch.cat(
                [
                    box,
                    torch.full(
                        (max_box_num - box.size(0), 4),
                        fill_value=-1,
                        dtype=box.dtype,
                        device=box.device,
                    ),
                ],
                dim=0,
            )
            for box in boxes
        ],
        dim=0,
    )
    # batched_box: [num_batch, num_box, 4],
    # scores, deltas: [num_batch * num_box, ...]
    # make sure their dims match
    assert batched_box.size(0) * batched_box.size(1) == scores.size(
        0
    ), f"boxes dim must match scores, but get boxes dim {batched_box.size(0)} * {batched_box.size(1)}, while scores dim {scores.size(0)}!"  # noqa: E501
    assert batched_box.size(0) * batched_box.size(1) == deltas.size(
        0
    ), f"boxes dim must match deltas, but get boxes dim {batched_box.size(0)} * {batched_box.size(1)}, while deltas dim {deltas.size(0)}!"  # noqa: E501

    output, output_float = torch.ops.horizon.rcnn_post_process(
        batched_box.to(torch.int32),
        scores,
        deltas,
        torch.zeros(len(boxes), 2).to(torch.int32)
        if image_sizes is None
        else image_sizes,
        fixed_image_h,
        fixed_image_w,
        nms_threshold,
        box_filter_threshold,
        num_classes,
        post_nms_top_k,
        delta_mean,
        delta_std,
        True if image_sizes is None else False,
        False,
        march,
    )
    return output, output_float


def _topk(
    input: Tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    march: str,
) -> Tuple[Tensor, Tensor]:
    assert sorted is True, "Only support `sorted=True` for now!"
    output, indices = torch.sort(
        input, dim=dim, descending=largest, stable=True
    )
    output = output.index_select(
        dim=dim, index=torch.arange(k).to(input.device)
    )
    indices = indices.index_select(
        dim=dim, index=torch.arange(k).to(input.device)
    )
    return output, indices.to(torch.int64)


def _bbox_clip(
    boxes: List[Tensor], clip_ratio: Tuple[float, float, float, float]
):
    msg = "total clip ratio in the same direction must be less than 1"
    assert clip_ratio[0] + clip_ratio[2] < 1.0, msg
    assert clip_ratio[1] + clip_ratio[3] < 1.0, msg

    ret: List[Tensor] = []

    for box in boxes:
        x1, y1, x2, y2 = box.float().unbind(dim=-1)
        h = y2 - y1
        w = x2 - x1
        clipped_box = torch.stack(
            [
                x1 + w * clip_ratio[0],
                y1 + h * clip_ratio[1],
                x2 - w * clip_ratio[2],
                y2 - h * clip_ratio[3],
            ],
            dim=-1,
        )

        if not box.is_floating_point():
            clipped_box = torch.ops.horizon.round(clipped_box)

        ret.append(clipped_box.to(box.dtype))

    return ret


def _abs(input: Tensor, overflow_mode: str) -> Tensor:
    if input.dtype == torch.int8:
        quant_max = 127
    elif input.dtype == torch.int16:
        quant_max = 32767
    else:
        raise ValueError("Only support dtype int8 or int16!")
    if overflow_mode == "saturate":
        input = torch.clamp_min(input, -quant_max)
    elif overflow_mode == "trunc":
        pass
    else:
        raise ValueError(
            "Unsupported overflow mode! Only 'saturate' or 'trunc' allowed!"
        )
    return torch.abs(input)


def _ceil(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
):
    res = torch.ops.horizon.bpu_ceil(
        input,
        input_scale,
        input_dtype,
        "qint32",
        march,
    )
    return _requantize(
        res,
        torch.ones_like(input_scale),
        torch.zeros_like(input_scale).to(torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )


def _floor(
    input: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    input_dtype: str,
    scale: Tensor,
    zero_point: Tensor,
    dtype: str,
    march: str,
):
    res = torch.ops.horizon.bpu_scale_requantization(
        input,
        input_scale,
        torch.ones_like(input_scale),
        input_dtype,
        "qint32",
        True,  # pre_right_shift_with_round, invalid for int8/int16 input
        False,  # post_right_shift WITHOUT round, IMPORTANT
        march,
    )
    return _requantize(
        res,
        torch.ones_like(input_scale),
        torch.zeros_like(input_scale).to(torch.long),
        "qint32",
        scale,
        zero_point,
        dtype,
        march,
    )


def _softmax_bernoulli2(
    x: Tensor, table: Tensor, max_value_only: bool, dim: int, march: str
):
    x = x.to(torch.int32)
    x = x - x.max(dim, keepdim=True)[0]
    index = x + 255
    x = torch.take(table, index.to(torch.int64))
    x = x.to(torch.float32) / x.to(torch.int32).sum(dim, True)
    if max_value_only:
        maxv = x.max(dim, True)[0]
        return maxv
    return x
