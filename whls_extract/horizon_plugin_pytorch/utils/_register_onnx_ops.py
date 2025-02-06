import inspect
import logging
from distutils.version import LooseVersion
from numbers import Integral, Real

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

import horizon_plugin_pytorch

logger = logging.getLogger(__name__)


def _unimplemented(op, msg):
    logger.warning(
        f"ONNX export failed on {op} because {msg} not supported",
        extra={"call_times_context": ("message")},
    )


# scale_quanti
@parse_args("v", "v", "v", "i", "i", "i", "b", "b", "b", "s", "s")
def symbolic_quantize(
    g,
    data,
    scale,
    zero_point,
    vector_dim,
    quant_min,
    quant_max,
    saturate,
    in_place,
    compat_mask,
    round_mode,
    march,
):
    num_bits = 8 if quant_max == 127 else 16
    zp_dtype = torch.int8 if num_bits == 8 else torch.int16
    # set zero_point to 0 and make onnx graph concise.
    # "constant" will be folded and its onnx graph is more concise than "cast"
    if vector_dim == -1:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(1, dtype=zp_dtype),
        )
        axis = 1
    else:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(zero_point.type().sizes(), dtype=zp_dtype),
        )
        axis = vector_dim

    return g.op(
        "horizon::HzDequantize",
        g.op(
            "horizon::HzQuantize",
            data,
            scale,
            zero_point,
            bits_i=num_bits,
            axis_i=axis,
        ),
        scale,
        zero_point,
        axis_i=axis,
    )


# scale_quanti_opt
@parse_args("v", "v", "i", "i", "i", "s")
def symbolic_quantize_opt(
    g,
    data,
    scale,
    quant_min,
    quant_max,
    vector_dim,
    round_mode,
):
    num_bits = 8 if quant_max == 127 else 16
    zp_dtype = torch.int8 if num_bits == 8 else torch.int16
    # set zero_point to 0 and make onnx graph concise.
    # "constant" will be folded and its onnx graph is more concise than "cast"
    if vector_dim == -1:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(1, dtype=zp_dtype),
        )
        axis = 1
    else:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(
                data.type().sizes()[vector_dim], dtype=zp_dtype
            ),
        )
        axis = vector_dim

    return g.op(
        "horizon::HzDequantize",
        g.op(
            "horizon::HzQuantize",
            data,
            scale,
            zero_point,
            bits_i=num_bits,
            axis_i=axis,
        ),
        scale,
        zero_point,
        axis_i=axis,
    )


# scale_requanti
@parse_args("v", "v", "v", "v", "v", "i", "s", "s", "b", "b", "s")
def symbolic_requantize(
    g,
    input,
    in_scale,
    out_scale,
    in_zero_point,
    out_zero_point,
    vector_dim,
    input_quanti_type,
    output_quanti_type,
    pre_rshift_with_round,
    post_rshift_with_round,
    march_str,
):
    return g.op(
        "horizon::Requantize",
        input,
        in_scale,
        out_scale,
        in_zero_point,
        out_zero_point,
        vector_dim_i=vector_dim,
        input_quanti_type_s=input_quanti_type,
        output_quanti_type_s=output_quanti_type,
        pre_rshift_with_round_i=pre_rshift_with_round,
        post_rshift_with_round_i=post_rshift_with_round,
        march_str_s=march_str,
    )


# quanti_resize
@parse_args("v", "v", "v", "s", "b", "i", "i", "f", "f", "b", "s")
def symbolic_resize(
    g,
    data,
    scale,
    zero_point,
    mode,
    align_corners,
    out_height,
    out_width,
    ratio_height,
    ratio_width,
    quantized_forward,
    march,
):
    # return g.op(
    #     "::horizon/Resize",
    #     data,
    #     scale,
    #     zero_point,
    #     mode_s=mode,
    #     align_corners_i=align_corners,
    #     out_height_i=out_height,
    #     out_width_i=out_width,
    #     ratio_height_f=ratio_height,
    #     ratio_width_f=ratio_width,
    #     quantized_forward_i=quantized_forward,
    #     march_s=march,
    # )
    coordinate_transformation_mode = (
        "align_corners" if align_corners else "pytorch_half_pixel"
    )

    # torch export bilinear as linear
    if mode == "bilinear":
        mode = "linear"
        logger.warning(
            "bilinear mode resize is not supported by model convert, "
            "will replace it with linear mode. If there is an accuracy "
            "issue, please change resize mode to nearest, cubic or linear",
            extra={"call_times_context": ("message")},
        )

    if ratio_height > 0 and ratio_width > 0:
        return g.op(
            "Resize",
            data,
            g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [1, 1, ratio_height, ratio_width], dtype=torch.float32
                ),
            ),
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,
            mode_s=mode,
        )
    if out_height > 0 and out_width > 0:
        return g.op(
            "Resize",
            data,
            g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
            g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
            g.op(
                "Constant",
                # 1. resize in n, c dim is not supported.
                # 2. can't get output n, c of custom op.
                # 3. dim with -1 value will be inferred by horizon nn.
                value_t=torch.tensor(
                    [-1, -1, out_height, out_width], dtype=torch.int64
                ),
            ),
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,
            mode_s=mode,
        )
    else:
        _unimplemented("resize", "scale factor and size are")


# bgr_to_yuv444
@parse_args("v", "b")
def symbolic_bgr_to_yuv444(g, data, channel_reversal):
    return g.op(
        "horizon::BgrToYuv444",
        data,
        channel_reversal_i=channel_reversal,
    )


# quanti_roi_resize
@parse_args("v", "v", "v", "v", "f", "i", "i", "b", "s", "b", "b", "s")
def symbolic_quanti_roi_resize(
    g,
    featuremap,
    rois,
    scale,
    zero_point,
    spatial_scale,
    out_height,
    out_width,
    aligned,
    interpolate_mode,
    roi_quantized,
    quantized_forward,
    march_str,
):
    return g.op(
        "horizon::RoiResize",
        featuremap,
        rois,
        scale,
        zero_point,
        spatial_scale_f=spatial_scale,
        out_height_i=out_height,
        out_width_i=out_width,
        aligned_i=aligned,
        interpolate_mode_s=interpolate_mode,
        roi_quantized_i=roi_quantized,
        quantized_forward_i=quantized_forward,
        march_str_s=march_str,
    )


# quanti_grid_sample
@parse_args("v", "v", "v", "s", "s", "b", "i", "s")
def symbolic_quanti_grid_sample(
    g,
    in_featuremap,
    in_grid,
    scale,
    mode,
    padding_mode,
    align_corners,
    coord_shift,
    march_str,
):
    return g.op(
        "horizon::GridSample",
        in_featuremap,
        in_grid,
        scale,
        mode_s=mode,
        padding_mode_s=padding_mode,
        align_corners_i=align_corners,
        coord_shift_i=coord_shift,
        march_str_s=march_str,
    )


# round
@parse_args("v")
def symbolic_round(g, data):
    return g.op("horizon::Round", data)


# max_iou_match
@parse_args("v", "v", "v", "f", "f", "b", "f", "b", "s")
def symbolic_max_iou_match(
    g,
    boxes,
    gt_boxes,
    gt_boxes_num,
    pos_iou,
    neg_iou,
    allow_low_quality_match,
    low_quality_match_iou,
    legacy_bbox,
    overlap_type,
):
    return g.op(
        "horizon::MaxIouMatch",
        boxes,
        gt_boxes,
        gt_boxes_num,
        pos_iou_f=pos_iou,
        neg_iou_f=neg_iou,
        allow_low_quality_match_i=allow_low_quality_match,
        low_quality_match_iou_f=low_quality_match_iou,
        legacy_bbox_i=legacy_bbox,
        overlap_type_s=overlap_type,
    )


# ig_region_match
@parse_args("v", "v", "v", "i", "f", "b", "b")
def symbolic_ig_region_match(
    g,
    boxes,
    ig_regions,
    ig_regions_num,
    class_num,
    ig_region_overlap,
    legacy_bbox,
    output_excluding_class_id_0,
):
    return g.op(
        "horizon::IgRegionMatch",
        boxes,
        ig_regions,
        ig_regions_num,
        class_num_i=class_num,
        ig_region_overlap_f=ig_region_overlap,
        legacy_bbox_i=legacy_bbox,
        output_excluding_class_id_0_i=output_excluding_class_id_0,
    )


# sort
@parse_args("v", "i", "b")
def symbolic_sort(g, data, dim, descending):
    return g.op("horizon::Sort", data, dim_i=dim, descending_i=descending)


# nms
@parse_args("v", "v", "f")
def symbolic_nms(g, det, scores, iou_threshold):
    return g.op("horizon::Nms", det, scores, iou_threshold_f=iou_threshold)


@parse_args("v")
def symbolic_functional_relu6(g, data):
    min_val = g.op("Constant", value_t=torch.tensor(0, dtype=torch.float))
    max_val = g.op("Constant", value_t=torch.tensor(6, dtype=torch.float))
    return g.op("Clip", data, min_val, max_val)


@parse_args("v", "v", "i", "i", "i")
def symbolic_grid_sampler(
    g, input, grid, interpolation_mode, padding_mode, align_corners=False
):
    interpolation_mode_list = ["bilinear", "nearest", "bicubic"]
    padding_mode_list = ["zeros", "border", "reflection"]
    return g.op(
        "horizon::GridSample",
        input,
        grid,
        mode_s=interpolation_mode_list[interpolation_mode],
        padding_mode_s=padding_mode_list[padding_mode],
        align_corners_i=align_corners,
    )


@parse_args("v", "i")
def symbolic_squeeze(
    g,
    input,
    dim=None,
):
    if dim is None:
        return g.op("Squeeze", input)

    return g.op("Squeeze", input, axes_i=[dim])


def symbolic_quantized_op(g, *args, **kwargs):
    r"""Refine this docstring in the future.

    Register quantized ops for ONNX. This function will flatten List[Tensor]
    inputs and outputs to single Tensor because autograd function do not
    support List[Tensor] now.

    Note: Must be used with `script_quantized_fn` defined in
          script_quantized_fn.py
    """

    # if torch version == 1.13
    #   g: torch.onnx._internal.jit_utils.GraphContext
    #   args: inputs
    #   kwargs: node attrs
    #
    # if torch version <= 1.11
    #   g: torch._C.Graph
    #   args: (node, inputs)
    #   kwargs: node attrs
    if LooseVersion(torch.__version__) >= LooseVersion("1.13"):
        n = g.original_node
    elif LooseVersion(torch.__version__) >= LooseVersion("1.12"):
        raise RuntimeError(
            f"Unsupport symbolic function definition in torch {torch.__version__}"  # noqa E501
        )
    else:
        n, *args = args

    # args format:
    # (
    #     function name,
    #     tuple of per list lengths of 'List[Tensor]' args in input args,
    #     input args with flatten list inputs,
    # )
    # example:
    #   origin args: func(Tensor, int, [Tensor1, Tensor2], int)
    #   processed args: (func_name, 2, Tensor, int, Tensor1, Tensor2, int)
    fn_name, per_list_lens, *args = args

    if not isinstance(fn_name, str):
        return None

    module = None
    if "." in fn_name:
        # float op forward or segment_lut methods
        module_name, func_name = fn_name.split(".")
        # module is 'self' arg in forward function
        module, *args = args
        if func_name == "forward":
            func = getattr(
                horizon_plugin_pytorch.nn, module_name, None
            ).forward
            fn_name = module_name  # use module_name in onnx
        elif func_name == "_forward":
            func = getattr(
                horizon_plugin_pytorch.nn, module_name, None
            )._forward
            fn_name = module_name  # use module_name in onnx
        elif func_name in (
            "_init_single_table_params",
            "_init_multi_table_params",
        ):
            func = getattr(
                horizon_plugin_pytorch.nn.quantized.SegmentLUT, func_name, None
            )
        else:
            raise ValueError("Unknown qualname {}".format(fn_name))
    else:
        func = getattr(
            horizon_plugin_pytorch.nn.quantized.functional, fn_name, None
        )
    if func is None:
        return None

    (
        arg_names,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = inspect.getfullargspec(func.__original_fn)
    # do not show 'self' arg in onnx
    arg_names = arg_names[1:] if module is not None else arg_names

    arg_idx_mapping = list(range(len(arg_names)))

    # list_of_tensor_arg_idx:
    #   A list of List[Tensor] args indexes in func.
    #   If no List[Tensor] args, will be []
    if func.list_of_tensor_arg_idx:
        # update idx to find args name in origin args
        list_arg_len_map = zip(
            reversed(func.list_of_tensor_arg_idx), reversed(per_list_lens)
        )
        for idx, list_len in list_arg_len_map:
            for _ in range(list_len - 1):
                arg_idx_mapping.insert(idx, idx)

    fn_name_mapping = {
        "grid_sample": "GridSamplePlugin",
        "Correlation": "HzCorrelation",
        "filter": "Filter",
        "point_pillars_preprocess": "HzPointPillarsPreprocess",
        "_horizon_nn_point_pillars_scatter": "HzScatter",
        "_horizon_nn_filter": "HzFilter",
    }
    if fn_name == "topk":
        # Topk cannot be exported through splicing, because torch will optimize
        # the small splicing op into topk and will meeting a error when
        # pass arg "sorted" to symbolic function.
        code = 'g.op("TopK", '
    else:
        code = 'g.op("horizon::{}", '.format(
            fn_name_mapping.get(fn_name, fn_name)
        )

    skip_inputs = {
        "grid_sample": [5, 6, 7, 8],
        "topk": [1, 5],
        "_horizon_nn_filter": [-1],  # input of filter is dynamic
    }.get(fn_name, [])
    skip_inputs = [len(args) + i if i < 0 else i for i in skip_inputs]
    # put Tensor args in the front
    for i, arg in enumerate(args):
        if isinstance(arg, torch._C.Value):
            if i in skip_inputs:
                continue
            code += "args[{}], ".format(i)

    if fn_name == "topk":
        code += (
            'g.op("Constant", value_t=torch.tensor'
            "([args[1]], dtype=torch.int64)), "
        )

    type_to_reg_mapping = {
        Real: "f",
        Integral: "i",
        bool: "i",
        str: "s",
        torch.Tensor: "t",
    }

    def get_type(arg):
        if isinstance(arg, Integral):
            return Integral
        elif isinstance(arg, Real):
            return Real
        elif isinstance(arg, str):  # for QuantDtype
            return str
        else:
            return type(arg)

    # process not Tensor args
    for i, arg in enumerate(args):
        if i in skip_inputs:
            continue
        reg_annt = type_to_reg_mapping.get(get_type(arg), None)
        # process list and tuple of (int, float, bool, s)
        if reg_annt is None:
            if isinstance(arg, (list, tuple)):
                reg_annt = type_to_reg_mapping.get(get_type(arg[0]), None)

        # Not support type arg(and arg is not None) will be converted to str
        if (
            arg is not None
            and reg_annt is None
            and not isinstance(arg, torch._C.Value)
        ):
            logger.warning(
                "FUNCTION '{}' ARG '{}' type is {}, ".format(
                    fn_name, arg_names[arg_idx_mapping[i]], type(arg)
                )
                + "which is not support in ONNX, will be converted to 'str'.",
                extra={"call_times_context": ("message")},
            )
            reg_annt = "s"
            args[i] = str(args[i])

        if reg_annt is not None:
            code += "{}_{}=args[{}], ".format(
                arg_names[arg_idx_mapping[i]], reg_annt, i
            )

    output_nodes = list(n.outputs())

    code = code[:-2] + ", outputs={})".format(len(output_nodes))

    ret = eval(code)

    if isinstance(ret, (list, tuple)):
        list_ret = ret
    else:
        list_ret = [ret]
    for r, node in zip(list_ret, output_nodes):
        r.setType(node.type())

    return ret


def register_all_custom_op_symbolic(opset=11):
    register_custom_op_symbolic(
        "horizon::scale_quanti", symbolic_quantize, opset
    )

    register_custom_op_symbolic(
        "horizon::scale_quanti_opt", symbolic_quantize_opt, opset
    )

    register_custom_op_symbolic(
        "horizon::scale_requanti", symbolic_requantize, opset
    )

    register_custom_op_symbolic(
        "horizon::quanti_resize", symbolic_resize, opset
    )

    register_custom_op_symbolic(
        "horizon::bgr_to_yuv444", symbolic_bgr_to_yuv444, opset
    )

    register_custom_op_symbolic(
        "horizon::quanti_roi_resize",
        symbolic_quanti_roi_resize,
        opset,
    )

    register_custom_op_symbolic("horizon::round", symbolic_round, opset)

    register_custom_op_symbolic(
        "horizon::quanti_grid_sample",
        symbolic_quanti_grid_sample,
        opset,
    )

    register_custom_op_symbolic(
        "horizon::max_iou_match", symbolic_max_iou_match, opset
    )

    register_custom_op_symbolic(
        "horizon::ig_region_match",
        symbolic_ig_region_match,
        opset,
    )

    register_custom_op_symbolic("horizon::sort", symbolic_sort, opset)

    register_custom_op_symbolic("horizon::nms", symbolic_nms, opset)

    register_custom_op_symbolic("::relu6", symbolic_functional_relu6, opset)

    # `torch.nn.functional.grid_sample` calls `torch.grid_sampler`
    register_custom_op_symbolic("::grid_sampler", symbolic_grid_sampler, opset)

    # Prevent torch squeeze from generating if subgraph.
    register_custom_op_symbolic("::squeeze", symbolic_squeeze, opset)

    register_custom_op_symbolic(
        "prim::PythonOp"
        if LooseVersion(torch.__version__) >= LooseVersion("1.11")
        else "::prim_PythonOp",
        symbolic_quantized_op,
        opset,
    )
