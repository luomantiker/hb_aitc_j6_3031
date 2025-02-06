import math


def get_blob_shape(blob):
    blob_shape = []
    if blob.shape.dim:
        # In most cases, blob have shape information.
        blob_shape = blob.shape.dim
    elif blob.channels:
        # Some old models use (num, channels, height, width)
        # represent shape information.
        blob_shape = (blob.num, blob.channels, blob.height, blob.width)
    elif len(blob.data) > 0:
        # Some models have neither shape field nor
        # (num, channels, height, width) fields.
        blob_shape = [len(blob.data)]
    else:
        raise ValueError("there is no supported shape information in the blob")
    return blob_shape


# In Caffe the Pooling and Convolution kernel parameters can be
# specified in the following two ways.
#
# 1. Pad, kernel size, and stride are all given as a single value for
#    equal dimensions in height and width or as Y, X pairs.
# 2. The *_h and *_w versions may also be used to specify both
#    spatial dimensions.
#
# example:
#   repeated uint32 kernel_size = 4;
#   optional uint32 kernel_h = 11;
#   optional uint32 kernel_w = 12;
#
# defalut:
#   The padding size; defaults to 0
#   The stride; defaults to 1
def get_kernel_value(optional, repeated, idx, default=None):
    # if *_h and *_w versions is set
    if optional:
        return optional
    if repeated:
        if isinstance(repeated, int):
            return repeated
        repeated = list(repeated)
        if len(repeated) == 1:
            return int(repeated[0])
        if idx >= len(repeated):
            raise ValueError(f"idx out of range:{idx}")
        return repeated[idx]
    # If no parameters are set, the default value is used.
    if default is None:
        raise ValueError("Unable to determine kernel parameter!")
    return default


def convolution_output_shape(input_size, kernel_size, stride, pad, dilation):
    return int((input_size + 2 * pad - dilation * (kernel_size - 1) - 1) / stride) + 1


def deconv_output_shape(input_size, kernel_size, stride, pad, dilation):
    kernel_ext = dilation * (kernel_size - 1) + 1
    return int(stride * (input_size - 1) + kernel_ext - 2 * pad)


def pooling_output_shape(input_size, kernel_size, stride, pad, ceil_mode=False):
    if ceil_mode:
        output_size = math.ceil((input_size + 2 * pad - kernel_size) / stride) + 1
    else:
        output_size = math.floor((input_size + 2 * pad - kernel_size) / stride) + 1

    if pad > 0 and (output_size - 1) * stride >= input_size + pad:
        # In some cases, caffe will correct the output shape after
        # the shape inference.
        # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp#L108
        # if (output_size - 1) * stride >= input_size + pad:
        #   output_size -= 1
        # ONNX does not support these cases.
        raise ValueError(
            "ONNX does not support (output_size - 1) * stride "
            ">= input_size + pad in Pooling"
        )
    return output_size


def shape_permute(node):
    permute_order = node.layer.permute_param.order
    input_shape = node.parents[0].output_shapes[0]
    permute_order.extend(i for i in range(len(input_shape)) if i not in permute_order)
    output_shape = [input_shape[i] for i in permute_order]

    return [output_shape]


def shape_upsample(node):
    n, c, h, w = node.parents[0].output_shapes[0]
    scale = node.layer.upsample_param.scale

    if (
        len(node.layer.bottom) == 2
        and node.layer.upsample_param.upsample_h > 0
        and node.layer.upsample_param.upsample_w > 0
    ):
        upsample_h = node.layer.upsample_param.upsample_h
        upsample_w = node.layer.upsample_param.upsample_w
        return [(n, c, upsample_h, upsample_w)]
    return [(n, c, h * scale, w * scale)]


def shape_pass_through(node):
    n, c, h, w = node.parents[0].output_shapes[0]
    try:
        num_output = node.layer.pass_through_param.num_output
        block_height = node.layer.pass_through_param.block_height
        block_width = node.layer.pass_through_param.block_width
    except AttributeError as exc:
        raise AttributeError("cannot get params of PassThrough layer") from exc
    if h % block_height != 0:
        raise ValueError(
            "PassThrough layer shape infer error: height % block_height != 0",
        )
    if w % block_width != 0:
        raise ValueError(
            "PassThrough layer shape infer error: width % block_width != 0",
        )
    new_h = int(h / block_height)
    new_w = int(w / block_width)
    new_c = c * block_width * block_height
    if num_output != new_c:
        raise ValueError(
            f"PassThrough layer shape infer error:infer output length:{new_c} "
            f"is not equal to param num_output:{num_output}",
        )
    return [(n, new_c, new_h, new_w)]


def shape_reshape(node):
    try:
        re_shape = node.layer.reshape_param.shape.dim[:]
    except Exception as e:  # noqa: F841
        re_shape = []
    if len(node.parents) != 1:
        raise ValueError("reashape layer can only have one input")
    input_shape = node.parents[0].output_shapes[0]

    num_input = 1  # calculate the number of input elements
    for dim in input_shape:
        num_input *= dim
    if re_shape == []:  # 不指定reshape
        output_shape = [[1, num_input]]
    else:  # 指定了需要reshape的格式
        output_shape = re_shape
        for i in range(len(re_shape)):
            if re_shape[i] == 0:  # 表示该dim和输入保持一个维度
                output_shape[i] = input_shape[i]
        for j in range(len(re_shape)):
            if output_shape[j] == -1:  # 表示该维度需要自动推断
                for d in output_shape:
                    num_input //= d
                output_shape[j] = int(num_input * -1)  # 推断出该-1的维度
    return [output_shape]


def shape_identity(node):
    if len(node.parents) == 0:
        raise ValueError(f"can not get input of {node.name} node")
    if len(node.input_index) == 0:
        return [node.parents[0].output_shapes[0]]
    return [node.parents[0].output_shapes[node.input_index[0]]]


def shape_data(node):
    if node.output_shapes is None:
        raise ValueError(f"can not get output shapes of {node.name} node")
    return node.output_shapes


def shape_concat(node):
    axis = node.layer.concat_param.axis
    output_shape = None
    for bottom in node.layer.bottom:
        for parent in node.parents:
            if parent.type == "Data":
                if parent.name == bottom:
                    # data layer
                    if output_shape is None:
                        output_shape = list(parent.output_shapes[0])
                    else:
                        output_shape[axis] += parent.output_shapes[0][axis]
                    break
            elif bottom in parent.layer.top:
                tmp_list = list(parent.layer.top)
                i = tmp_list.index(bottom)
                tmp_shape = list(parent.output_shapes[i])
                if output_shape is None:
                    output_shape = list(tmp_shape)
                else:
                    output_shape[axis] += tmp_shape[axis]
                break
    return [tuple(output_shape)]


def shape_flatten(node):
    if len(node.parents) != 1:
        raise ValueError(f"{node.name} node has the wrong number of inputs")
    input_shape = node.parents[0].output_shapes[0]
    num_elements = 1
    axis = 1
    if hasattr(node.layer, "flatten_param"):
        axis = node.layer.flatten_param.axis
        end_axis = node.layer.flatten_param.end_axis
        assert axis == 1 or axis == 1 - len(
            input_shape,
        ), "only support axis == 1 now in flatten op"
        assert (
            end_axis == -1 or end_axis == len(input_shape) - 1
        ), "only support end_axis == -1 now in flatten op"

    for i in range(axis, len(input_shape)):
        num_elements *= input_shape[i]
    return [(input_shape[0], num_elements)]


def shape_inner_product(node):
    if len(node.parents) != 1:
        raise ValueError(f"{node.name} node has the wrong number of inputs")
    input_shape = node.parents[0].output_shapes[0]
    axis = node.layer.inner_product_param.axis
    shape = [input_shape[i] for i in range(axis)]
    shape.append(node.layer.inner_product_param.num_output)
    return [shape]


def shape_pooling(node):
    if len(node.parents) != 1:
        raise ValueError(f"{node.name} node has the wrong number of inputs")
    pooling_param = node.layer.pooling_param
    if pooling_param.global_pooling:
        input_shape = node.parents[0].output_shapes[0]
        return [(input_shape[0], input_shape[1], 1, 1)]

    input_shape = node.parents[0].output_shapes[0]
    kernel_height = get_kernel_value(
        pooling_param.kernel_h,
        pooling_param.kernel_size,
        0,
    )
    kernel_height = get_kernel_value(
        pooling_param.kernel_h,
        pooling_param.kernel_size,
        0,
    )
    kernel_width = get_kernel_value(
        pooling_param.kernel_w,
        pooling_param.kernel_size,
        1,
    )
    stride_height = get_kernel_value(
        pooling_param.stride_h,
        pooling_param.stride,
        0,
        default=1,
    )
    stride_width = get_kernel_value(
        pooling_param.stride_w,
        pooling_param.stride,
        1,
        default=1,
    )
    pad_height = get_kernel_value(
        pooling_param.pad_h,
        pooling_param.pad,
        0,
        default=0,
    )
    pad_width = get_kernel_value(
        pooling_param.pad_w,
        pooling_param.pad,
        1,
        default=0,
    )

    if hasattr(pooling_param, "round_mode") and pooling_param.round_mode == 1:
        ceil_mode = False
    else:
        ceil_mode = True

    input_height = input_shape[2]
    input_width = input_shape[3]
    output_height = pooling_output_shape(
        input_height,
        kernel_height,
        stride_height,
        pad_height,
        ceil_mode,
    )
    output_width = pooling_output_shape(
        input_width,
        kernel_width,
        stride_width,
        pad_width,
        ceil_mode,
    )
    if pooling_param.pool == 0 and len(node.get_output_names()) == 2:
        return [
            (input_shape[0], input_shape[1], output_height, output_width),
            (input_shape[0], input_shape[1], output_height, output_width),
        ]
    return [(input_shape[0], input_shape[1], output_height, output_width)]


def shape_convolution(node):
    if len(node.parents) != 1:
        raise ValueError(f"{node.name} node has the wrong number of inputs")
    convolution_param = node.layer.convolution_param

    input_shape = node.parents[0].output_shapes[0]
    kernel_height = get_kernel_value(
        convolution_param.kernel_h,
        convolution_param.kernel_size,
        0,
    )
    kernel_width = get_kernel_value(
        convolution_param.kernel_w,
        convolution_param.kernel_size,
        1,
    )
    stride_height = get_kernel_value(
        convolution_param.stride_h,
        convolution_param.stride,
        0,
        default=1,
    )
    stride_width = get_kernel_value(
        convolution_param.stride_w,
        convolution_param.stride,
        1,
        default=1,
    )
    pad_height = get_kernel_value(
        convolution_param.pad_h,
        convolution_param.pad,
        0,
        default=0,
    )
    pad_width = get_kernel_value(
        convolution_param.pad_w,
        convolution_param.pad,
        1,
        default=0,
    )

    dilation_height = get_kernel_value(False, convolution_param.dilation, 0, default=1)
    dilation_width = get_kernel_value(False, convolution_param.dilation, 1, default=1)
    output_channel = convolution_param.num_output or input_shape[1]

    input_height = input_shape[2]
    input_widht = input_shape[3]
    output_height = convolution_output_shape(
        input_height,
        kernel_height,
        stride_height,
        pad_height,
        dilation_height,
    )
    output_widht = convolution_output_shape(
        input_widht,
        kernel_width,
        stride_width,
        pad_width,
        dilation_width,
    )
    return [(input_shape[0], output_channel, output_height, output_widht)]


def shape_axpy(node):
    x_shape = node.parents[1].output_shapes[0]
    return [x_shape]


def shape_roipooling(node):
    pooled_h = node.layer.roi_pooling_param.pooled_h
    pooled_w = node.layer.roi_pooling_param.pooled_w
    input_shape = node.parents[0].output_shapes[0]
    rois_shape = node.parents[1].output_shapes[0]
    return [(rois_shape[0], input_shape[1], pooled_h, pooled_w)]


def shape_slice(node):
    slice_param = node.layer.slice_param
    axis = slice_param.axis
    slice_points = slice_param.slice_point
    input_shape = node.parents[0].output_shapes[0]
    split_len = []
    if not slice_points:
        output_num = len(node.get_output_names())
        input_shape = node.parents[0].output_shapes[0]
        channels = int(input_shape[axis] / output_num)
        split_len.extend(channels for _ in range(output_num))
    else:
        for i in range(len(slice_points)):
            if i == 0:
                split_len.append(slice_points[i])
                continue
            split_len.append(slice_points[i] - slice_points[i - 1])
        split_len.append(input_shape[axis] - slice_points[-1])

    output_shapes = []
    for length in split_len:
        tmp = list(input_shape)
        tmp[axis] = length
        output_shapes.append(tuple(tmp))
    return output_shapes


def shape_split(node):
    input_shape = node.parents[0].output_shapes[0]
    size = len(node.layer.top)
    return [input_shape for _ in range(size)]


def shape_reduction(node):
    axis = node.layer.reduction_param.axis
    input_shape = node.parents[0].output_shapes[0]
    output_shape = input_shape[:axis]
    return [output_shape]


def shape_psroipooling(node):
    output_dim = node.layer.psroi_pooling_param.output_dim
    group_size = node.layer.psroi_pooling_param.group_size

    rois_shape = node.parents[1].output_shapes[0]
    return [(rois_shape[0], output_dim, group_size, group_size)]


def shape_deconvolution(node):
    # Only 2D deconv is supported now.
    # input shape is N x C x H x W
    # input weight is
    if len(node.parents) != 1:
        raise ValueError(f"{node.name} node has the wrong number of inputs")
    input_shape = node.parents[0].output_shapes[0]

    deconv_param = node.layer.convolution_param
    group = deconv_param.group
    if input_shape[1] % group != 0:
        raise ValueError(
            f"{node.name} node has the wrong number of inputs or group param",
        )

    # Convolution Parameter 'num_output' is output channel
    num_output = deconv_param.num_output or input_shape[1]

    kernel_height = get_kernel_value(deconv_param.kernel_h, deconv_param.kernel_size, 0)
    kernel_width = get_kernel_value(deconv_param.kernel_w, deconv_param.kernel_size, 1)
    stride_height = get_kernel_value(
        deconv_param.stride_h,
        deconv_param.stride,
        0,
        default=1,
    )
    stride_width = get_kernel_value(
        deconv_param.stride_w,
        deconv_param.stride,
        1,
        default=1,
    )
    pad_height = get_kernel_value(deconv_param.pad_h, deconv_param.pad, 0, default=0)
    pad_width = get_kernel_value(deconv_param.pad_w, deconv_param.pad, 1, default=0)
    dilation_height = get_kernel_value(False, deconv_param.dilation, 0, default=1)
    dilation_width = get_kernel_value(False, deconv_param.dilation, 1, default=1)
    output_channel = num_output

    input_height = input_shape[2]
    input_widht = input_shape[3]
    output_height = deconv_output_shape(
        input_height,
        kernel_height,
        stride_height,
        pad_height,
        dilation_height,
    )
    output_width = deconv_output_shape(
        input_widht,
        kernel_width,
        stride_width,
        pad_width,
        dilation_width,
    )
    return [(input_shape[0], output_channel, output_height, output_width)]


def shape_spp(node):
    input_shape = node.parents[0].output_shapes[0]
    return [[input_shape[0], input_shape[1], 1, 1]]


def shape_matmul(node):
    if (len(node.parents[0].output_shapes[0]) != 2) or (
        len(node.parents[1].output_shapes[0]) != 2
    ):
        raise ValueError("ONNX matmul layer need rank of inputs as 2")
    w_1, h_1 = node.parents[0].output_shapes[0]
    w_2, h_2 = node.parents[1].output_shapes[0]
    try:
        dim_1 = node.layer.matmul_param.dim_1
        dim_2 = node.layer.matmul_param.dim_2
        dim_3 = node.layer.matmul_param.dim_3
    except AttributeError as exc:
        raise AttributeError("MatMul layer param error") from exc
    if (dim_1 != w_1) or (dim_2 != h_1) or (dim_2 != w_2) or (dim_3 != h_2):
        raise ValueError(
            "ONNX matmul layer only support the shape of inputs as "
            "[dim_1, dim_2] and [dim_2, dim_3]",
        )
    return [(w_1, h_2)]


def shape_proposal(node):
    cls_shape = node.parents[0].output_shapes[0]
    post_num = node.layer.nms_param.post_nms_topn
    return [(cls_shape[0] * post_num, 5)]


def shape_roi_post_process(node):
    batch_size = node.layer.roi_post_process_param.batch_size
    post_num = node.layer.nms_param.post_nms_topn
    return [(batch_size, post_num, 6)]


def shape_argmax(node):
    input_shape = node.parents[0].output_shapes[0]
    axis = node.layer.argmax_param.axis
    output_shape = ()
    for i, dim in zip(range(len(input_shape)), input_shape):
        if i == axis:
            output_shape += (1,)
        else:
            output_shape += (dim,)
    return [output_shape]


def shape_crop(node):
    input_shape0 = node.parents[0].output_shapes[0]
    input_shape1 = node.parents[1].output_shapes[0]
    axis = node.layer.crop_param.axis

    output_shape = ()
    for i in range(len(input_shape0)):
        if i >= axis:
            output_shape += (input_shape1[i],)
        else:
            output_shape += (input_shape0[i],)
    return [output_shape]


def shape_crelu(node):
    input_shape = node.parents[0].output_shapes[0]
    return [(input_shape[0], 2 * input_shape[1], input_shape[2], input_shape[3])]


def shape_resize(node):
    input_shape = node.parents[0].output_shapes[0]
    resize_param = node.layer.resize_param
    input_height = input_shape[2]
    input_width = input_shape[3]
    pad_beg = node.layer.resize_param.pad_beg
    pad_end = node.layer.resize_param.pad_end
    assert pad_beg == 0 and pad_end == 0, "pad != 0, not support yet"
    height_in_eff = input_height + pad_beg + pad_end
    width_in_eff = input_width + pad_beg + pad_end
    assert (
        resize_param.zoom_factor != 1
        or resize_param.shrink_factor != 1
        or (resize_param.height != 0 and resize_param.width != 0)
    ), (
        "Resize op must have as least one shape param: "
        "[zoom_factor, shrink_factor, (height, width)]"
    )

    if resize_param.height != 0:
        output_height = resize_param.height
        output_width = resize_param.width
    else:
        shrink_factor = resize_param.shrink_factor
        zoom_factor = resize_param.zoom_factor
        output_height = (height_in_eff - 1) / shrink_factor + 1
        output_height += (output_height - 1) * (zoom_factor - 1)
        output_width = (width_in_eff - 1) / shrink_factor + 1
        output_width += (output_width - 1) * (zoom_factor - 1)

    return [(input_shape[0], input_shape[1], int(output_height), int(output_width))]


def shape_lstm(node):
    input_shape = node.parents[0].output_shapes[0]
    batch_num = input_shape[1]
    time_steps = input_shape[0]
    recurrent_param = node.layer.recurrent_param
    num_output = recurrent_param.num_output

    return [(time_steps, batch_num, num_output)]


def shape_continuation_indicator(node):
    param = node.layer.continuation_indicator_param

    return [(param.time_step, param.batch_size)]


def shape_spatial_transformer(node):
    assert len(node.parents) != 0, f"can not get input of {node.name} node"
    input_shape = node.parents[0].output_shapes[0]
    input_num = input_shape[0]
    input_channel = input_shape[1]
    input_height = input_shape[2]
    input_width = input_shape[3]
    output_h = input_height
    output_w = input_width
    if hasattr(node.layer, "st_param"):
        output_h_param = node.layer.st_param.output_H
        output_w_param = node.layer.st_param.output_W
        if output_h_param:
            output_h = output_h_param
        if output_w_param:
            output_w = output_w_param
    return [(input_num, input_channel, output_h, output_w)]
