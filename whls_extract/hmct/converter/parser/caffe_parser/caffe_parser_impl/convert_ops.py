import logging
import math

import numpy as np

from hmct.ir.onnx_utils import TensorProto, helper, numpy_helper

from .shape_inference import get_blob_shape, get_kernel_value


def convert_scale(node):
    # Scale op may have 1 or 2 inputs in caffe:
    # if there are two inputs:
    #     the inputs may have different dim like (2,3,4,4) vs (2,3)
    #     Mul op in onnx only supports same dims(numpy-like broadcast),
    #     so, we insert a reshape op to expand the dim of scale
    #     e.g. input(3,4,6,6), scale(3,4)-->(3,4,1,1), axis=0
    #     e.g. input(2,4,6,6), scale(4,6)-->(1,4,6,1), axis=1
    #     So, single Scale op will be converted to Reshape + Mul
    #     (reshape is inserted before the second input)
    #     if bias_term == True
    #         an Add op will be inserted after mul
    # if there is one input:
    #     scale will be converted to batch_normalization
    #     if axis=1 and num_axes=1
    #     else scale will be converted to Mul

    scale_axis = node.layer.scale_param.axis
    init_t = []
    input_dim = node.parents[0].output_shapes[0]
    if scale_axis < 0:
        scale_axis += len(input_dim)
    if len(node.parents) > 1:
        second_shape = node.parents[1].output_shapes[0]
        second_input_name = node.get_input_names()[1]

        scale_dim = second_shape
        mul_input_name = node.get_input_names()
        onnx_nodes = []
        if len(scale_dim) != len(input_dim):
            reshape_dim = [1] * scale_axis
            reshape_dim.extend(scale_dim)
            reshape_dim.extend([1] * (len(input_dim) - len(scale_dim) - scale_axis))
            reshape_dim[0] = -1

            reshape_input_name = []
            reshape_input_name.append(second_input_name)
            reshape_dim_name = node.name + "_reshape_dim"
            reshape_input_name.append(reshape_dim_name)
            init_t.append(
                helper.make_tensor(
                    reshape_dim_name,
                    TensorProto.INT64,
                    [len(reshape_dim)],
                    reshape_dim,
                ),
            )
            reshape_name = node.name + "_mul_reshape"
            reshape_node = helper.make_node(
                "Reshape",
                inputs=reshape_input_name,
                outputs=[reshape_name],
                name=reshape_name,
            )
            second_input_name = reshape_name
            onnx_nodes.append(reshape_node)

        if len(mul_input_name) > 1:
            mul_input_name[1] = second_input_name
        else:
            mul_input_name.append(second_input_name)

        if node.layer.scale_param.bias_term:
            mul_out_name = [node.name + "_mul"]
            mul_name = node.name + "_mul"
        else:
            mul_out_name = node.get_output_names()
            mul_name = node.name

        mul_node = helper.make_node(
            "Mul",
            inputs=mul_input_name,
            outputs=mul_out_name,
            name=mul_name,
        )
        onnx_nodes.append(mul_node)

        # bias_term
        if node.layer.scale_param.bias_term:
            bias_input_name = node.name + "_bias"
            bias_data = np.array(node.blobs[0].data, np.float32).reshape(second_shape)
            init_t.append(numpy_helper.from_array(bias_data, bias_input_name))
            bias_input = [mul_out_name[0], bias_input_name]
            add_node = helper.make_node(
                "Add",
                inputs=bias_input,
                outputs=node.get_output_names(),
                name=node.name,
            )
            onnx_nodes.append(add_node)

    else:
        # Scale op only has one input, convert it to batchnorm
        num_axes = node.layer.scale_param.num_axes
        assert (
            scale_axis == 1 and num_axes == 1
        ), "parser error: convert scale op to bn only support axis=1 and num_axes=1"
        second_shape = input_dim[scale_axis]
        input_names = node.get_input_names()
        scale_name = node.name + "_scale"
        input_names.append(scale_name)
        try:
            scale_data = np.array(node.blobs[0].data, np.float32).reshape(second_shape)
        except TypeError:
            # print(f"Warning: Scale op {node.name} only has one input"
            #     + " and there is no blob data in it. Using filler value")
            if node.layer.scale_param.filler.type != "constant":
                ValueError(
                    f"scale op filler do not support "
                    f"{node.layer.scale_param.filler.type} yet"
                )
            fill_val = node.layer.scale_param.filler.value
            fill_val = 1 if fill_val == 0 else fill_val
            scale_data = np.ones(second_shape, np.float32) * fill_val
        init_t.append(numpy_helper.from_array(scale_data, scale_name))
        bias_name = node.name + "_bias"
        if node.layer.scale_param.bias_term:
            bias_data = np.array(node.blobs[1].data, np.float32).reshape(second_shape)
        else:
            bias_data = np.zeros(second_shape, np.float32)
        init_t.append(numpy_helper.from_array(bias_data, bias_name))
        input_names.append(bias_name)

        mean_name = node.name + "_mean"
        input_names.append(mean_name)
        mean_data = np.zeros(second_shape, np.float32)
        init_t.append(numpy_helper.from_array(mean_data, mean_name))
        var_name = node.name + "_var"
        input_names.append(var_name)
        var_data = np.ones(second_shape, np.float32)
        init_t.append(numpy_helper.from_array(var_data, var_name))

        onnx_node = helper.make_node(
            "BatchNormalization",
            inputs=input_names,
            outputs=node.get_output_names(),
            epsilon=0.0,
            name=node.name,
        )
        onnx_nodes = [onnx_node]
    return onnx_nodes, init_t


def convert_permute(node):
    transpose_perm = node.layer.permute_param.order
    onnx_node = helper.make_node(
        "Transpose",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        perm=transpose_perm,
    )
    return [onnx_node]


def convert_concat(node):
    concat_param_axis = node.layer.concat_param.axis
    onnx_node = helper.make_node(
        "Concat",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        axis=concat_param_axis,
    )
    return [onnx_node]


def convert_dropout(node):
    ratio = node.layer.dropout_param.dropout_ratio
    onnx_node = helper.make_node(
        "Dropout",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        ratio=ratio,
    )
    return [onnx_node]


def convert_lrn(node):
    lrn_alpha = node.layer.lrn_param.alpha
    lrn_beta = node.layer.lrn_param.beta
    lrn_bias = node.layer.lrn_param.k
    lrn_size = node.layer.lrn_param.local_size
    onnx_node = helper.make_node(
        "LRN",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        alpha=lrn_alpha,
        beta=lrn_beta,
        bias=lrn_bias,
        size=lrn_size,
    )
    return [onnx_node]


def convert_reshape(node):
    input_name = node.get_input_names()
    reshape_param = node.layer.reshape_param.shape.dim[:]
    reshape_param_name = node.name + "_reshape_param"
    input_name.append(reshape_param_name)
    reshape_init_t = helper.make_tensor(
        reshape_param_name,
        TensorProto.INT64,
        [len(reshape_param)],
        reshape_param,
    )
    reshape_onnx_node = helper.make_node(
        "Reshape",
        inputs=input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [reshape_onnx_node], reshape_init_t


def convert_innerproduct(node, blob_names):
    # there is no innerproduct in onnx,
    # use reshape + gemm to replace innerproduct in caffe
    input_dim = node.parents[0].output_shapes[0]
    onnx_nodes = []
    node_input = node.get_input_names()[0]
    expose_inits = []

    axis = node.layer.inner_product_param.axis
    if len(input_dim) > 2:
        fisrt_dim = 1
        second_dim = 1
        for i in range(axis):
            fisrt_dim *= input_dim[i]

        for i in range(len(input_dim) - axis):
            second_dim *= input_dim[axis + i]

        in_reshape_param_name = "Reshape_in_" + str(node.name)
        in_reshape_node_name = "Reshape_in_" + str(node.name) + "_node"
        in_reshape_node_out = "Reshape_in_" + str(node.name) + "_out"
        in_reshape_param = [-1, second_dim]
        in_reshape_init_t = helper.make_tensor(
            in_reshape_param_name,
            TensorProto.INT64,
            [len(in_reshape_param)],
            in_reshape_param,
        )
        in_reshape_node = helper.make_node(
            "Reshape",
            inputs=[node_input, in_reshape_param_name],
            outputs=[in_reshape_node_out],
            name=in_reshape_node_name,
        )
        expose_inits.append(in_reshape_init_t)
        onnx_nodes.append(in_reshape_node)
        node_input = in_reshape_node_out

    if axis > 1:
        gemm_node_name = "Gemm_" + str(node.name)
        gemm_node_output = gemm_node_name + "_out"
    else:
        gemm_node_name = node.name
        gemm_node_output = node.get_output_names()[0]
    # gemm
    transB = 0 if node.layer.inner_product_param.transpose else 1  # noqa: N806
    gemm_dict = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": transB}
    gemm_inputs = []
    gemm_inputs.append(node_input)
    gemm_inputs.extend(blob_names)
    gemm_node = helper.make_node(
        "Gemm",
        inputs=gemm_inputs,
        outputs=[gemm_node_output],
        name=gemm_node_name,
        **gemm_dict,
    )
    onnx_nodes.append(gemm_node)

    if axis > 1:
        reshape_param_name = "Reshape_out_" + str(node.name)
        reshape_param = node.output_shapes[0]
        reshape_init_t = helper.make_tensor(
            reshape_param_name,
            TensorProto.INT64,
            [len(reshape_param)],
            reshape_param,
        )
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[gemm_node_output, reshape_param_name],
            outputs=node.get_output_names(),
            name=node.name,
        )
        expose_inits.append(reshape_init_t)
        onnx_nodes.append(reshape_node)
    return onnx_nodes, expose_inits


def convert_flatten(node):
    # param axis has different meaning in caffe and onnx, do not support yet
    axis = 1
    flatten_onnx_node = helper.make_node(
        "Flatten",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        axis=axis,
    )
    return [flatten_onnx_node]


def convert_eltwise(node):
    eltwise_param = node.layer.eltwise_param
    inputs = node.get_input_names()
    outputs = node.get_output_names()
    name = node.name
    onnx_nodes = []
    init_tensors_info = []
    if eltwise_param.operation == eltwise_param.SUM:
        if len(eltwise_param.coeff) == 0:
            onnx_nodes.append(
                helper.make_node("Sum", inputs=inputs, outputs=outputs, name=name),
            )
        else:
            if len(eltwise_param.coeff) != 2:
                raise ValueError("Coefficient parameter should be blob-wise.")
            coeff1 = np.array([eltwise_param.coeff[0]], np.float32)
            init_tensors_info.append(numpy_helper.from_array(coeff1, name + "_coeff1"))
            onnx_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=([inputs[0], name + "_coeff1"]),
                    outputs=[name + "_scale1"],
                    name=name + "_scale1",
                ),
            )
            coeff2 = np.array([eltwise_param.coeff[1]], np.float32)
            init_tensors_info.append(numpy_helper.from_array(coeff2, name + "_coeff2"))
            onnx_nodes.extend(
                (
                    helper.make_node(
                        "Mul",
                        inputs=[inputs[1], name + "_coeff2"],
                        outputs=[name + "_scale2"],
                        name=name + "_scale2",
                    ),
                    helper.make_node(
                        "Add",
                        inputs=[name + "_scale1", name + "_scale2"],
                        outputs=outputs,
                        name=name,
                    ),
                )
            )
    elif eltwise_param.operation == eltwise_param.PROD:
        onnx_nodes.append(
            helper.make_node("Mul", inputs=inputs, outputs=outputs, name=name),
        )
    elif eltwise_param.operation == eltwise_param.MAX:
        onnx_nodes.append(
            helper.make_node("Max", inputs=inputs, outputs=outputs, name=name),
        )
    else:
        raise ValueError("unknown eltwise parameter ")
    return onnx_nodes, init_tensors_info


def convert_relu(node):
    negative_slope = node.layer.relu_param.negative_slope
    if negative_slope is not None and negative_slope != 0:
        onnx_node = helper.make_node(
            "LeakyRelu",
            inputs=node.get_input_names(),
            outputs=node.get_output_names(),
            alpha=negative_slope,
            name=node.name,
        )
    else:
        onnx_node = helper.make_node(
            "Relu",
            inputs=node.get_input_names(),
            outputs=node.get_output_names(),
            name=node.name,
        )
    return [onnx_node]


def convert_pooling(node):
    pooling_param = node.layer.pooling_param
    global_pooling = pooling_param.global_pooling
    inputs = node.get_input_names()
    outputs = node.get_output_names()
    if global_pooling:
        if pooling_param.pool == 0:
            # MAX
            onnx_node = helper.make_node(
                "GlobalMaxPool",
                inputs=inputs,
                outputs=outputs,
                name=node.name,
            )
        elif pooling_param.pool == 1:
            # AVG
            onnx_node = helper.make_node(
                "GlobalAveragePool",
                inputs=inputs,
                outputs=outputs,
                name=node.name,
            )
        else:
            raise ValueError(
                "ONNX not supported Pool type [%d]." % (pooling_param.pool),
            )
    else:
        if pooling_param.pool == 0:
            pooling_type = "MaxPool"
        elif pooling_param.pool == 1:
            pooling_type = "AveragePool"
        else:
            raise ValueError(
                "ONNX not supported Pool type [%d]." % (pooling_param.pool),
            )

        if (hasattr(pooling_param, "round_mode") and pooling_param.round_mode == 1) or (
            hasattr(pooling_param, "ceil_mode") and not pooling_param.ceil_mode
        ):
            ceil_mode = False
        else:
            ceil_mode = True

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

        kernel_shape = [kernel_height, kernel_width]
        pads = [pad_height, pad_width, pad_height, pad_width]
        strides = [stride_height, stride_width]

        if pooling_type == "AveragePool":
            onnx_node = helper.make_node(
                pooling_type,
                inputs=inputs,
                outputs=outputs,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                ceil_mode=ceil_mode,
                count_include_pad=1,
                name=node.name,
            )
        else:
            onnx_node = helper.make_node(
                pooling_type,
                inputs=inputs,
                outputs=outputs,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                ceil_mode=ceil_mode,
                name=node.name,
            )

    return [onnx_node]


def convert_convolution(node, blob_names):
    convolution_param = node.layer.convolution_param

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
    group = convolution_param.group or 1

    kernel_shape = [kernel_height, kernel_width]
    pads = [pad_height, pad_width, pad_height, pad_width]
    strides = [stride_height, stride_width]
    dilations = [dilation_height, dilation_width]

    inputs = node.get_input_names()
    outputs = node.get_output_names()
    inputs.extend(blob_names)
    onnx_node = helper.make_node(
        "Conv",
        inputs=inputs,
        outputs=outputs,
        dilations=dilations,
        group=group,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        name=node.name,
    )

    return [onnx_node]


def convert_batch_norm(node, blob_names):
    # ['x', 's', 'bias', 'mean', 'var']
    inputs = node.get_input_names()
    outputs = node.get_output_names()
    inputs.extend(blob_names)
    # epsilon
    eps = node.layer.batch_norm_param.eps
    onnx_node = helper.make_node(
        "BatchNormalization",
        inputs=inputs,
        outputs=outputs,
        epsilon=eps,
        name=node.name,
    )
    return [onnx_node]


def convert_softmax(node):
    axis = node.layer.softmax_param.axis
    shape = node.output_shapes[0]
    axis = axis if axis >= 0 else len(shape) + axis
    if axis == len(shape) - 1 or (
        len(shape) == 4 and axis == 1 and shape[2] == 1 and shape[3] == 1
    ):
        # If axis is not the last dimension
        onnx_node = helper.make_node(
            "Softmax",
            inputs=node.get_input_names(),
            outputs=node.get_output_names(),
            axis=axis,
            name=node.name,
        )
        return [onnx_node]

    # If the softmax.axis is not the last axis of the input data,
    # need to insert transpose operations to ensure computational
    # consistency: transposes the input(node0), do softmax(node1),
    # transposed back(node2).
    if len(shape) != 4:
        raise ValueError(
            f"The rank of Softmax layer {node.name} output should be 4",
        )
    perm = [0, 1, 2, 3]
    perm[3] = perm[axis]
    perm[axis] = 3
    node0_name = node.name + "_T0"
    node1_name = node.name + "_T1"
    node0 = helper.make_node(
        "Transpose",
        inputs=node.get_input_names(),
        outputs=[node0_name],
        perm=perm,
        name=node0_name,
    )
    node1 = helper.make_node(
        "Softmax",
        inputs=[node0_name],
        outputs=[node1_name],
        axis=-1,
        name=node1_name,
    )
    node2 = helper.make_node(
        "Transpose",
        inputs=[node1_name],
        outputs=node.get_output_names(),
        perm=perm,
        name=node.name,
    )
    return [node0, node1, node2]


def convert_pass_through(node):
    block_height = node.layer.pass_through_param.block_height
    block_width = node.layer.pass_through_param.block_width
    assert block_height == block_width, (
        "parser error: convert pass_through op to SpaceToDepth "
        "only support block_width and block_height are equal."
    )
    onnx_node = helper.make_node(
        "SpaceToDepth",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        blocksize=block_height,
        name=node.name,
    )
    return [onnx_node]


def convert_max_unpool(node, upsample_name=None):  # noqa: ARG001
    upsample_param = node.layer.upsample_param
    scale = upsample_param.scale
    kernel_shape = (scale, scale)
    strides = (scale, scale)
    upsample_h = upsample_param.upsample_h
    upsample_w = upsample_param.upsample_w
    onnx_node = helper.make_node(
        "HzMaxUnpool",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        kernel_shape=kernel_shape,
        strides=strides,
        upsample_h=upsample_h,
        upsample_w=upsample_w,
        name=node.name,
        domain="horizon",
    )
    return [onnx_node]


def convert_upsample(node, scale_name, mode):
    input_name = []
    input_name.extend(node.get_input_names())
    input_roi_name = node.name + "_roi_input"
    roi_init = helper.make_tensor(input_roi_name, TensorProto.FLOAT, [0], [])
    input_name.extend((input_roi_name, scale_name))
    onnx_node = helper.make_node(
        "Resize",
        inputs=input_name,
        outputs=node.get_output_names(),
        mode=mode,
        name=node.name,
    )

    return [onnx_node], [roi_init]


def convert_sigmoid(node):
    onnx_node = helper.make_node(
        "Sigmoid",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [onnx_node]


def convert_tanh(node):
    onnx_node = helper.make_node(
        "Tanh",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [onnx_node]


def convert_bias(node):
    # Bias op in caffe supports input and bias has different dim like
    # (2,3,4,4) vs (2,3). Add op in onnx only supports same dims
    # (numpy-like broadcast), so, we insert a reshape op to expand the
    # dim of bias:
    # e.g. input(3,4,6,6), bias(3,4)-->(3,4,1,1), axis=0
    # e.g. input(2,4,6,6), bias(4,6)-->(1,4,6,1), axis=1

    # The number of inputs of Bias may be 1 or 2.
    # if Bias op has 2 inputs:
    #   single Bias op will be converted to Reshape+Add
    # if Bias op has 1 inputs:
    #   single Bias op will be converted to Add
    #   (the shape of Blob will be changed)
    input_dim = node.parents[0].output_shapes[0]
    bias_axis = node.layer.bias_param.axis
    if bias_axis < 0:
        bias_axis += len(input_dim)

    if len(node.parents) > 1:
        reshape_init_t = []
        second_shape = node.parents[1].output_shapes[0]
        second_input_name = node.get_input_names()[1]
        bias_dim = second_shape
        reshape_dim = [1] * bias_axis
        reshape_dim.extend(bias_dim)
        reshape_dim.extend([1] * (len(input_dim) - len(bias_dim) - bias_axis))
        reshape_dim[0] = -1  # dim[0] = -1 is used to support dynamic batch size

        reshape_input_name = []
        reshape_input_name.append(second_input_name)
        reshape_dim_name = node.name + "_reshape_dim"
        reshape_input_name.append(reshape_dim_name)
        reshape_init_t.append(
            helper.make_tensor(
                reshape_dim_name,
                TensorProto.INT64,
                [len(reshape_dim)],
                reshape_dim,
            ),
        )
        reshape_name = node.name + "_reshape"
        reshape_node = helper.make_node(
            "Reshape",
            inputs=reshape_input_name,
            outputs=[reshape_name],
            name=reshape_name,
        )

        add_input_name = node.get_input_names()
        add_input_name[1] = reshape_name

        add_node = helper.make_node(
            "Add",
            inputs=add_input_name,
            outputs=node.get_output_names(),
            name=node.name,
        )
        return [reshape_node, add_node], reshape_init_t

    num_axes = node.layer.bias_param.num_axes
    if num_axes != 1:
        logging.warning("[Warning]: The num_axes of Bias is not equal to 1")
        # TODO: (dongfang.li) handle num_axes not equal to 1
    bias_shape = input_dim
    bias_init_t = []
    bias_blob_name = node.name + "_bias"
    bias_origin_shape = get_blob_shape(node.blobs[0])
    shape_temp = [1] * bias_axis
    shape_temp.extend(bias_origin_shape)
    shape_temp.extend([1] * (len(bias_shape) - len(shape_temp)))
    bias_data = np.reshape(node.blobs[0].data, shape_temp)
    data_temp = np.ones(bias_shape).astype(np.float32)
    bias_data = bias_data * data_temp
    bias_data = bias_data.astype(np.float32)
    bias_init_t.append(numpy_helper.from_array(bias_data, bias_blob_name))
    add_input_name = node.get_input_names()
    add_input_name.append(bias_blob_name)
    add_node = helper.make_node(
        "Add",
        inputs=add_input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [add_node], bias_init_t


def convert_exp(node):
    # caffe layer parameters
    base = node.layer.exp_param.base
    scale = node.layer.exp_param.scale
    shift = node.layer.exp_param.shift

    onnx_node = []
    init_tensors_info = []
    inter_output_name = node.get_input_names()[0]
    if not math.isclose(scale, 1.0):
        node_name = node.name + "_mul_scale"
        first_input_name = inter_output_name
        second_input_name = node.name + "_scale"
        inter_output_name = node.name + "_mul_output"

        mul_node = helper.make_node(
            "Mul",
            inputs=[first_input_name, second_input_name],
            outputs=[inter_output_name],
            name=node_name,
        )
        onnx_node.append(mul_node)

        second_input = np.array([scale], np.float32)
        init_tensors_info.append(
            numpy_helper.from_array(second_input, second_input_name),
        )

    if not math.isclose(shift, 0.0):
        node_name = node.name + "_add_shift"
        first_input_name = inter_output_name
        second_input_name = node.name + "_shift"
        inter_output_name = node.name + "_add_output"

        add_node = helper.make_node(
            "Add",
            inputs=[first_input_name, second_input_name],
            outputs=[inter_output_name],
            name=node_name,
        )
        onnx_node.append(add_node)

        second_input = np.array([shift], np.float32)
        init_tensors_info.append(
            numpy_helper.from_array(second_input, second_input_name),
        )

    node_output_name = node.get_output_names()[0]
    if math.isclose(base, -1.0):
        exp_node = helper.make_node(
            "Exp",
            inputs=[inter_output_name],
            outputs=[node_output_name],
            name=node.name,
        )
        onnx_node.append(exp_node)
    else:
        if base <= 0:
            raise ValueError(
                f"Base of Exp layer {node.name} should be -1 or larger than 0",
            )
        first_input_name = node.name + "_base"
        pow_node = helper.make_node(
            "Pow",
            inputs=[first_input_name, inter_output_name],
            outputs=[node_output_name],
            name=node.name,
        )
        onnx_node.append(pow_node)

        first_input = np.array([base], np.float32)
        init_tensors_info.append(numpy_helper.from_array(first_input, first_input_name))

    return onnx_node, init_tensors_info


def convert_power(node):
    # caffe layer parameters
    power = node.layer.power_param.power
    scale = node.layer.power_param.scale
    shift = node.layer.power_param.shift

    onnx_node = []
    init_tensors_info = []
    inter_output_name = node.get_input_names()[0]
    if not math.isclose(scale, 1.0):
        node_name = node.name + "_mul_scale"
        first_input_name = inter_output_name
        second_input_name = node.name + "_scale"
        inter_output_name = node.name + "_mul_output"

        mul_node = helper.make_node(
            "Mul",
            inputs=[first_input_name, second_input_name],
            outputs=[inter_output_name],
            name=node_name,
        )
        onnx_node.append(mul_node)

        second_input = np.array([scale], np.float32)
        init_tensors_info.append(
            numpy_helper.from_array(second_input, second_input_name),
        )

    if not math.isclose(shift, 0.0):
        node_name = node.name + "_add_shift"
        first_input_name = inter_output_name
        second_input_name = node.name + "_shift"
        inter_output_name = node.name + "_add_output"

        add_node = helper.make_node(
            "Add",
            inputs=[first_input_name, second_input_name],
            outputs=[inter_output_name],
            name=node_name,
        )
        onnx_node.append(add_node)

        second_input = np.array([shift], np.float32)
        init_tensors_info.append(
            numpy_helper.from_array(second_input, second_input_name),
        )

    node_output_name = node.get_output_names()[0]
    second_input_name = node.name + "_power"
    pow_node = helper.make_node(
        "Pow",
        inputs=[inter_output_name, second_input_name],
        outputs=[node_output_name],
        name=node.name,
    )
    second_input = np.array([power], np.float32)
    init_tensors_info.append(numpy_helper.from_array(second_input, second_input_name))
    onnx_node.append(pow_node)
    return onnx_node, init_tensors_info


def convert_axpy(node):
    # F = a * X + Y
    # Shape info:
    # a:  N x C          --> bottom[0]
    # X:  N x C x H x W  --> bottom[1]
    # Y:  N x C x H x W  --> bottom[2]
    # F:  N x C x H x W  --> top[0]
    # from the paper of se-resnet, axpy op works as
    # channel-wise multiplication first, then element-add.

    mul_input_name = node.get_input_names()[:2]
    mul_name = node.name + "_mul"
    mul_node = helper.make_node(
        "Mul",
        inputs=mul_input_name,
        outputs=[mul_name],
        name=mul_name,
    )
    element_add_input_name = node.get_input_names()[2:]
    element_add_input_name.append(mul_name)
    element_add_node = helper.make_node(
        "Add",
        inputs=element_add_input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [mul_node, element_add_node]


def convert_abs_val(node):
    onnx_node = helper.make_node(
        "Abs",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [onnx_node]


def convert_elu(node):
    alpha = node.layer.elu_param.alpha
    onnx_node = helper.make_node(
        "Elu",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        alpha=alpha,
    )
    return [onnx_node]


def convert_log(node):
    # use equation:
    # \log_{base}{(shift+multi*input)} =
    # \frac{\ln(shift+multi*input)}{ln(base)}
    onnx_node = []
    init_tensors_info = []
    node0_name = node.name + "_Mul"
    node1_name = node.name + "_Add"
    node2_name = node.name + "_Log"
    base = node.layer.log_param.base
    scale = node.layer.log_param.scale
    shift = node.layer.log_param.shift
    pre_out_tmp = node.get_input_names()
    base_name = node.name + "_base"
    scale_name = node.name + "_scale"
    shift_name = node.name + "_shift"
    blob_shape = [1]
    if not math.isclose(scale, 1.0):
        pre_out_tmp.append(scale_name)
        scale_arr = np.ones(blob_shape, np.float32) * scale
        init_tensors_info.append(numpy_helper.from_array(scale_arr, scale_name))
        node0 = helper.make_node(
            "Mul",
            inputs=pre_out_tmp,
            outputs=[node0_name],
            name=node0_name,
        )
        pre_out_tmp = [node0_name]
        onnx_node.append(node0)
    if not math.isclose(shift, 0.0):
        pre_out_tmp.insert(0, shift_name)
        shift_arr = np.ones(blob_shape, np.float32) * shift
        init_tensors_info.append(numpy_helper.from_array(shift_arr, shift_name))
        node1 = helper.make_node(
            "Add",
            inputs=pre_out_tmp,
            outputs=[node1_name],
            name=node1_name,
        )
        pre_out_tmp = [node1_name]
        onnx_node.append(node1)
    log_output = node.get_output_names()
    log_output_name = node.name
    if not (math.isclose(base, -1.0) or math.isclose(base, math.e)):
        log_output = [node2_name]
        log_output_name = node2_name
    node2 = helper.make_node(
        "Log",
        inputs=pre_out_tmp,
        outputs=log_output,
        name=log_output_name,
    )
    pre_out_tmp = [node2_name]
    onnx_node.append(node2)
    if not (math.isclose(base, -1.0) or math.isclose(base, math.e)):
        if base <= 0:
            raise ValueError(
                f"Base of Log layer {node.name} should be -1 or larger than 0",
            )
        pre_out_tmp.append(base_name)
        base_arr = np.ones(blob_shape, np.float32) * (1 / math.log(base))
        init_tensors_info.append(numpy_helper.from_array(base_arr, base_name))
        node3 = helper.make_node(
            "Mul",
            inputs=pre_out_tmp,
            outputs=node.get_output_names(),
            name=node.name,
        )
        onnx_node.append(node3)
    return onnx_node, init_tensors_info


def convert_threshold(node):
    # caffe layer parameters
    # TODO(tingcheng): need set threshold default value?
    threshold = node.layer.threshold_param.threshold

    onnx_node = []
    init_tensors_info = []
    inter_output_name = node.get_input_names()[0]
    if not math.isclose(threshold, 0.0):
        node_name = node.name + "_threshold_shift"
        first_input_name = inter_output_name
        second_input_name = node.name + "_shift"
        inter_output_name = node.name + "_shift_output"

        add_node = helper.make_node(
            "Add",
            inputs=[first_input_name, second_input_name],
            outputs=[inter_output_name],
            name=node_name,
        )
        onnx_node.append(add_node)

        second_input = np.array([-threshold], np.float32)
        init_tensors_info.append(
            numpy_helper.from_array(second_input, second_input_name),
        )

    node_name = node.name + "_threshold_sign"
    sign_output_name = node.name + "_sign_output"
    first_input_name = inter_output_name
    sign_node = helper.make_node(
        "Sign",
        inputs=[inter_output_name],
        outputs=[sign_output_name],
        name=node_name,
    )
    onnx_node.append(sign_node)

    # TODO: Clip inputs definations changed in Version 11,
    #       inputs should be reset when opset change to V11.
    min_input_name = node.name + "_clip_min"
    min_input = np.array(0, np.float32)
    init_tensors_info.append(numpy_helper.from_array(min_input, min_input_name))
    clip_input_names = [sign_output_name, min_input_name]
    clip_node = helper.make_node(
        "Clip",
        inputs=clip_input_names,
        outputs=[node.get_output_names()[0]],
        name=node.name,
    )
    onnx_node.append(clip_node)
    return onnx_node, init_tensors_info


def convert_roipooling(node):
    # caffe layer parameters
    pooled_h = node.layer.roi_pooling_param.pooled_h
    pooled_w = node.layer.roi_pooling_param.pooled_w
    spatial_scale = node.layer.roi_pooling_param.spatial_scale

    roipooling_node = helper.make_node(
        "MaxRoiPool",
        inputs=[node.get_input_names()[0], node.get_input_names()[1]],
        outputs=node.get_output_names(),
        pooled_shape=[pooled_h, pooled_w],
        spatial_scale=spatial_scale,
        name=node.name,
    )

    return [roipooling_node]


def convert_prelu(node, slope_shape):
    # FillerParameter 'filler' of PReLU will be ignored in onnx.
    # It's used for the initialization of input tensor 'slope'
    # when it's not  provided.
    # filler = node.layer.prelu_param.filler
    channel_shared = node.layer.prelu_param.channel_shared
    input_dims = node.parents[0].output_shapes[0]
    slope_data = np.array(node.blobs[0].data, np.float32)

    if len(slope_shape) != 1:
        raise ValueError(f"unspport slope shape of PReLU:{slope_shape}")
    if channel_shared:
        if slope_shape[0] != 1:
            raise ValueError(
                f"channel shared slope shape should be 1, but get:{slope_shape[0]}",
            )
        slope_shape[0] = input_dims[1]
        slope_data = np.tile(slope_data, input_dims[1])
    else:
        # slope_shape should equal to input channel
        if slope_shape[0] != input_dims[1]:
            raise ValueError("slope_shape of PReLU should equal to input channel")

    # In onnx: tensor slope should be unidirectional broadcastable to
    # input tensor X
    if len(input_dims) == 4:
        onnx_slope_shape = np.array([1, slope_shape[0], 1, 1], np.int64)
    elif len(input_dims) == 2:
        onnx_slope_shape = np.array([1, slope_shape[0]], np.int64)
    else:
        raise ValueError(f"unspport input dims of prelu:{input_dims}")

    node_input_name = node.get_input_names()[0]
    input_slope_name = node.name + "_slope"
    slope_init_tensor = helper.make_tensor(
        input_slope_name,
        TensorProto.FLOAT,
        onnx_slope_shape,
        slope_data,
    )

    prelu_input_name = [node_input_name, input_slope_name]
    prelu_node = helper.make_node(
        "PRelu",
        inputs=prelu_input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [prelu_node], [slope_init_tensor]


def convert_bnll(node):
    exp_output_name = node.name + "_exp_out"
    exp_name = node.name + "exp"
    exp_node = helper.make_node(
        "Exp",
        inputs=node.get_input_names(),
        outputs=[exp_output_name],
        name=exp_name,
    )

    add_array = np.ones([1], np.float32)  # let onnx 'Add' op multidirectional
    add_second_input = node.name + "_add_one"
    init_tensor = numpy_helper.from_array(add_array, add_second_input)

    add_first_input = exp_output_name
    add_output_name = node.name + "_add_out"
    add_name = node.name + "_add"
    add_node = helper.make_node(
        "Add",
        inputs=[add_first_input, add_second_input],
        outputs=[add_output_name],
        name=add_name,
    )

    log_node = helper.make_node(
        "Log",
        inputs=[add_output_name],
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [exp_node, add_node, log_node], [init_tensor]


def convert_normalize(node, scale_name):
    eps = node.layer.norm_param.eps
    across_spatial = node.layer.norm_param.across_spatial
    channel_shared = node.layer.norm_param.channel_shared
    input_names = node.get_input_names()
    input_names.append(scale_name)
    onnx_node = helper.make_node(
        "HzNormalize",
        inputs=input_names,
        outputs=node.get_output_names(),
        name=node.name,
        domain="horizon",
        eps=eps,
        across_spatial=across_spatial,
        channel_shared=channel_shared,
    )
    return [onnx_node]


def convert_slice(node):
    slice_param = node.layer.slice_param
    axis = slice_param.axis
    slice_points = slice_param.slice_point
    input_shape = node.parents[0].output_shapes[0]
    split = []
    if not slice_points:
        output_num = len(node.get_output_names())
        channels = int(input_shape[axis] / output_num)
        split.extend(channels for _ in range(output_num))
    else:
        for i in range(len(slice_points)):
            if i == 0:
                split.append(slice_points[i])
                continue
            split.append(slice_points[i] - slice_points[i - 1])
        split.append(input_shape[axis] - slice_points[-1])
    onnx_node = helper.make_node(
        "Split",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        axis=axis,
        split=split,
    )
    return [onnx_node]


def convert_mvn(node):
    normalize_variance = node.layer.mvn_param.normalize_variance
    across_channels = node.layer.mvn_param.across_channels
    eps = node.layer.mvn_param.eps

    mvn_node = helper.make_node(
        "HzMeanVarianceNormalization",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        domain="horizon",
        normalize_variance=normalize_variance,
        across_channels=across_channels,
        eps=eps,
    )
    return [mvn_node]


def convert_reduction(node):
    onnx_nodes = []
    init_tensors = []
    reduction_param = node.layer.reduction_param
    axis = reduction_param.axis
    coeff = reduction_param.coeff
    shape = node.parents[0].output_shapes[0]
    nodenames = [node.name]
    outputnames = node.get_output_names()

    # convert neg axis
    if axis < 0:
        axis = len(shape) + axis
    if not math.isclose(coeff, 1.0):
        nodenames.append(node.name + "_reduce")
        outputnames.append(node.name + "_inner")

    axes = list(range(axis, len(shape)))
    if reduction_param.operation == reduction_param.SUM:
        onnx_nodes.append(
            helper.make_node(
                "ReduceSum",
                inputs=node.get_input_names(),
                outputs=[outputnames[-1]],
                name=nodenames[-1],
                axes=tuple(axes),
                keepdims=0,
            ),
        )
    elif reduction_param.operation == reduction_param.ASUM:
        abs_node_name = node.name + "_abs"
        abs_node = helper.make_node(
            "Abs",
            inputs=node.get_input_names(),
            outputs=[abs_node_name],
            name=abs_node_name,
        )
        reducesum_node = helper.make_node(
            "ReduceSum",
            inputs=[abs_node_name],
            outputs=[outputnames[-1]],
            name=nodenames[-1],
            axes=tuple(axes),
            keepdims=0,
        )
        onnx_nodes.extend((abs_node, reducesum_node))
    elif reduction_param.operation == reduction_param.SUMSQ:
        reduce_sum_square_node = helper.make_node(
            "ReduceSumSquare",
            inputs=node.get_input_names(),
            outputs=[outputnames[-1]],
            name=nodenames[-1],
            axes=tuple(axes),
            keepdims=0,
        )
        onnx_nodes.append(reduce_sum_square_node)
    elif reduction_param.operation == reduction_param.MEAN:
        axes = []
        for i in range(axis, len(shape)):
            axes.append(i)
        onnx_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=node.get_input_names(),
                outputs=[outputnames[-1]],
                name=nodenames[-1],
                axes=tuple(axes),
                keepdims=0,
            ),
        )
    else:
        raise ValueError(
            "unknown reduction operation : " + str(reduction_param.operation),
        )
    if not math.isclose(coeff, 1.0):
        scale_name = node.name + "_coeff"
        scale_arr = np.ones(shape[:axis], np.float32) * float(coeff)
        if axis == 0:
            scale_arr = np.float32(coeff)
        init_tensors.append(numpy_helper.from_array(scale_arr, scale_name))
        onnx_nodes.append(
            helper.make_node(
                "Mul",
                inputs=[outputnames[-1], scale_name],
                outputs=[outputnames[0]],
                name=nodenames[0],
            ),
        )

    return onnx_nodes, init_tensors


def convert_psroipooling(node):
    # caffe layer parameters
    output_dim = node.layer.psroi_pooling_param.output_dim
    group_size = node.layer.psroi_pooling_param.group_size
    spatial_scale = node.layer.psroi_pooling_param.spatial_scale

    psroipooling_node = helper.make_node(
        "HzPSRoiPooling",
        inputs=[node.get_input_names()[0], node.get_input_names()[1]],
        outputs=node.get_output_names(),
        output_dim=output_dim,
        group_size=group_size,
        spatial_scale=spatial_scale,
        name=node.name,
        domain="horizon",
    )

    return [psroipooling_node]


def convert_deconvolution(node):
    # Only 2D deconv is supported now.
    # input shape is N x C x H x W
    # input weight is
    if len(node.parents) != 1:
        raise ValueError(
            f"deconvolution layer {node.name} can only have one input",
        )
    input_shape = node.parents[0].output_shapes[0]
    weight_blob = node.blobs[0] if len(node.layer.blobs) == 0 else node.layer.blobs[0]
    weight_shape = get_blob_shape(weight_blob)
    deconv_param = node.layer.convolution_param
    group = deconv_param.group
    if (input_shape[1] % group != 0) or (input_shape[1] != weight_shape[0]):
        raise ValueError(f"input shape dim error:{input_shape[1]}")
    if deconv_param.axis != 1:
        raise NotImplementedError(
            f"Currently model convert only support deconvolution layer"
            f" with axis=1. But layer:{node.name} has axis:{deconv_param.axis}",
        )
    kernel_height = get_kernel_value(deconv_param.kernel_h, deconv_param.kernel_size, 0)
    kernel_width = get_kernel_value(deconv_param.kernel_w, deconv_param.kernel_size, 1)
    bias_term = deconv_param.bias_term

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

    input_weight_name = node.name + "_weight"
    weight_init_tensor = helper.make_tensor(
        input_weight_name,
        TensorProto.FLOAT,
        weight_shape,
        weight_blob.data,
    )

    kernel_shape = [kernel_height, kernel_width]
    deconv_input = node.get_input_names()
    deconv_input.append(input_weight_name)
    init_tensors_info = [weight_init_tensor]
    if bias_term:
        input_bias_name = node.name + "_bias"
        bias_blob = node.blobs[1]
        bias_shape = get_blob_shape(bias_blob)
        bias_init_tensor = helper.make_tensor(
            input_bias_name,
            TensorProto.FLOAT,
            bias_shape,
            bias_blob.data,
        )
        init_tensors_info.append(bias_init_tensor)
        deconv_input.append(input_bias_name)

    deconv_node = helper.make_node(
        "ConvTranspose",
        inputs=deconv_input,
        outputs=node.get_output_names(),
        name=node.name,
        dilations=[dilation_height, dilation_width],
        group=group,
        kernel_shape=kernel_shape,
        pads=[pad_height, pad_width, pad_height, pad_width],
        strides=[stride_height, stride_width],
    )
    return [deconv_node], init_tensors_info


def convert_spp(node):
    onnx_nodes = []
    spp_param = node.layer.spp_param
    pyramid_height = spp_param.pyramid_height
    input_shape = node.parents[0].output_shapes[0]

    if spp_param.pool == 0:
        pooling_type = "MaxPool"
    elif spp_param.pool == 1:
        pooling_type = "AveragePool"
    else:
        # STOCHASTIC method did not implemented yet in Horizon Caffe.
        raise ValueError("ONNX not supported Pool type [%d]." % (spp_param.pool))

    if pyramid_height == 1:
        kernel_h, kernel_w, pad_h, pad_w = spp_pooling_param(
            0,
            input_shape[2],
            input_shape[3],
        )

        onnx_node = helper.make_node(
            pooling_type,
            inputs=node.get_input_names(),
            outputs=node.get_output_names(),
            name=node.name,
            kernel_shape=[kernel_h, kernel_w],
            pads=[pad_h, pad_w, pad_h, pad_w],
            strides=[kernel_h, kernel_w],
        )
        return [onnx_node]
    concat_bottom_vec = []
    for i in range(pyramid_height):
        # continue
        kernel_h, kernel_w, pad_h, pad_w = spp_pooling_param(
            i,
            input_shape[2],
            input_shape[3],
        )
        onnx_nodes.extend(
            (
                helper.make_node(
                    pooling_type,
                    inputs=node.get_input_names(),
                    outputs=[node.name + "_pooling_" + str(i)],
                    name=node.name + "_pooling_" + str(i),
                    kernel_shape=[kernel_h, kernel_w],
                    pads=[pad_h, pad_w, pad_h, pad_w],
                    strides=[kernel_h, kernel_w],
                ),
                helper.make_node(
                    "Flatten",
                    inputs=[node.name + "_pooling_" + str(i)],
                    outputs=[node.name + "_flatten_" + str(i)],
                    name=node.name + "_flatten_" + str(i),
                    axis=1,
                ),
            )
        )
        concat_bottom_vec.append(node.name + "_flatten_" + str(i))

    onnx_nodes.append(
        helper.make_node(
            "Concat",
            inputs=concat_bottom_vec,
            outputs=node.get_output_names(),
            name=node.name,
            axis=1,
        ),
    )
    return onnx_nodes


def spp_pooling_param(pyramid_level, bottom_h, bottom_w):
    num_bins = pow(2, pyramid_level)

    kernel_h = math.ceil(bottom_h / num_bins)
    remainder_h = kernel_h * num_bins - bottom_h
    pad_h = math.floor((remainder_h + 1) / 2)

    kernel_w = math.ceil(bottom_w / num_bins)
    remainder_w = kernel_w * num_bins - bottom_w
    pad_w = math.floor((remainder_w + 1) / 2)
    return kernel_h, kernel_w, pad_h, pad_w


def convert_matmul(node):
    if (len(node.parents[0].output_shapes[0]) != 2) or (
        len(node.parents[1].output_shapes[0]) != 2
    ):
        raise ValueError("ONNX matmul layer need rank of inputs as 2")
    dim_1 = node.layer.matmul_param.dim_1
    dim_2 = node.layer.matmul_param.dim_2
    dim_3 = node.layer.matmul_param.dim_3
    w_1, h_1 = node.parents[0].output_shapes[0]
    w_2, h_2 = node.parents[1].output_shapes[0]
    if (dim_1 != w_1) or (dim_2 != h_1) or (dim_2 != w_2) or (dim_3 != h_2):
        raise ValueError(
            "ONNX matmul layer only support the shape of inputs as"
            "[dim_1, dim_2] and [dim_2, dim_3]",
        )

    input_names = node.get_input_names()
    blob_c = np.zeros((w_1, h_2), np.float32)
    input_names.append(node.name + "_C")
    init_tensor = numpy_helper.from_array(blob_c, node.name + "_C")
    onnx_node = helper.make_node(
        "Gemm",
        inputs=input_names,
        outputs=node.get_output_names(),
        beta=0.0,
        name=node.name,
    )
    return [onnx_node], [init_tensor]


def convert_proposal(node):
    scales = node.layer.proposal_param.scale
    ratios = node.layer.proposal_param.ratio
    feat_stride = node.layer.proposal_param.feat_stride
    classes = node.layer.proposal_param.classes
    min_size = node.layer.proposal_param.min_size
    base_size = node.layer.proposal_param.base_size
    stds = node.layer.proposal_param.std
    means = node.layer.proposal_param.mean

    if len(stds) == 0:
        stds = [1.0, 1.0, 1.0, 1.0]
    elif len(stds) == 1:
        stds = [stds[0], stds[0], stds[0], stds[0]]

    if len(means) == 0:
        means = [0.0, 0.0, 0.0, 0.0]
    elif len(means) == 1:
        means = [means[0], means[0], means[0], means[0]]

    pre_nms_topn = node.layer.nms_param.pre_nms_topn
    post_nms_topn = node.layer.nms_param.post_nms_topn
    force_suppressed = 1 if node.layer.nms_param.force_suppressed is True else 0

    nms_thresh = node.layer.nms_param.nms_thresh

    bbox_decode_out = node.name + "_bbox_decode_out"
    bbox_decode_name = node.name + "_bbox_decode"
    bbox_decode_node = helper.make_node(
        "HzBBoxDecode",
        inputs=node.get_input_names(),
        outputs=[bbox_decode_out],
        domain="horizon",
        scales=scales,
        ratios=ratios,
        feat_stride=feat_stride,
        base_size=base_size,
        classes=classes,
        min_size=min_size,
        stds=stds,
        means=means,
        name=bbox_decode_name,
    )
    nms_out = node.name + "_nms_out"
    nms_name = node.name + "_nms"
    nms_node = helper.make_node(
        "HzNonMaxSuppression",
        inputs=[bbox_decode_out],
        outputs=[nms_out],
        pre_nms_topn=pre_nms_topn,
        post_nms_topn=post_nms_topn,
        force_suppressed=force_suppressed,
        nms_thresh=nms_thresh,
        name=nms_name,
        domain="horizon",
    )

    bbox_to_roi_node = helper.make_node(
        "HzBBoxtoRoi",
        inputs=[nms_out],
        outputs=node.get_output_names(),
        name=node.name,
        domain="horizon",
    )

    return [bbox_decode_node, nms_node, bbox_to_roi_node]


def convert_roi_post_process(node):
    stds = node.layer.roi_post_process_param.std
    means = node.layer.roi_post_process_param.mean
    batch_size = node.layer.roi_post_process_param.batch_size
    if len(stds) == 0:
        stds = [1.0, 1.0, 1.0, 1.0]
    elif len(stds) == 1:
        stds = [stds[0], stds[0], stds[0], stds[0]]

    if len(means) == 0:
        means = [0.0, 0.0, 0.0, 0.0]
    elif len(means) == 1:
        means = [means[0], means[0], means[0], means[0]]

    pre_nms_topn = node.layer.nms_param.pre_nms_topn
    post_nms_topn = node.layer.nms_param.post_nms_topn
    force_suppressed = 1 if node.layer.nms_param.force_suppressed is True else 0

    nms_thresh = node.layer.nms_param.nms_thresh
    score_thresh = node.layer.nms_param.score_thresh

    roi_decode_out = node.name + "_roi_decode_out"
    roi_decode_name = node.name + "_roi_decode"
    roi_decode_node = helper.make_node(
        "HzRoiDecode",
        inputs=node.get_input_names(),
        outputs=[roi_decode_out],
        batch_size=batch_size,
        stds=stds,
        means=means,
        name=roi_decode_name,
        domain="horizon",
    )

    nms_node = helper.make_node(
        "HzNonMaxSuppression",
        inputs=[roi_decode_out],
        outputs=node.get_output_names(),
        pre_nms_topn=pre_nms_topn,
        post_nms_topn=post_nms_topn,
        force_suppressed=force_suppressed,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh,
        name=node.name,
        domain="horizon",
    )

    return [roi_decode_node, nms_node]


def convert_argmax(node):
    axis = node.layer.argmax_param.axis
    if axis == 0:
        logging.warning("[Warning]: The axis of ArgMax op is equal to 0")
    arg_name = node.name + "_arg"
    arg_out = arg_name + "_out"

    arg_node = helper.make_node(
        "ArgMax",
        inputs=node.get_input_names(),
        outputs=[arg_out],
        name=arg_name,
        axis=axis,
    )
    cast_node = helper.make_node(
        "Cast",
        inputs=[arg_out],
        outputs=node.get_output_names(),
        name=node.name,
        to=TensorProto.FLOAT,
    )
    return [arg_node, cast_node]


def convert_crop(node):
    axis = node.layer.crop_param.axis
    offsets = node.layer.crop_param.offset
    if len(offsets) == 1:
        input_dim = node.parents[0].output_shapes[0]
        offsets = [
            offsets[0],
        ] * (len(input_dim) - axis)

    onnx_node = helper.make_node(
        "HzCrop",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        axis=axis,
        offsets=offsets,
        name=node.name,
        domain="horizon",
    )
    return [onnx_node]


def convert_crelu(node):
    try:
        filler = node.layer.scale_param.bias_filler.value
    except Exception:
        filler = 0.0
    onnx_node = helper.make_node(
        "HzCRelu",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        name=node.name,
        domain="horizon",
        axis=1,
        num_axes=1,
        bias_term=1,
        bias_filler=filler,
    )
    return [onnx_node]


def convert_relux(node, clip_value):
    input_name = node.get_input_names()
    input_min_name = node.name + "_min_value"
    input_max_name = node.name + "_max_value"
    input_name.extend([input_min_name, input_max_name])
    init_min_value = numpy_helper.from_array(np.array(0, np.float32), input_min_name)
    init_max_value = numpy_helper.from_array(
        np.array(clip_value, np.float32), input_max_name
    )
    onnx_node = helper.make_node(
        "Clip",
        inputs=input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    return [onnx_node], [init_min_value, init_max_value]


def convert_resize(node):
    input_name = node.get_input_names()
    roi_param_name = node.name + "_output_roi"
    scale_param_name = node.name + "_output_scale"
    size_param_name = node.name + "_output_size"
    input_name.append(roi_param_name)
    input_name.append(scale_param_name)
    input_name.append(size_param_name)
    output_size = node.output_shapes[0]
    init_size_info = helper.make_tensor(
        size_param_name, TensorProto.INT64, [len(output_size)], output_size
    )
    init_scale_info = helper.make_tensor(scale_param_name, TensorProto.FLOAT, [0], [])
    init_roi_info = helper.make_tensor(roi_param_name, TensorProto.FLOAT, [0], [])
    onnx_node = helper.make_node(
        "Resize",
        inputs=input_name,
        outputs=node.get_output_names(),
        name=node.name,
        coordinate_transformation_mode="align_corners",
        mode="linear",
    )

    return [onnx_node], [init_size_info, init_scale_info, init_roi_info]


def convert_lstm(node, blob_names):
    node_input = node.get_input_names()[0]

    recurrent_param = node.layer.recurrent_param
    num_output = recurrent_param.num_output
    input_shape = node.parents[0].output_shapes[0]
    batch_num = input_shape[1]
    time_steps = input_shape[0]

    lstm_inputs = []
    onnx_nodes = []
    init_info = []
    if len(input_shape) > 3:
        in_reshape_dim = [time_steps, batch_num, -1]
        in_reshape_input_name = []
        in_reshape_input_name.append(node_input)
        in_reshape_name = node.name + "_in_reshape"
        in_reshape_dim_name = in_reshape_name + "_dim"
        in_reshape_output_name = in_reshape_name + "_out"
        in_reshape_input_name.append(in_reshape_dim_name)
        in_reshape_init_t = helper.make_tensor(
            in_reshape_dim_name,
            TensorProto.INT64,
            [len(in_reshape_dim)],
            in_reshape_dim,
        )
        in_reshape_node = helper.make_node(
            "Reshape",
            inputs=in_reshape_input_name,
            outputs=[in_reshape_output_name],
            name=in_reshape_name,
        )
        onnx_nodes.append(in_reshape_node)
        init_info.append(in_reshape_init_t)
        lstm_inputs.append(in_reshape_output_name)
    else:
        lstm_inputs.append(node_input)

    lstm_inputs.extend(blob_names)

    lstm_name = node.name + "_lstm"
    lstm_node_output = node.name + "_out"
    lstm_node = helper.make_node(
        "LSTM",
        inputs=lstm_inputs,
        outputs=[lstm_node_output],
        hidden_size=num_output,
        name=lstm_name,
    )

    onnx_nodes.append(lstm_node)
    reshape_dim = [time_steps, batch_num, num_output]

    reshape_input_name = []
    reshape_input_name.append(lstm_node_output)
    reshape_dim_name = node.name + "_reshape_dim"
    reshape_input_name.append(reshape_dim_name)
    reshape_init_t = helper.make_tensor(
        reshape_dim_name,
        TensorProto.INT64,
        [len(reshape_dim)],
        reshape_dim,
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=reshape_input_name,
        outputs=node.get_output_names(),
        name=node.name,
    )
    init_info.append(reshape_init_t)
    onnx_nodes.append(reshape_node)
    return onnx_nodes, init_info


def convert_spatial_transformer(node):
    # STN only support affine convert, the second input of STN should be (N, 6)
    # STN is converted to a list of onnx nodes: [reshape, conv, HzGridSample]
    num_theta = 6
    default_theta_value = -1000
    st_param = node.layer.st_param

    is_pre_define_theta = [False] * num_theta
    pre_define_theta = [-1] * num_theta
    pre_define_x_theta_count = 0
    pre_define_y_theta_count = 0

    # load pre-define theta
    if st_param.theta_1_1 != default_theta_value:
        pre_define_x_theta_count += 1
        is_pre_define_theta[0] = True
        pre_define_theta[0] = st_param.theta_1_1
    if st_param.theta_1_2 != default_theta_value:
        pre_define_x_theta_count += 1
        is_pre_define_theta[1] = True
        pre_define_theta[1] = st_param.theta_1_2
    if st_param.theta_1_3 != default_theta_value:
        pre_define_x_theta_count += 1
        is_pre_define_theta[2] = True
        pre_define_theta[2] = st_param.theta_1_3
    if st_param.theta_2_1 != default_theta_value:
        pre_define_y_theta_count += 1
        is_pre_define_theta[3] = True
        pre_define_theta[3] = st_param.theta_2_1
    if st_param.theta_2_2 != default_theta_value:
        pre_define_y_theta_count += 1
        is_pre_define_theta[4] = True
        pre_define_theta[4] = st_param.theta_2_2
    if st_param.theta_2_3 != default_theta_value:
        pre_define_y_theta_count += 1
        is_pre_define_theta[5] = True
        pre_define_theta[5] = st_param.theta_2_3
    pre_define_count = pre_define_y_theta_count + pre_define_x_theta_count

    def check_valid(node):
        assert len(node.parents) == 2, (
            f"the previous node of STN should have 2 input, "
            f"but {len(node.parents)} is given"
        )
        conv_input_shape = node.parents[1].output_shapes[0]
        assert len(conv_input_shape) == 2, (
            f"the dimension of STN's second input should be 2, "
            f"but {len(conv_input_shape)} is given"
        )
        assert conv_input_shape[1] + pre_define_count == num_theta, (
            f"the theta for affine should be 6, but the amount of "
            f"the second input and pre_define_count is "
            f"{conv_input_shape[1]}+{pre_define_count}"
            f"={conv_input_shape[1] + pre_define_count}"
        )
        output_size = node.output_shapes[0]
        assert len(output_size) == 4, (
            "STN convert error in convert_ops.py, " + "len(output_shape) should be 4."
        )
        if hasattr(node.layer, "st_param"):
            transform_type = node.layer.st_param.transform_type
            sampler_type = node.layer.st_param.sampler_type
            # to_compute_dU is only used during backward
            # to_compute_du = node.layer.st_param.to_compute_dU
            if transform_type and transform_type != "affine":
                raise ValueError("Transformation type only supports affine now!")
            if sampler_type and sampler_type != "bilinear":
                raise ValueError("Sampler type only supports bilinear now!")

    check_valid(node)
    input_name = node.get_input_names()
    output_size = node.output_shapes[0]
    output_h = output_size[2]
    output_w = output_size[3]

    # the original second input shape is (N, X), we need add a reshape node to
    # expand it to (N, X, 1, 1)
    reshape_dim = [-1, num_theta - pre_define_count, 1, 1]
    reshape_input_name = []
    reshape_dim_name = node.name + "_reshape_dim"
    reshape_input_name.extend((input_name[1], reshape_dim_name))
    init_t = []
    init_t.append(
        helper.make_tensor(
            reshape_dim_name,
            TensorProto.INT64,
            [len(reshape_dim)],
            reshape_dim,
        ),
    )
    reshape_name = node.name + "_reshape"
    reshape_node = helper.make_node(
        "Reshape",
        inputs=reshape_input_name,
        outputs=[reshape_name],
        name=reshape_name,
    )

    # calculate weight and bias for conv
    # bias is in use when there is at least 1 pre-define theta
    def compute_weight():
        weight_shape = [output_h * output_w * 2, num_theta - pre_define_count]
        weight_data = np.zeros(weight_shape, np.float32)
        bias_shape = [output_h * output_w * 2]
        bias_data = np.zeros(bias_shape, np.float32)

        for i in range(output_h * output_w):
            x_theta_index = 0
            x_grid_value = [
                int(i / output_w) * 1.0 / output_h * 2 - 1,
                (i % output_w) * 1.0 / output_w * 2 - 1,
                1,
            ]
            if not is_pre_define_theta[0]:
                weight_data[2 * i][x_theta_index] = x_grid_value[0]
                x_theta_index += 1
            else:
                bias_data[2 * i] += pre_define_theta[0] * x_grid_value[0]
            if not is_pre_define_theta[1]:
                weight_data[2 * i][x_theta_index] = x_grid_value[1]
                x_theta_index += 1
            else:
                bias_data[2 * i] += pre_define_theta[1] * x_grid_value[1]
            if not is_pre_define_theta[2]:
                weight_data[2 * i][x_theta_index] = x_grid_value[2]
                x_theta_index += 1
            else:
                bias_data[2 * i] += pre_define_theta[2] * x_grid_value[2]
            for j in range(num_theta // 2 - pre_define_y_theta_count):
                weight_data[2 * i][x_theta_index + j] = 0
            y_theta_index = 0
            y_grid_value = [
                int(i / output_w) * 1.0 / output_h * 2 - 1,
                (i % output_w) * 1.0 / output_w * 2 - 1,
                1,
            ]

            for j in range(num_theta // 2 - pre_define_x_theta_count):
                weight_data[2 * i + 1][j] = 0
                y_theta_index += 1
            if not is_pre_define_theta[3]:
                weight_data[2 * i + 1][y_theta_index] = y_grid_value[0]
                y_theta_index += 1
            else:
                bias_data[2 * i + 1] += pre_define_theta[3] * y_grid_value[0]
            if not is_pre_define_theta[4]:
                weight_data[2 * i + 1][y_theta_index] = y_grid_value[1]
                y_theta_index += 1
            else:
                bias_data[2 * i + 1] += pre_define_theta[4] * y_grid_value[1]
            if not is_pre_define_theta[5]:
                weight_data[2 * i + 1][y_theta_index] = y_grid_value[2]
                y_theta_index += 1
            else:
                bias_data[2 * i + 1] += pre_define_theta[5] * y_grid_value[2]
        weight_shape.extend((1, 1))
        weight_data = np.reshape(weight_data, weight_shape)
        return weight_data, bias_data

    # we use conv to compute affine transformation
    weight_data, bias_data = compute_weight()
    conv_node_name = node.name + "_conv"
    conv_weight_name = node.name + "_conv_weight"
    conv_bias_name = node.name + "_conv_bias"
    init_t.extend(
        (
            numpy_helper.from_array(weight_data, conv_weight_name),
            numpy_helper.from_array(bias_data, conv_bias_name),
        )
    )
    conv_input_name = []
    conv_input_name.extend((reshape_name, conv_weight_name, conv_bias_name))
    conv_node = helper.make_node(
        "Conv",
        inputs=conv_input_name,
        outputs=[conv_node_name],
        name=conv_node_name,
        group=1,
        dilations=[1, 1],
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    # add a HzGridSample node
    sampler_input_name = []
    sampler_input_name.extend((input_name[0], conv_node_name))
    sampler_node = helper.make_node(
        "HzGridSample",
        inputs=sampler_input_name,
        outputs=node.get_output_names(),
        sizes=[output_h, output_w],
        name=node.name,
        domain="horizon",
    )

    return [reshape_node, conv_node, sampler_node], init_t
