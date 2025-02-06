import logging
import operator
import traceback

import numpy as np

from hmct.ir.onnx_utils import (
    ModelProto,
    TensorProto,
    checker,
    helper,
    numpy_helper,
    shape_inference,
)

from .caffe_parser import ModelParser
from .convert_ops import *  # noqa: F403
from .custom_op import convert_custom
from .shape_inference import get_blob_shape


def caffe_to_onnx(node_list):
    onnx_nodes = []
    initializer = []

    def convert_node(node):
        onnx_node = None
        if node.type == "ReLU":
            onnx_node = convert_relu(node)
        elif node.type == "Softmax":
            onnx_node = convert_softmax(node)
        elif node.type == "Pooling":
            onnx_node = convert_pooling(node)
        elif node.type == "Convolution":
            if node.blobs is None or len(node.blobs) < 1:
                raise ValueError(
                    f"Convolution layer {node.name} has no "
                    f"parameters, check the caffemodel",
                )
            blob_names = []
            weight_name = node.name + "_weight"
            weight_data = np.array(node.blobs[0].data, np.float32).reshape(
                get_blob_shape(node.blobs[0]),
            )
            initializer.append(numpy_helper.from_array(weight_data, weight_name))
            blob_names.append(weight_name)
            if len(node.blobs) == 2:
                bias_name = node.name + "_bias"
                bias_shape = get_blob_shape(node.blobs[1])
                if len(bias_shape) == 4:
                    # from some model(BLVC caffe googlenetv1,
                    # the shape of conv_bias is (1,1,1,C).
                    # it is ok for onnxruntime CPU, but onnxruntime
                    # GPU need b_shape.NumDimensions() == 1
                    from functools import reduce

                    bias_shape = reduce(operator.mul, bias_shape)
                bias_data = np.array(node.blobs[1].data, np.float32).reshape(bias_shape)
                initializer.append(numpy_helper.from_array(bias_data, bias_name))
                blob_names.append(bias_name)
            onnx_node = convert_convolution(node, blob_names)
        elif node.type == "BatchNorm":
            scale_name = node.name + "_scale"
            bias_name = node.name + "_bias"
            mean_name = node.name + "_mean"
            var_name = node.name + "_var"
            blob_names = [scale_name, bias_name, mean_name, var_name]

            if node.blobs is None or len(node.blobs) < 1:
                raise ValueError(
                    f"BatchNorm layer {node.name} has no "
                    f"parameters, check the caffemodel",
                )
            mean_data = np.array(node.blobs[0].data, np.float32).reshape(
                get_blob_shape(node.blobs[0]),
            )
            var_data = np.array(node.blobs[1].data, np.float32).reshape(
                get_blob_shape(node.blobs[1]),
            )

            rescale_factor = node.blobs[2].data[0]
            if rescale_factor != 0 and rescale_factor != 1:
                mean_data /= rescale_factor
                var_data /= rescale_factor

            initializer.append(numpy_helper.from_array(mean_data, mean_name))
            initializer.append(numpy_helper.from_array(var_data, var_name))
            if len(node.blobs) == 3:  # bn in yolov2 does not have scale and bias
                scale_data = np.ones(get_blob_shape(node.blobs[0]), np.float32)
                bias_data = np.zeros(get_blob_shape(node.blobs[0]), np.float32)
                initializer.append(numpy_helper.from_array(scale_data, scale_name))
                initializer.append(numpy_helper.from_array(bias_data, bias_name))
                onnx_node = convert_batch_norm(node, blob_names)
            else:
                if len(node.blobs) != 5:
                    raise ValueError(
                        f"the number of bn layer {node.name}'s parameter is wrong",
                    )
                scale_data = np.array(node.blobs[3].data, np.float32).reshape(
                    get_blob_shape(node.blobs[3]),
                )
                bias_data = np.array(node.blobs[4].data, np.float32).reshape(
                    get_blob_shape(node.blobs[4]),
                )

                initializer.append(numpy_helper.from_array(scale_data, scale_name))
                initializer.append(numpy_helper.from_array(bias_data, bias_name))

                onnx_node = convert_batch_norm(node, blob_names)

        elif node.type == "Data":
            return
        elif node.type == "Eltwise":
            onnx_node, init_tensors_info = convert_eltwise(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Flatten":
            onnx_node = convert_flatten(node)
        elif node.type == "Reshape":
            onnx_node, reshapeparam_init_t = convert_reshape(node)
            initializer.append(reshapeparam_init_t)
        elif node.type == "InnerProduct":
            blob_names = []
            weight_name = node.name + "_weight"
            bias_name = node.name + "_bias"
            if node.blobs is None or len(node.blobs) < 1:
                raise ValueError(
                    f"Innerproduct layer {node.name} has "
                    f"no parameters, check the caffemodel",
                )
            weight_shape = get_blob_shape(node.blobs[0])
            if len(weight_shape) == 4:
                # get_blob_shape for BLVC Googlenet will return an 4 dim list
                # only 2 dim is necessary here
                weight_shape = weight_shape[2:]
            weight_data = np.array(node.blobs[0].data, np.float32).reshape(weight_shape)
            initializer.append(numpy_helper.from_array(weight_data, weight_name))
            blob_names.append(weight_name)
            if len(node.blobs) == 2:
                bias_shape = get_blob_shape(node.blobs[1])
                # Note: the dimension of the bias_shape may be 1 or 4.
                # If the dimension is 4, in general, the values of the first
                # three dimensions are all 1(like: (1, 1, 1, 1000)), so only
                # need to take the values of the last two dimensions.
                if len(bias_shape) == 4:
                    bias_shape = bias_shape[2:]
                bias_data = np.array(node.blobs[1].data, np.float32).reshape(bias_shape)
            else:
                # Gemm op in onnx9 must have bias, if innerproduct op in caffe
                # does not have bias, add a zero-pad bias for Gemm in onnx
                node_output_shape = node.output_shapes[0]
                assert len(node_output_shape) >= 2, (
                    f"the length of Gemm node:"
                    f"{node.name}'s output is less than 2, error"
                )
                bias_shape = node_output_shape[1:2]
                bias_data = np.zeros(bias_shape, np.float32).reshape(bias_shape)
            initializer.append(numpy_helper.from_array(bias_data, bias_name))
            blob_names.append(bias_name)
            onnx_node, init_tensors_info = convert_innerproduct(
                node,
                blob_names,
            )
            initializer.extend(init_tensors_info)
        elif node.type == "Concat":
            onnx_node = convert_concat(node)
        elif node.type == "Dropout":
            onnx_node = convert_dropout(node)
        elif node.type == "LRN":
            onnx_node = convert_lrn(node)
        elif node.type == "PassThrough":
            onnx_node = convert_pass_through(node)
        elif node.type == "Upsample":
            if len(node.layer.bottom) == 2:
                onnx_node = convert_max_unpool(node)
            else:
                scale_name = node.name + "_scale"
                upsample_scale = node.layer.upsample_param.scale
                scale_data = np.array([1, 1, upsample_scale, upsample_scale]).astype(
                    np.float32,
                )
                initializer.append(numpy_helper.from_array(scale_data, scale_name))
                onnx_node, init_tensors_info = convert_upsample(
                    node, scale_name, "nearest"
                )
                initializer.extend(init_tensors_info)
        elif node.type == "Permute":
            onnx_node = convert_permute(node)
        elif node.type == "Sigmoid":
            onnx_node = convert_sigmoid(node)
        elif node.type == "TanH":
            onnx_node = convert_tanh(node)
        elif node.type == "Exp":
            onnx_node, init_tensors_info = convert_exp(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Log":
            onnx_node, init_tensors_info = convert_log(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Scale":
            onnx_node, reshape_init_t = convert_scale(node)
            initializer.extend(reshape_init_t)
        elif node.type == "Bias":
            onnx_node, reshape_init_t = convert_bias(node)
            initializer.extend(reshape_init_t)
        elif node.type == "Power":
            onnx_node, init_tensors_info = convert_power(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Axpy":
            onnx_node = convert_axpy(node)
        elif node.type == "AbsVal":
            onnx_node = convert_abs_val(node)
        elif node.type == "ELU":
            onnx_node = convert_elu(node)
        elif node.type == "Threshold":
            onnx_node, init_tensors_info = convert_threshold(node)
            if len(init_tensors_info):
                initializer.extend(init_tensors_info)
        elif node.type == "ROIPooling":
            onnx_node = convert_roipooling(node)
        elif node.type == "PReLU":
            # We require the slope input to be passed in.
            # For onnx will not handle filler parameter to
            # init slope tensor.
            if node.blobs is None or len(node.blobs) < 1:
                raise ValueError(
                    f"PReLU layer {node.name} has no slope "
                    f"parameters, check the caffemodel",
                )
            slope_shape = get_blob_shape(node.blobs[0])

            onnx_node, init_tensors_info = convert_prelu(node, slope_shape)
            initializer.extend(init_tensors_info)
        elif node.type == "BNLL":
            onnx_node, init_tensors_info = convert_bnll(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Normalize":
            scale_name = node.name + "_scale"
            if len(node.layer.blobs) == 0:
                scale_blobs = node.blobs[0]
            else:
                scale_blobs = node.layer.blobs[0]
            scale_shape = get_blob_shape(scale_blobs)
            scale_data = np.array(scale_blobs.data, np.float32).reshape(scale_shape)
            initializer.append(numpy_helper.from_array(scale_data, scale_name))
            onnx_node = convert_normalize(node, scale_name)
        elif node.type == "Slice":
            onnx_node = convert_slice(node)
        elif node.type == "MVN":
            onnx_node = convert_mvn(node)
        elif node.type == "Reduction":
            onnx_node, init_tensors_info = convert_reduction(node)
            if len(init_tensors_info):
                initializer.extend(init_tensors_info)
        elif node.type == "PSROIPooling":
            onnx_node = convert_psroipooling(node)
        elif node.type == "Deconvolution":
            onnx_node, init_tensors_info = convert_deconvolution(node)
            initializer.extend(init_tensors_info)
        elif node.type == "SPP":
            onnx_node = convert_spp(node)
        elif node.type == "MatMul":
            onnx_node, init_tensors_info = convert_matmul(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Proposal":
            onnx_node = convert_proposal(node)
        elif node.type == "RoiPostProcess":
            onnx_node = convert_roi_post_process(node)
        elif node.type == "ArgMax":
            onnx_node = convert_argmax(node)
        elif node.type == "RReLU":
            # RReLu can be convert as ReLU
            onnx_node = convert_relu(node)
        elif node.type == "Crop":
            onnx_node = convert_crop(node)
        elif node.type == "CReLU":
            onnx_node = convert_crelu(node)
        elif node.type == "ReLU6":
            onnx_node, init_tensors_info = convert_relux(node, clip_value=6)
            initializer.extend(init_tensors_info)
        elif node.type == "Resize":
            onnx_node, init_tensors_info = convert_resize(node)
            initializer.extend(init_tensors_info)
        elif node.type == "Custom":
            onnx_node = convert_custom(node)
        elif node.type == "SpatialTransformer":
            onnx_node, init_tensors_info = convert_spatial_transformer(node)
            initializer.extend(init_tensors_info)
        elif node.type == "LSTM":
            blob_names = []
            weight_shape = get_blob_shape(node.blobs[0])
            weight_name = node.name + "_weight"

            weight_data = np.array(node.blobs[0].data, np.float32).reshape(
                [4, int(weight_shape[0] / 4), weight_shape[1]],
            )

            weight_shape = [1, weight_shape[0], weight_shape[1]]
            weight_data = np.concatenate(
                [weight_data[0], weight_data[2], weight_data[1], weight_data[3]],
            ).reshape(weight_shape)

            initializer.append(numpy_helper.from_array(weight_data, weight_name))

            blob_names.append(weight_name)

            rec_weight_shape = get_blob_shape(node.blobs[2])
            rec_weight_name = node.name + "_rec"
            rec_weight_data = np.array(node.blobs[2].data, np.float32).reshape(
                [4, rec_weight_shape[1], rec_weight_shape[1]],
            )

            rec_weight_shape = [1, rec_weight_shape[0], rec_weight_shape[1]]

            rec_weight_data = np.concatenate(
                [
                    rec_weight_data[0],
                    rec_weight_data[2],
                    rec_weight_data[1],
                    rec_weight_data[3],
                ],
            ).reshape(rec_weight_shape)
            initializer.append(
                numpy_helper.from_array(rec_weight_data, rec_weight_name),
            )
            blob_names.append(rec_weight_name)

            bias_shape = get_blob_shape(node.blobs[1])

            bias_name = node.name + "_bias"
            bias_data = np.array(node.blobs[1].data, np.float32).reshape([4, -1])

            bias_shape = [1, bias_shape[0]]
            bias_data = np.concatenate(
                [bias_data[0], bias_data[2], bias_data[1], bias_data[3]],
            ).reshape(bias_shape)

            rec_bias_data = np.zeros_like(bias_data)
            bias_data = np.concatenate([bias_data, rec_bias_data], axis=1)
            initializer.append(numpy_helper.from_array(bias_data, bias_name))
            blob_names.append(bias_name)

            onnx_node, init_tensors_info = convert_lstm(
                node,
                blob_names,
            )
            initializer.extend(init_tensors_info)
        else:
            raise ValueError(f"Not support layer type={node.type}")
        if node.type != "Split" and onnx_node[-1].name != node.name:
            # Note: need to ensure that the name of the last node in onnx_node
            # is the same as that of the original caffe node.
            raise ValueError(
                "The name of the last node in onnx_node "
                "should be same as that of the original caffe node."
            )
        onnx_nodes.extend(onnx_node)

    for i, node in enumerate(node_list):
        try:
            convert_node(node)
        except Exception as e:  # noqa: PERF203
            error_info = "\n=======================ERROR INFO=======================\n"
            error_info += traceback.format_exc()
            error_info += f"Error Exception: {e}\n"
            error_info += f"Error occurs during caffe to onnx of node:\n{node}"
            error_info += "\nPrevious context information of error node:"
            for i, parent_node in enumerate(node.parents):
                error_info += (
                    f"\nprevious {i}-th node: {parent_node.name} "
                    f"({parent_node.type}), output shape: "
                    f"{parent_node.output_shapes}"
                )
            error_info += "\n=======================ERROR END========================"
            logging.error(error_info)
            exit(1)

    return (onnx_nodes, initializer)


def build_onnx_model(graph, opset_version=11):
    # 1. Get ONNX nodes.
    onnx_nodes, initializer = caffe_to_onnx(graph.nodes)

    # 2. Create graph inputs.
    inputs = []
    inputs.extend(
        helper.make_tensor_value_info(
            input_var.name, TensorProto.FLOAT, input_var.output_shapes[0]
        )
        for input_var in graph.get_input_nodes()
    )
    inputs.extend(
        helper.make_tensor_value_info(
            constant_value.name,
            constant_value.data_type,
            constant_value.dims,
        )
        for constant_value in initializer
    )

    # 3. Create graph outputs.
    outputs = []
    for output_node in graph.get_output_nodes():
        for name, shape in zip(
            output_node.get_output_names(),
            output_node.output_shapes,
        ):
            if name not in graph.output_layer_map:
                continue
            outputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape),
            )

    # 4. Get node's output shape from caffe node.
    # Custom op cannot infer shapes.
    # The output shape needs to be recorded in the graph.
    value_info = []  # create value info
    for node in graph.nodes:
        if node.type == "Custom":
            for i, name, _ in zip(
                range(len(node.get_output_names())),
                node.get_output_names(),
                node.output_shapes,
            ):
                value_info.append(
                    helper.make_tensor_value_info(
                        name,
                        TensorProto.FLOAT,
                        node.output_shapes[i],
                    ),
                )

    # 5. Make graph.
    graph = helper.make_graph(
        onnx_nodes,
        graph.name,
        inputs,
        outputs,
        initializer,
        value_info=value_info,
    )
    # 6. Make model.
    onnx_model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", opset_version),
            helper.make_opsetid("horizon.custom", 1),
            helper.make_opsetid("horizon", 1),
        ],
    )
    # 7. Check the model.
    onnx_model = shape_inference.infer_shapes(onnx_model)
    checker.check_model(onnx_model)
    return onnx_model


# 将caffe模型转成onnx模型
def convert_caffe_to_onnx(prototxt_file, caffemodel_file) -> ModelProto:
    """Convert caffe graph to onnx model.

    Args:
        prototxt_file: File name of caffe prototxt.
        caffemodel_file: File name of caffe model.

    Returns:
        Converted onnx ModelProto.
    """
    caffe_graph = ModelParser(prototxt_file, caffemodel_file).parser()
    return build_onnx_model(caffe_graph)
