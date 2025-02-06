import logging
import traceback

from google.protobuf import text_format

from . import horizon_caffe_pb2
from .custom_op import shape_custom
from .shape_inference import *  # noqa: F403

_convert_map = {
    "BatchNorm": shape_identity,
    "Scale": shape_identity,
    "ReLU": shape_identity,
    "Eltwise": shape_identity,
    "InnerProduct": shape_inner_product,
    "Pooling": shape_pooling,
    "Convolution": shape_convolution,
    "Softmax": shape_identity,
    "Concat": shape_concat,
    "Dropout": shape_identity,
    "LRN": shape_identity,
    "Flatten": shape_flatten,
    "Data": shape_data,
    "Reshape": shape_reshape,
    "PassThrough": shape_pass_through,
    "Upsample": shape_upsample,
    "Permute": shape_permute,
    "Sigmoid": shape_identity,
    "TanH": shape_identity,
    "Exp": shape_identity,
    "Bias": shape_identity,
    "Power": shape_identity,
    "Axpy": shape_axpy,
    "AbsVal": shape_identity,
    "ELU": shape_identity,
    "Log": shape_identity,
    "Threshold": shape_identity,
    "ROIPooling": shape_roipooling,
    "PReLU": shape_identity,
    "BNLL": shape_identity,
    "Normalize": shape_identity,
    "Slice": shape_slice,
    "MVN": shape_identity,
    "Split": shape_split,
    "Reduction": shape_reduction,
    "PSROIPooling": shape_psroipooling,
    "Deconvolution": shape_deconvolution,
    "SPP": shape_spp,
    "MatMul": shape_matmul,
    "Proposal": shape_proposal,
    "RoiPostProcess": shape_roi_post_process,
    "ArgMax": shape_argmax,
    "RReLU": shape_identity,
    "Crop": shape_crop,
    "CReLU": shape_crelu,
    "ReLU6": shape_identity,
    "Resize": shape_resize,
    "LSTM": shape_lstm,
    "ContinuationIndicator": shape_continuation_indicator,
    "SpatialTransformer": shape_spatial_transformer,
    "Custom": shape_custom,
}

_caffe_layer_types = _convert_map.keys()


class CaffeNode:
    def __init__(self, name, node_type, layer=None):
        if name == "":
            raise ValueError("Node name should not be empty")
        self.name = name
        self.type = node_type
        self.layer = layer
        self.blobs = None

        self.output_shapes = None
        # map of new unique output name and original caffe output name
        self.output_name_map = {}
        # list of output which is also graph output
        self.graph_output = []
        self.parents = []
        self.children = []
        self.input_index = []

    def update_output_name_map(self, unique_names, output=None):
        self.output_name_map.clear()
        if output is None:
            output = []
        self.graph_output = output
        if self.layer is None:
            # input node
            self.output_name_map[self.name] = self.name
        elif len(self.layer.top) == 1:
            # node has only one outout
            if self.layer.top[0] in output:
                self.output_name_map[self.layer.top[0]] = self.layer.top[0]
            else:
                new_name = self.name
                index = 1
                while new_name in unique_names:
                    new_name = self.name + "_" + str(index)
                    index += 1
                self.output_name_map[self.layer.top[0]] = new_name
        else:
            # node has multiple outputs
            for index, top_name in enumerate(self.layer.top):
                if top_name in output:
                    self.output_name_map[top_name] = top_name
                else:
                    new_name = self.name + "_" + str(index)
                    dup_count = 1
                    while new_name in unique_names:
                        new_name = self.name + "_" + str(index + dup_count)
                        dup_count += 1
                    self.output_name_map[top_name] = new_name

    def mark_as_output(self):
        self.as_output = True

    def add_parent(self, parent_node):
        # assert parent_node not in self.parents
        if parent_node not in self.parents:
            self.parents.append(parent_node)
            if self not in parent_node.children:
                parent_node.children.append(self)

    def add_child(self, child_node):
        if child_node not in self.children:
            self.children.append(child_node)
            if self not in child_node.parents:
                child_node.parents.append(self)

    def get_input_names(self):
        return (
            [
                parent.output_name_map[bottom]
                for bottom in self.layer.bottom
                for parent in self.parents
                if bottom in parent.output_name_map
            ]
            if self.parents
            else []
        )

    def get_output_names(self):
        return list(self.output_name_map.values())

    def infer_shape(self):
        try:
            self.output_shapes = _convert_map[self.type](self)
        except Exception:
            error_info = "\n=======================ERROR INFO=======================\n"
            error_info += traceback.format_exc()
            error_info += f"Error occurs when infering shape of node:\n{self}"
            error_info += "\nPrevious context information of error node:"
            for i, node in enumerate(self.parents):
                error_info += (
                    f"\nprevious {i}-th node: {node.name} ({node.type}), "
                    f"output shape: {node.output_shapes}"
                )
            error_info += "\n=======================ERROR END========================"
            logging.error(error_info)
            exit(1)

    def __str__(self):
        return str(self.layer).replace("\n", "; ")

    def __repr__(self):
        return f"{self.name} (0x{id(self):x})"

    def add_input_index(self, index):
        self.input_index.append(index)


class CaffeGraph:
    def __init__(self, prototxt):
        self.proto = prototxt
        self.name = ""
        self.layers = None
        self.model_inputs = {}  # map of input name and shape
        # all names of input and output
        self.unique_names = set()
        # map of model output tensor name and node name
        self.output_layer_map = {}
        # map of all node output tensor name and node name
        self.top_layer_map = {}
        # map of all node input tensor name and node name
        self.bottom_layer_map = {}

        self.nodes = []  # CaffeNode object list
        self.nodename_node_map = {}  # map of node name and CaffeNode object
        self.top_index_map = {}
        self._parse_proto()

    def _parse_proto_input(self):
        """Parse input shape from ModelProto."""
        if len(self.proto.input) == 0:
            # Some caffe prototxt define the input layer by layer{}
            # specification, not input
            remove_layers = []
            for layer in self.layers:
                if layer.type == "Input":
                    input_dim = list(layer.input_param.shape[0].dim)
                    if len(layer.top) != 1:
                        raise ValueError(
                            f"The number of top of Input layer {layer.name} is not 1.",
                        )
                    self.model_inputs[layer.top[0]] = input_dim
                    remove_layers.append(layer)
            # remove input layers
            for layer in remove_layers:
                self.layers.remove(layer)
        elif len(self.proto.input) == 1:
            if len(self.proto.input_dim):
                input_dim = list(map(int, self.proto.input_dim))
            elif len(self.proto.input_shape):
                input_dim = list(map(int, self.proto.input_shape[0].dim))
            else:
                raise ValueError("Cannot find input size")
            self.model_inputs[self.proto.input[0]] = input_dim
        else:
            # For same models(op tests), there may be two input layers(eltwise)
            # or more input layers(concat).
            if len(self.proto.input_dim):
                # In this case, all input will share same input dims.
                for i in range(len(self.proto.input)):
                    self.model_inputs[self.proto.input[i]] = list(
                        map(int, self.proto.input_dim),
                    )
            elif len(self.proto.input_shape):
                for i in range(len(self.proto.input)):
                    self.model_inputs[self.proto.input[i]] = list(
                        map(int, self.proto.input_shape[i].dim),
                    )
            else:
                raise ValueError("Cannot find input size")

    def update_top_index_map(self):
        for layer in self.layers:
            for index, top in enumerate(layer.top):
                if top not in self.top_index_map:
                    self.top_index_map[top] = {layer.name: index}
                else:
                    self.top_index_map[top].update({layer.name: index})

    def _make_input_node(self):
        nodes = []
        for name, shape in self.model_inputs.items():
            node = CaffeNode(name, "Data")
            node.output_shapes = [tuple(shape)]
            nodes.append(node)
        return nodes

    def _make_node(self, layer):
        if layer.type not in _caffe_layer_types:
            raise ValueError(
                f"Not support layer name={layer.name} type={layer.type}",
            )

        return CaffeNode(layer.name, layer.type, layer=layer)

    def _check_layer_phase_test(self, layer):
        """Check the phase of a caffe layer.

        Each caffe layer may has an attribute include {phase: TEST} or
        include {phase: TRAIN}.
        If a layer specified it's phase=TRAIN, we will not parse this layer.
        """
        if len(layer.include) != 0:
            for message in layer.include:
                if (
                    isinstance(message, horizon_caffe_pb2.NetStateRule)
                    and message.phase == 0
                ):
                    # enum Phase {TRAIN = 0; TEST = 1;}
                    logging.warning(
                        "caffe protoxt contains a TRAIN phase layer:"
                        + layer.name
                        + ", we will ignore it during the deploy phase.",
                    )
                    return False
        return True

    def _parse_node_dependency(self):
        """Establish dependencies between nodes and parse outputs."""
        input_name_set, output_name_list = set(), []
        # establish input nodes top layer map
        for node in self.nodes:
            self.nodename_node_map[node.name] = node
        for layer in self.layers:
            if not self._check_layer_phase_test(layer):
                continue
            node = self._make_node(layer)
            self.nodes.append(node)
            self.nodename_node_map[layer.name] = node
            for bottom in layer.bottom:
                if bottom not in layer.top:
                    input_name_set.add(bottom)
                    self.unique_names.add(bottom)
                if bottom in self.model_inputs:
                    node.add_parent(self.nodename_node_map[bottom])
                elif bottom in self.top_layer_map:
                    parent = self.nodename_node_map[self.top_layer_map[bottom]]
                    self.nodename_node_map[layer.name].add_parent(parent)
                    self.nodename_node_map[layer.name].add_input_index(
                        self.top_index_map[bottom][parent.name],
                    )
                else:
                    raise ValueError(
                        f"Bottom:{bottom} of layer:{layer.name} should be "
                        f"in top_layer_map."
                    )
            for top in layer.top:
                output_name_list.append(top)
                self.unique_names.add(top)
                self.top_layer_map.update({top: layer.name})
        for name in output_name_list:
            if name not in input_name_set:
                self.output_layer_map[name] = self.top_layer_map[name]

    def _update_node_input_output(self):
        for node in self.nodes:
            outputs = []
            if node.name in self.output_layer_map.values():
                outputs.extend(
                    top
                    for top in node.layer.top
                    if top in self.output_layer_map
                    and self.output_layer_map[top] == node.name
                )
            node.update_output_name_map(self.unique_names, outputs)

    def _parse_proto(self):
        self.layers = self.proto.layers or self.proto.layer
        if self.proto.name == "":
            self.name = "unnamed-caffe-model"
        else:
            self.name = self.proto.name
        self._parse_proto_input()
        logging.info(f"Find {len(self.model_inputs)} inputs in the model:")
        for name, shape in self.model_inputs.items():
            logging.info(f"Got input '{name}' with shape {shape}.")

        self.update_top_index_map()
        self.nodes = self._make_input_node()
        self._parse_node_dependency()
        self._update_node_input_output()

    def set_input_shapes(self, input_shapes):
        input_specified = False
        for input_name, input_shape in input_shapes.items():
            if input_name not in self.model_inputs:
                raise ValueError(
                    f"Name '{input_name}' of input shape is not found in prototxt",
                )
            if len(input_shape) != len(self.model_inputs[input_name]):
                raise ValueError(
                    f"For input {input_name}, its shape ({input_shape}) is "
                    f"different from {self.model_inputs[input_name]}, which "
                    f"is parsed from prototxt.",
                )
            if self.model_inputs[input_name] == input_shape:
                continue
            logging.info(
                f"Replace the input {input_name}'s shape from "
                f"{self.model_inputs[input_name]} to {input_shape}"
            )
            input_specified = True
            self.model_inputs[input_name] = input_shape
        if input_specified:
            for name, shape in self.model_inputs.items():
                logging.info(f"Got input '{name}' with shape {shape}.")

    def get_input_nodes(self):
        return [node for node in self.nodes if node.type == "Data"]

    def get_output_nodes(self):
        output_nodes = []
        for node_name in self.output_layer_map.values():
            if self.nodename_node_map[node_name] not in output_nodes:
                output_nodes.append(self.nodename_node_map[node_name])
        return output_nodes

    def set_node_params(self, params):
        for name, _, blobs in params:
            if name in self.nodename_node_map:
                node = self.nodename_node_map[name]
                node.blobs = blobs

    def infer_shape(self):
        def infer_shape_node(node):
            if node.output_shapes is None:
                if len(node.parents) == 0:
                    raise ValueError(
                        f"The layer {node.name} has no input. "
                        f"Check that the model is correct, or "
                        f"remove the layer from the model.",
                    )
                if node.parents[0].output_shapes is None:
                    infer_shape_node(node.parents[0])
                node.infer_shape()

        for node in self.nodes:
            infer_shape_node(node)

    def _remove_nodes(self, nodes):
        new_nodes = [node for node in self.nodes if node not in nodes]
        self.nodes.clear()
        self.nodename_node_map.clear()
        self.nodes = new_nodes
        self.nodename_node_map = {node.name: node for node in new_nodes}

    def _find_index(self, layerio, name):
        for i, item in enumerate(layerio):
            if name == item:
                return i
        return -1

    def fuse_batch_norm_and_scale(self):
        scale_nodes = []
        for node in self.nodes:
            if len(node.parents) == 0:
                continue
            parent = node.parents[0]
            if (
                node.type != "Scale"
                or parent.type != "BatchNorm"
                or len(parent.children) > 1
            ):
                continue

            scale_nodes.append(node)
            if len(node.graph_output) > 0:
                # scale node output as graph output
                if len(node.children) > 0:
                    raise ValueError(
                        f"Scale node {node.name} has graph output "
                        f"should not have child!",
                    )
                if len(node.graph_output) > 1:
                    raise ValueError(
                        f"Scale node {node.name} should not have "
                        f"more than one graph output",
                    )
                parent.layer.top[0] = node.layer.top[0]
                parent.children.remove(node)
                parent.update_output_name_map(self.unique_names, node.graph_output)
                self.output_layer_map[node.graph_output[0]] = parent.name
            else:
                parent.children.remove(node)
                for child in node.children:
                    child_bottom_index = self._find_index(
                        child.layer.bottom,
                        node.layer.top[0],
                    )
                    # As the children's bottom may change, we should modify
                    # their bottom names in case we parsing input names miss.
                    child.layer.bottom[child_bottom_index] = node.layer.bottom[0]
                    child_parent_index = child.parents.index(node)
                    child.parents[child_parent_index] = parent
                    parent.children.append(child)

            if node.blobs is not None:
                if parent.blobs is not None:
                    parent.blobs.extend(node.blobs)
                else:
                    # TODO: is this situation legal?
                    parent.blobs = node.blobs

        self._remove_nodes(scale_nodes)

    def remove_continuation_indicator(self):
        remove_nodes = []
        for node in self.nodes:
            if node.type == "ContinuationIndicator":
                remove_nodes.append(node)
                if len(node.graph_output) > 0:
                    raise ValueError(
                        """ContinuationIndicator layer '"""
                        + node.name
                        + """' shouldn't be graph output""",
                    )
        self._remove_nodes(remove_nodes)

    def remove_split(self):
        split_nodes = []
        for node in self.nodes:
            if node.type != "Split":
                continue
            split_nodes.append(node)
            parent = node.parents[0]
            if len(node.graph_output) > 1:
                raise ValueError(
                    """Split layer '"""
                    + node.name
                    + """' shouldn't have more than one graph output""",
                )
            if len(node.graph_output) == 1:
                for child in parent.children:
                    if child.name == node.name:
                        continue
                    child_bottom_index = self._find_index(
                        child.layer.bottom,
                        node.layer.bottom[0],
                    )
                    if child_bottom_index != -1:
                        child.layer.bottom[child_bottom_index] = node.graph_output[0]
                parent_top_index = self._find_index(
                    parent.layer.top,
                    node.layer.bottom[0],
                )
                if parent_top_index == -1:
                    raise ValueError("Internal graph parser error!")
                # Change parent layer top to ensure graph output consistent
                for child in parent.children:
                    child_bottom_index = self._find_index(
                        child.layer.bottom,
                        parent.layer.top[parent_top_index],
                    )
                    # As the children's bottom may change, we should modify
                    # their bottom names in case we parsing input names miss.
                    child.layer.bottom[child_bottom_index] = node.graph_output[0]
                parent.layer.top[parent_top_index] = node.graph_output[0]

                parent_output = parent.graph_output
                parent_output.extend(node.graph_output)
                parent.update_output_name_map(self.unique_names, parent_output)
                # Don't forget to update graph output-layer relation map
                self.output_layer_map[node.graph_output[0]] = parent.name

                parent.children.remove(node)

                # Move node's children to parent
                for _, top in enumerate(node.layer.top):
                    if top == node.graph_output[0]:
                        continue
                    for _, child in enumerate(node.children):
                        child_bottom_index = self._find_index(child.layer.bottom, top)
                        if child_bottom_index == -1:
                            continue
                        child.layer.bottom[child_bottom_index] = node.graph_output[0]
                        child_parent_index = child.parents.index(node)
                        child.parents[child_parent_index] = parent
                        parent.add_child(child)
            else:
                parent.children.remove(node)
                for _, child in enumerate(node.children):
                    for _, top in enumerate(node.layer.top):
                        if top in child.layer.bottom:
                            child_bottom_index = self._find_index(
                                child.layer.bottom,
                                top,
                            )
                            child.layer.bottom[child_bottom_index] = node.layer.bottom[
                                0
                            ]
                            child_parent_index = child.parents.index(node)
                            child.parents[child_parent_index] = parent
                            parent.add_child(child)
        self._remove_nodes(split_nodes)

    def __str__(self):
        def get_blob_shapes(node):
            if node.blobs:
                blob_shapes = [blob.shape.dim for blob in node.blobs]
            else:
                blob_shapes = "--"
            return blob_shapes

        hdr = "{:<20} {:<30} {:<30} {:<20} {:<10}".format(
            "Type",
            "Name",
            "Attr",
            "Output",
            "Ops",
        )
        s = [hdr, "-" * 120]
        for node in self.nodes:
            blob_shapes = get_blob_shapes(node)
            output_shapes = node.output_shapes or "--"
            if node.type == "Convolution":
                filter_shape = tuple(blob_shapes[0])
                ops = (
                    2
                    * filter_shape[1]
                    * filter_shape[2]
                    * filter_shape[3]
                    * output_shapes[0][0]
                    * output_shapes[0][1]
                    * output_shapes[0][2]
                    * output_shapes[0][3]
                )
            else:
                filter_shape = "--"
                ops = "--"
            s.append(
                f"{node.type:<20} {node.name:<30} {filter_shape!s:<30} "
                f"{output_shapes!s:<20} {ops!s:<10}"
            )
        return "\n".join(s)

    def get_graph_input(self):
        return self.model_inputs


class ModelParser:
    def __init__(self, prototxt_file, caffemodel_file):
        self.prototxt_file = prototxt_file
        self.caffemodel_file = caffemodel_file
        self.proto = None  # horizon_caffe_pb2.NetParameter object
        self.caffemodel = None
        self.params = []

        self._load_prototxt()
        self._load_caffemodel()

    def _load_prototxt(self):
        self.proto = horizon_caffe_pb2.NetParameter()
        with open(self.prototxt_file) as f:
            text_format.Merge(f.read(), self.proto)

    def _load_caffemodel(self):
        if self.caffemodel_file is not None:
            self.caffemodel = horizon_caffe_pb2.NetParameter()
            with open(self.caffemodel_file, "rb") as f:
                self.caffemodel.ParseFromString(f.read())
            layers = self.caffemodel.layers or self.caffemodel.layer
            for layer in layers:
                self.params.append((layer.name, layer.type, layer.blobs))

    def parser(self, input_shapes=None):
        graph = CaffeGraph(self.proto)
        # ModelParser can specify the model input shape.
        # Check the name and shape are correct.
        if input_shapes is not None:
            graph.set_input_shapes(input_shapes)

        graph.set_node_params(self.params)
        graph.infer_shape()
        graph.fuse_batch_norm_and_scale()
        graph.remove_split()
        graph.remove_continuation_indicator()

        return graph


def get_caffe_input(prototxt_file):
    """Get input name with shape in caffe prototxt.

    Args:
        prototxt_file: File of caffe prototxt.

    Returns:
        Dict with keys as input name and values as input shape.
    """
    proto = horizon_caffe_pb2.NetParameter()
    with open(prototxt_file) as f:
        text_format.Merge(f.read(), proto)
    graph = CaffeGraph(proto)
    return graph.get_graph_input()
