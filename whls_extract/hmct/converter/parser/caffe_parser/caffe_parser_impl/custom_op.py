import logging

from hmct.custom.op_registration import (
    get_op_implement,
    get_op_shape_infer,
    register_op_as_identity,
)
from hmct.ir.onnx_utils import TensorProto, helper


def shape_custom(node):
    custom_param = node.layer.custom_param
    shapes = []
    if len(custom_param.shape) == len(node.get_output_names()):
        logging.info(
            "[Custom Op]: Node name {}, output shape: {}".format(
                node.name,
                str(custom_param.shape).replace("\n", " "),
            ),
        )
        for i, _, shape in zip(
            range(len(custom_param.shape)),
            node.get_output_names(),
            custom_param.shape,
        ):
            shape = tuple(custom_param.shape[i].dim[:])
            shapes.append(shape)
    else:
        # get custom op's infer_shape function.
        infer_shape_func = get_op_shape_infer(custom_param.kind)
        if infer_shape_func is not None:

            def _get_input_shape(input_name):
                for parent in node.parents:
                    for output_name, output_shape in zip(
                        parent.get_output_names(),
                        parent.output_shapes,
                    ):
                        if output_name == input_name:
                            return output_shape
                return None

            # get custom op's input shapes
            input_shapes = []
            for input_name in node.get_input_names():
                input_shape = _get_input_shape(input_name)
                input_shapes.append(input_shape)

            # infer shape with custom op's infer_shape function.
            output_shapes = infer_shape_func(input_shapes)
            shapes.extend(output_shapes)
            logging.info(
                f"[Custom Op]: node name {node.name}, "
                f"input shapes: {input_shapes}, "
                f"output shapes: {output_shapes}",
            )
        else:
            raise NotImplementedError(
                f"[Custom Op]: node name {node.name}, "
                f"no output shapes is provided, "
                f"and no infer_shape function is registered.",
            )

    return shapes


def convert_custom(node):
    custom_param = node.layer.custom_param
    # Get custom op's module and class_name.
    impl = get_op_implement(custom_param.kind)
    if impl is None:
        register_op_as_identity(custom_param.kind, len(node.get_output_names()))
        impl = get_op_implement(custom_param.kind)

    cls = impl["cls"]
    module = impl["module"]
    class_name = impl["name"]
    compute = "compute"

    # Parsing op's parameters.
    import inspect

    import yaml

    params = custom_param.params
    params_dict = yaml.safe_load(params)
    required_params = inspect.getfullargspec(cls.__init__)
    required_args = required_params.args
    defaults = required_params.defaults
    args = {}
    for i, arg in zip(range(len(required_args)), required_args):
        if arg == "self":
            continue
        if arg in params_dict:
            args[arg] = str(params_dict[arg])
        else:
            if i < len(required_args) - len(defaults):
                raise ValueError(f"Require arg: {arg}")
            args[arg] = str(defaults[i + len(defaults) - len(required_args)])

    # Create PyOp node.
    custom_node = helper.make_node(
        op_type="PyOp",
        name=node.name,
        domain="horizon.custom",
        inputs=node.get_input_names(),
        outputs=node.get_output_names(),
        input_types=[TensorProto.FLOAT],
        output_types=[TensorProto.FLOAT],
        module=module,
        class_name=class_name,
        compute=compute,
        **args,
    )

    return [custom_node]
