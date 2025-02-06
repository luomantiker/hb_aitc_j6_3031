import numpy as np

from hbdk4.compiler.overlay import Function, Value, Operation
from hbdk4.compiler import ir as mlir


class OnnxConvertor:
    def __init__(self, func: Function):

        from onnx import TensorProto

        self.onnx_type_mapping = {
            "si8": TensorProto.INT8,
            "ui8": TensorProto.UINT8,
            "si16": TensorProto.INT16,
            "ui16": TensorProto.UINT16,
            "si32": TensorProto.INT32,
            "ui32": TensorProto.UINT32,
            "f32": TensorProto.FLOAT,
            "f16": TensorProto.FLOAT16,
            "bf16": TensorProto.BFLOAT16,
            "si64": TensorProto.INT64,
            "!hbir.bool8": TensorProto.BOOL,
        }

        self.func = func

        self.placeholders = {}
        self.consts = {}
        self.nodes = []
        self.debug_name = {}
        self.debug_cnt = 0

    def set_debug_name(self, value, name):
        self.debug_name[value.value] = name

    def get_debug_name(self, value):
        return self.debug_name[value.value]

    def get_or_emit_debug_name(self, value):
        if value.value in self.debug_name:
            return self.get_debug_name(value)
        else:
            name = str(self.debug_cnt)
            self.debug_cnt += 1
            self.debug_name[value.value] = name
            return str(name)

    def build_placeholder(self, name: str, tensor_type: Value):

        from onnx import helper

        if tensor_type.quant_info is not None:
            dtype = tensor_type.quant_info.storage_type.__str__()
        else:
            dtype = tensor_type.dtype

        placeholder = helper.make_tensor_value_info(
            name,
            self.onnx_type_mapping[dtype],
            tensor_type.shape,
            tensor_type.__str__(),
            False,
        )
        self.placeholders[name] = placeholder

    def build_const(self, const_op: Operation):
        name = self.get_or_emit_debug_name(const_op.outputs[0])
        array = np.array(mlir.DenseElementsAttr(const_op.op.attributes["values"]))

        from onnx import helper

        tensor_type = const_op.outputs[0].type
        if tensor_type.quant_info is not None:
            dtype = tensor_type.quant_info.storage_type.__str__()
        else:
            dtype = tensor_type.dtype

        tensor = helper.make_tensor(
            name,
            self.onnx_type_mapping[dtype],
            const_op.outputs[0].type.shape,
            array.tobytes(),
            True,
        )
        self.consts[name] = tensor

    def build_node(self, op: Operation):
        operand_names = [self.get_debug_name(val) for val in op.inputs]
        result_names = [self.get_or_emit_debug_name(val) for val in op.outputs]

        for idx, name in enumerate(result_names):
            self.build_placeholder(name, op.outputs[idx].type)

        onnx_op_type = op.type

        if op.type in ["hbtl.call"]:
            onnx_op_type += "." + op.schema.namespace + "." + op.schema.signature
        onnx_op_type = onnx_op_type.replace(".", "_")

        from onnx import helper

        node = helper.make_node(
            onnx_op_type,
            operand_names,
            result_names,
            op.name,
            op.__str__(),
            **op.attributes
        )
        self.nodes.append(node)

    def gen_onnx(self, path):

        arg_names = []
        for idx, arg in enumerate(self.func.flatten_inputs):
            name = arg.name
            arg_names.append(name)
            self.build_placeholder(name, arg.type)
            self.set_debug_name(arg, name)

        for idx, ret in enumerate(self.func.flatten_outputs):
            name = ret.name
            self.set_debug_name(ret, name)

        ret_names = None

        for op in self.func.operations:
            if op.type == "hbir.constant":
                self.build_const(op)
            elif op.type == "cf.br":
                continue
            elif op.type == "func.return":
                ret_names = [self.get_debug_name(val) for val in op.inputs]
                break
            else:
                self.build_node(op)

        import onnx
        from onnx import helper

        graph = helper.make_graph(
            self.nodes,
            self.func.name,
            [self.placeholders[name] for name in arg_names],
            [self.placeholders[name] for name in ret_names],
            self.consts.values(),
            self.func.__str__(),
            self.placeholders.values(),
        )
        model = helper.make_model(graph)

        onnx_version = onnx.__version__
        major, minor, patch = map(int, onnx_version.split("."))
        # Check if the version is below 3.9.0
        if major < 3 or (major == 3 and minor < 9):
            onnx.save(model, path)
        else:
            onnx.save(model, path, save_as_external_data=True)
