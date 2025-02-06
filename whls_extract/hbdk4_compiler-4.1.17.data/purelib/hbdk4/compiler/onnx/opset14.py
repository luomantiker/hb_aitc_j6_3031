from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir


class Opset14(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 14, True)


class BatchNormalization(Opset14):
    def __init__(self):
        super().__init__("BatchNormalization")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        scale: mlir.Value,
        bias: mlir.Value,
        mean: mlir.Value,
        var: mlir.Value,
        *,
        epsilon: float = 1e-05,
        momentum: float = 0.9,
        training_mode: int = 1,
    ):
        input_len = len(adaptor.operands[0].type.shape)
        axis = [i for i in range(input_len)]
        permutes_c_last = axis[0:1] + axis[2:] + axis[1:2]
        permutes_c_ahead = (
            axis[0:1] + axis[input_len - 1 : input_len] + axis[1 : input_len - 1]
        )
        data = hbir.transpose(data, permutes_c_last)
        data = hbir.batchnorm(
            data, weight=scale, bias=bias, mean=mean, var=var, eps=epsilon
        )
        data = hbir.transpose(data, permutes_c_ahead, output_type=y)
        return data


BatchNormalization()
