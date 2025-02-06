import os

import torch
from hbdk4.compiler import compile, convert, leap, visualize
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from horizon_plugin_pytorch import March, set_march
from horizon_plugin_pytorch.quantization import FakeQuantState
from horizon_plugin_pytorch.quantization import hbdk4 as hb4
from horizon_plugin_pytorch.quantization import prepare, set_fake_quantize
from horizon_plugin_pytorch.quantization.qconfig_template import (
    default_qat_qconfig_setter,
)

global_march = March.NASH_E

leap.load_library(
    os.path.join(os.path.dirname(__file__), "csrc", "custom_matmul.so")
)


def custom_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    input_device = x.device
    ret = leap.custom.user.MatMul(x.cpu(), y.cpu())
    return ret.to(input_device)


class CppToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = nn.Conv2d(3, 10, 5)

    def forward(self, x, y):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        if self.training:
            return x
        else:
            return custom_matmul(x.reshape(1, -1), y)


float_model = CppToyNet().cuda()
example_input = (torch.rand(1, 3, 5, 5).cuda(), torch.rand(10, 5).cuda())
float_model(*example_input)

set_march(global_march)
qat_model = prepare(
    float_model,
    example_inputs=example_input,
    qconfig_setter=default_qat_qconfig_setter,
)

qat_model.eval()
set_fake_quantize(qat_model, FakeQuantState.VALIDATION)
qat_model(*example_input)

hbir_model = hb4.export(qat_model, example_input)
# visualize(hbir_model, "{}.onnx".format(__file__))

quantized_hbir_model = convert(hbir_model, global_march)
compile(quantized_hbir_model, "custom_cpp_op.hbm", global_march)
