import numpy as np
import torch
from torch.nn import Module

from ..qat.softmax_bernoulli2 import SoftmaxBernoulli2 as QatSoftmaxBernoulli2
from .functional import softmax_bernoulli2


class SoftmaxBernoulli2(Module):
    _FLOAT_MODULE = QatSoftmaxBernoulli2

    def __init__(self, dim=None, max_value_only=False, exp_scale=None):
        super(SoftmaxBernoulli2, self).__init__()
        from horizon_plugin_pytorch import get_march

        assert (
            get_march() == "bernoulli2"
        ), "SoftmaxBernoulli2 only works with bernoulli2 march"
        self.dim = dim
        self.max_value_only = max_value_only
        self.register_buffer("exp_scale", exp_scale)

    def _generate_table(self, input_scale, output_scale):
        quant_input = np.arange(-255, 1, 1, dtype=np.float32)
        quant_input *= input_scale.item()
        quant_output = np.exp(quant_input) / output_scale.item()
        quant_output = np.round(quant_output)
        quant_output_max = 255
        quant_output = np.clip(quant_output, 0, quant_output_max)
        table = torch.tensor(
            quant_output, dtype=torch.int32, device=self.exp_scale.device
        )
        return table

    def forward(self, x):
        input_scale = x.q_scale()
        table = self._generate_table(input_scale, self.exp_scale)
        x = x.as_subclass(torch.Tensor)
        return softmax_bernoulli2(x, table, self.max_value_only, self.dim)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        quantized_softmax = cls(
            dim=mod.dim,
            exp_scale=mod.activation_exp_post_process.scale,
            max_value_only=mod.max_value_only,
        )
        return quantized_softmax
