import torch
from torch.nn import Module

from ..softmax_bernoulli2 import SoftmaxBernoulli2 as FloatSoftmaxBernoulli2


class SoftmaxBernoulli2(Module):
    _FLOAT_MODULE = FloatSoftmaxBernoulli2

    def __init__(self, dim=None, max_value_only=False, qconfig=None):
        super(SoftmaxBernoulli2, self).__init__()
        from horizon_plugin_pytorch.dtype import qint8
        from horizon_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        self.dim = dim
        self.max_value_only = max_value_only
        self.exp_qconfig = replace_qconfig_dtype(qconfig, qint8)
        self.activation_exp_post_process = self.exp_qconfig.activation()
        self.qconfig = qconfig
        from horizon_plugin_pytorch import get_march

        assert (
            get_march() == "bernoulli2"
        ), "SoftmaxBernoulli2 only works for march=bernoulli2"

    def forward(self, x):
        x = x.as_subclass(torch.Tensor)
        x = torch.exp(x)
        x = x - x.max(dim=self.dim, keepdim=True)[0]
        x = self.activation_exp_post_process(x)
        x = x.as_subclass(torch.Tensor)
        sum_x = torch.sum(x, dim=self.dim, keepdim=True)
        output = x / sum_x
        if self.max_value_only:
            output = torch.max(output, dim=self.dim, keepdim=True)[0]
        return output

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        qat_softmax = cls(mod.dim, mod.max_value_only, mod.qconfig)
        return qat_softmax
