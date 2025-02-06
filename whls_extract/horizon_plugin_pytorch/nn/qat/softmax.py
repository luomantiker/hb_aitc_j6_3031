import logging
from distutils.version import LooseVersion

import torch
from torch import Tensor
from torch.nn import Module, functional

from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.memory_opt import MemoryOptManager, MemoryOptSwitch
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.checkpoint import checkpoint
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.typeguard import typechecked
from ..softmax import Softmax as HorizonSoftmax
from .functional_modules import FloatFunctional
from .reciprocal import Reciprocal
from .segment_lut import SegmentLUT

logger = logging.getLogger(__name__)
softmax_checkpoint_switch = MemoryOptSwitch(
    "SoftmaxCkpt", 0, [0, 1, 2, 3, 4], levels=[0, None, None, None, 1]
)
MemoryOptManager.register_switch(softmax_checkpoint_switch)


class QuantiSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, scale, input_type, output_type, max_softmax_value):
        types = {"qint8": torch.int8, "qint16": torch.int16}
        type_info = torch.iinfo(types[input_type])
        qxmin = type_info.min
        qxmax = type_info.max
        qtensor = QTensor(
            torch.clamp(torch.round(data / scale), qxmin, qxmax).to(
                torch.int8
            ),
            scale,
            input_type,
        )
        from horizon_plugin_pytorch.nn.quantized.softmax import QuantSoftmax

        softmax = QuantSoftmax(output_type, max_softmax_value).to(data.device)
        q_res = softmax(qtensor)
        q_res = q_res.dequantize()
        ctx.save_for_backward(data, q_res)
        return q_res

    @staticmethod
    def backward(ctx, grad_out):
        data, softmax_res = ctx.saved_tensors
        if LooseVersion(torch.__version__.split("+")[0]) >= LooseVersion(
            "1.13.0"
        ):
            in_grad = torch._softmax_backward_data(
                output=softmax_res,
                dim=1,
                grad_output=grad_out,
                input_dtype=data.dtype,
            )
        else:
            in_grad = torch._softmax_backward_data(
                output=softmax_res, dim=1, grad_output=grad_out, input=data
            )
        return in_grad, None, None, None, None


class Softmax(torch.nn.Module):

    _FLOAT_MODULE = torch.nn.Softmax

    def __init__(self, qconfig=None):
        super(Softmax, self).__init__()
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_post_process = self.qconfig.activation()
        self.register_buffer("max_softmax_value", torch.tensor([0.0]))

    @torch.cuda.amp.autocast(enabled=False)
    @typechecked
    def forward(self, input: QTensor) -> QTensor:
        if not input.is_quantized:
            input = QTensor(
                input.as_subclass(torch.Tensor).float(),
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        assert input.dtype == "qint8", "Only support qint8 input"

        with torch.no_grad():
            max_softmax = torch.max(
                torch.softmax(
                    input.as_subclass(torch.Tensor).detach()
                    - torch.max(
                        input.as_subclass(torch.Tensor).detach(),
                        dim=1,
                        keepdims=True,
                    ).values,
                    dim=1,
                )
            )
        if self.max_softmax_value < max_softmax:
            self.max_softmax_value.copy_(max_softmax.clone().detach())
        return self.activation_post_process(
            QuantiSoftmaxFunction.apply(
                input.as_subclass(torch.Tensor),
                input.q_scale(),
                input.dtype,
                self.activation_post_process.dtype,
                self.max_softmax_value,
            )
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        if SegmentLUT.activated():
            return SegmentLUTSoftmax.from_float(mod)
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        assert mod.dim in [None, 1], "Only support softmax along channel dim"
        qconfig = mod.qconfig
        qat_softmax = cls(qconfig=qconfig)
        return qat_softmax


class SegmentLUTSoftmax(Module):
    _FLOAT_MODULE = (torch.nn.Softmax, HorizonSoftmax)
    _version: int = 3

    def __init__(
        self,
        dim=None,
        min_sub_out=-12.0,
        reciprocal_kwargs=None,
        qconfig=None,
    ):
        super(SegmentLUTSoftmax, self).__init__()
        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"
        if min_sub_out is not None:
            assert min_sub_out < 0, "min_sub_out must be less than 0"

        self.dim = dim
        self.qconfig = qconfig
        self.min_sub_out = min_sub_out

        from horizon_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)

        # exp input is always -a <= x <= 0(a > 0), so after fake quanti in
        # qat, 0 <= exp(x) <= 1 and exp out_scale = 2 / 65535. If a << 0, such
        # as a = 2000, the actual segment lut table is generated on range
        # [-2000, 2000], which leads to poor quantized precision. So specific
        # input range here and clamp sub min out to a fixed value.
        # With out_scale = 2 / 65535, fake_quanti(exp(-12.0)) = 0. Clamp sub
        # min out(exp input) to -12.0 set input_range = [-12.0, 0.0] to avoid
        # [0, 12] table generation.
        self.sub = FloatFunctional(qconfig=int16_qconfig)
        self.exp = SegmentLUT(
            torch.exp,
            False,
            input_range=[self.min_sub_out, 0.0],
            gradients=[0.0, None],
            qconfig=int16_qconfig,
        )
        self.sum = FloatFunctional(qconfig=int16_qconfig)
        self.reciprocal = Reciprocal(
            lut_kwargs=reciprocal_kwargs, qconfig=int16_qconfig
        )
        self.mul = FloatFunctional(qconfig=qconfig)

        self.forward_mapping = {
            0: self._forward_ckpt_l0,
            1: self._forward_ckpt_l1,
            2: self._forward_ckpt_l2,
            3: self._forward_ckpt_l3,
            4: self._forward_ckpt_l4,
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # in version 3, we replace the reciprocal from SegmentLUT to Reciprocal
        if get_version(self, prefix, local_metadata) < 3:
            old_reciprocal_param_names = []
            for k in state_dict:
                if k.startswith(prefix + "reciprocal"):
                    old_reciprocal_param_names.append(k)

            prefix_len = len(prefix + "reciprocal")
            for n in old_reciprocal_param_names:
                new_name = n[:prefix_len] + ".reciprocal" + n[prefix_len:]
                state_dict[new_name] = state_dict.pop(n)

            # use old SegmentLUT params
            self.reciprocal.reciprocal.auto_divide_strategy = "evenly"
            self.reciprocal.reciprocal.inverse_func = None

        super(SegmentLUTSoftmax, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _forward_ckpt_l0(self, input: QTensor):
        input = self.sub.sub(input, input.max(dim=self.dim, keepdim=True)[0])
        exp = self.exp(input)
        exp_sum = self.sum.sum(exp, dim=self.dim, keepdim=True)
        exp_sum_reciprocal = self.reciprocal(exp_sum)
        ret = self.mul.mul(exp, exp_sum_reciprocal)

        return ret

    def _forward_ckpt_l1(self, input: QTensor):
        input = self.sub.sub(input, input.max(dim=self.dim, keepdim=True)[0])
        exp = self.exp(input)
        exp_sum = self.sum.sum(exp, dim=self.dim, keepdim=True)
        exp_sum_reciprocal = self.reciprocal.forward_wo_fq(exp_sum)
        ret = checkpoint(self._forward_ckpt_l1_tail, exp, exp_sum_reciprocal)

        return ret

    def _forward_ckpt_l1_tail(self, exp: QTensor, exp_sum_reciprocal: Tensor):
        exp_sum_reciprocal = (
            self.reciprocal.reciprocal.activation_post_process(
                exp_sum_reciprocal
            )
        )
        ret = self.mul.mul(exp, exp_sum_reciprocal)

        return ret

    def _forward_ckpt_l2(self, input: QTensor):
        input = self.sub.sub(input, input.max(dim=self.dim, keepdim=True)[0])
        exp = self.exp(input)
        ret = checkpoint(self._forward_ckpt_l2_tail, exp)

        return ret

    def _forward_ckpt_l2_tail(self, exp: QTensor):
        exp_sum = self.sum.sum(exp, dim=self.dim, keepdim=True)
        exp_sum_reciprocal = self.reciprocal(exp_sum)
        ret = self.mul.mul(exp, exp_sum_reciprocal)

        return ret

    def _forward_ckpt_l3(self, input: QTensor):
        input = self.sub.sub(input, input.max(dim=self.dim, keepdim=True)[0])
        exp = self.exp.forward_wo_fq(input)
        ret = checkpoint(self._forward_ckpt_l3_tail, exp)

        return ret

    def _forward_ckpt_l3_tail(self, exp: Tensor):
        exp = self.exp.activation_post_process(exp)
        exp_sum = self.sum.sum(exp, dim=self.dim, keepdim=True)
        exp_sum_reciprocal = self.reciprocal(exp_sum)
        ret = self.mul.mul(exp, exp_sum_reciprocal)

        return ret

    def _forward_ckpt_l4(self, input: QTensor):
        ret = checkpoint(self._forward_ckpt_l0, input)
        return ret

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: QTensor):
        if not input.is_quantized:
            input = QTensor(
                input.as_subclass(torch.Tensor).float(),
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        if self.dim is None:
            self.dim = functional._get_softmax_dim("softmax", input.dim(), 5)

        ret = self.forward_mapping[softmax_checkpoint_switch.value](input)
        return ret

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) in cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + (c.__name__ for c in cls._FLOAT_MODULE)
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_softmax = cls(
            mod.dim,
            min_sub_out=mod.min_sub_out
            if hasattr(mod, "min_sub_out")
            else -12.0,
            reciprocal_kwargs=mod.reciprocal_kwargs
            if hasattr(mod, "reciprocal_kwargs")
            else None,
            qconfig=qconfig,
        )
        return qat_softmax
