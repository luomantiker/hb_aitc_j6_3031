import copy

from torch import Tensor
from torch.nn import Module

from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn import segment_lut as float_segment_lut
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version


class SegmentLUT(Module):
    _version = 2
    _FLOAT_MODULE = float_segment_lut.SegmentLUT
    supported_march = (
        March.BAYES,
        March.BAYES_E,
        March.META,
        March.NASH,
        March.NASH_E,
        March.NASH_M,
        March.NASH_P,
    )

    def __init__(
        self,
        simulated_func,
        is_centrosymmetric=False,
        dividing_points=None,
        input_range=None,
        auto_divide_strategy="evenly",
        inverse_func=None,
        gradients=None,
        qconfig=None,
        num_entries_per_table=64,
    ):
        assert self.activated(), "SegmentLUT only support march in {}!".format(
            self.supported_march
        )
        assert qconfig, "qconfig must be provided for QAT module"
        assert qconfig.activation, (
            "activation_post_process must included "
            + "in qconfig for qat.SegmentLUT"
        )
        super(SegmentLUT, self).__init__()

        self.simulated_func = simulated_func
        self.is_centrosymmetric = is_centrosymmetric
        self.dividing_points = dividing_points
        self.input_range = input_range
        self.auto_divide_strategy = auto_divide_strategy
        self.inverse_func = inverse_func
        self.gradients = gradients
        self.num_entries_per_table = num_entries_per_table
        self.qconfig = qconfig

        self.activation_post_process = self.qconfig.activation()
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def forward(self, input: QTensor):
        input = pre_process(self.activation_pre_process, input)
        out = self.simulated_func(input.as_subclass(Tensor))
        return self.activation_post_process(out)

    def forward_wo_fq(self, input: QTensor):
        return self.simulated_func(input.as_subclass(Tensor))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        get_version(self, prefix, local_metadata)
        # we removed the input_scale buffer
        key = prefix + "input_scale"
        if key in state_dict:
            state_dict.pop(key)

        super(SegmentLUT, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def activated(cls):
        return get_march() in cls.supported_march

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            simulated_func=mod.simulated_func,
            is_centrosymmetric=mod.is_centrosymmetric,
            dividing_points=mod.dividing_points,
            input_range=mod.input_range,
            auto_divide_strategy=mod.auto_divide_strategy,
            inverse_func=mod.inverse_func,
            gradients=mod.gradients,
            qconfig=mod.qconfig,
            num_entries_per_table=mod.num_entries_per_table,
        )
        return qat_mod


class QuantizedQATSegmentLUT(Module):
    def __init__(self, qat_mod, enabled=True):
        assert isinstance(qat_mod, SegmentLUT)
        super().__init__()
        from horizon_plugin_pytorch.nn.quantized import (
            SegmentLUT as QuantizedSegmentLUT,
        )

        self.qat_mod = qat_mod
        self.quantized_mod = QuantizedSegmentLUT.from_float(qat_mod)
        self.quantized_forward = enabled
        self.input_scale = None
        self.output_scale = None
        self.input_dtype = None
        self.output_dtype = None

    def forward(self, input: QTensor):
        self.input_dtype = input.dtype
        self.input_scale = input.q_scale()
        if self.quantized_forward:
            input = input.to_quantized()
            ret = self.quantized_mod(input)
            ret = ret.to_fake_quantized()
        else:
            ret = self.qat_mod(input)
        self.output_dtype = ret.dtype
        self.output_scale = ret.q_scale()
        return ret

    @classmethod
    def convert_segment_lut(cls, model: Module, enabled=True):
        holder = []
        for n, m in model.named_children():
            if isinstance(m, cls):
                continue
            if isinstance(m, SegmentLUT):
                holder.append((n, m))
            else:
                cls.convert_segment_lut(m, enabled)
        for n, m in holder:
            convert_m = cls(m, enabled)
            convert_m._forward_hooks = copy.deepcopy(m._forward_hooks)
            convert_m._forward_pre_hooks = copy.deepcopy(m._forward_pre_hooks)
            convert_m._forward_hooks_with_kwargs = copy.deepcopy(
                m._forward_hooks_with_kwargs
            )
            convert_m._forward_pre_hooks_with_kwargs = copy.deepcopy(
                m._forward_pre_hooks_with_kwargs
            )
            setattr(model, n, convert_m)

    @classmethod
    def recover_qat_segment_lut(cls, model: Module):
        for n, m in model.named_children():
            if isinstance(m, cls):
                setattr(model, n, m.qat_mod)
            else:
                cls.recover_qat_segment_lut(m)
