import logging
from contextlib import contextmanager
from typing import Iterable, Union

import torch
from torch import Tensor, nn

from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils._quant_mapping_extra import (
    register_float_to_qat_mapping,
)
from horizon_plugin_pytorch.utils.misc import pytree_convert

logger = logging.getLogger(__name__)


def init_activation_preprocesses(obj):
    # if input number is unknown, let the obj
    # handle activation_preprocesses itself
    if obj._input_num is None:
        return

    if hasattr(obj.qconfig, "input") and obj.qconfig.input is not None:
        if obj._input_num == 1:
            obj.activation_pre_process = obj.qconfig.input()
        else:
            if isinstance(obj.qconfig.input, (list, tuple)):
                assert len(obj.qconfig.input) == obj._input_num
                obj.activation_pre_process = torch.nn.ModuleList(
                    (
                        nn.Identity() if q is None else q()
                        for q in obj.qconfig.input
                    )
                )
            else:
                obj.activation_pre_process = torch.nn.ModuleList(
                    (
                        nn.Identity()
                        if obj.qconfig.input is None
                        else obj.qconfig.input()
                        for i in range(obj._input_num)
                    )
                )

    else:
        obj.activation_pre_process = None


def init_qat(obj):
    init_activation_preprocesses(obj)

    if obj.qconfig.activation is not None:
        obj.activation_post_process = obj.qconfig.activation()
    else:
        obj.activation_post_process = None


def common_init(obj, *args, **kwargs):
    qconfig = kwargs.pop("qconfig", None)
    if qconfig is None:
        raise ValueError(
            "qconfig must be provided for QAT module as keyword argument"
        )

    super(type(obj), obj).__init__(*args, **kwargs)

    obj.qconfig = qconfig
    init_qat(obj)


def single_input_forward(obj, input: Union[Tensor, QTensor], *args, **kwargs):
    if obj.activation_pre_process is not None and isinstance(input, Tensor):
        input = obj.activation_pre_process(input)

    if isinstance(input, QTensor):
        input = input.as_subclass(Tensor)

    output = super(type(obj), obj).forward(
        input.as_subclass(Tensor), *args, **kwargs
    )

    if obj.activation_post_process is None:
        return output
    else:
        return obj.activation_post_process(output)


def double_input_forward(
    obj, x: Union[Tensor, QTensor], y: Union[Tensor, QTensor]
):
    if obj.activation_pre_process is not None:
        if isinstance(x, Tensor):
            x = obj.activation_pre_process[0](x)
        if isinstance(y, Tensor):
            y = obj.activation_pre_process[1](y)

    if isinstance(x, QTensor):
        x = x.as_subclass(Tensor)
    if isinstance(y, QTensor):
        y = y.as_subclass(Tensor)

    output = super(type(obj), obj).forward(x, y)

    if obj.activation_post_process is None:
        return output
    else:
        return obj.activation_post_process(output)


def multi_input_forward(obj, *args, **kwargs):
    assert obj.activation_pre_process is None, "Unsupported"

    output = super(type(obj), obj).forward(
        *pytree_convert(args, QTensor, lambda x: x.as_subclass(Tensor)),
        **pytree_convert(kwargs, QTensor, lambda x: x.as_subclass(Tensor)),
    )
    if obj.activation_post_process is None:
        return output
    else:
        return obj.activation_post_process(output)


def common_from_float(cls, mod):
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

    mod.__class__ = cls
    init_qat(mod)

    return mod


class QATModuleMeta(type):
    def __new__(cls, clsname, bases, attrs, input_num=1):
        if input_num is not None:
            assert isinstance(input_num, int)

        if "__init__" in attrs:
            # append class defined init after common_init and common_from_float
            custom_init = attrs["__init__"]

            def init(obj, *args, **kwargs):
                common_init(obj, *args, **kwargs)
                custom_init(obj)

            def from_float(cls, *args, **kwargs):
                mod = common_from_float(cls, *args, **kwargs)
                custom_init(mod)

                return mod

        else:
            init = common_init
            from_float = common_from_float

        if "forward" not in attrs:
            # define forward according to input_num to improve performance
            if input_num == 1:
                forward = single_input_forward
            elif input_num == 2:
                forward = double_input_forward
            else:
                forward = multi_input_forward

            attrs["forward"] = forward

        if "_FLOAT_MODULE" not in attrs:
            attrs["_FLOAT_MODULE"] = bases[0]
        attrs["__init__"] = init
        attrs["from_float"] = classmethod(from_float)
        attrs["_input_num"] = input_num

        new_cls = type(clsname, bases, attrs)

        register_float_to_qat_mapping(new_cls._FLOAT_MODULE, new_cls)

        return new_cls


###############################################################################
# This is an example to explain how to use QATModuleMeta
###############################################################################


# This is a float operation
class FloatModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return input


# And we can write its QAT implementation as following
class QATModuleExample(FloatModule, metaclass=QATModuleMeta, input_num=1):
    # Optional: Define __init__ to do custom modification
    def __init__(self) -> None:
        pass

    # Optional: Define forward to cover default forward
    # generated by QATModuleMeta
    def forward(self, input):
        return input


# The output class act like following code:
class QATModuleExampleImpl(FloatModule):
    def __init__(self, *args, **kwargs) -> None:
        self.qconfig = kwargs.pop("qconfig", None)
        FloatModule.__init__(*args, **kwargs)
        init_qat(self)
        if hasattr(QATModuleExample, "__init__"):
            QATModuleExample.__init__(self)

    def forward(self, input):
        if hasattr(QATModuleExample, "forward"):
            return QATModuleExample.forward(self, input.as_subclass(Tensor))
        else:
            return single_input_forward(self, input)

    @classmethod
    def from_float(cls, mod: FloatModule):
        mod.__class__ = cls
        init_qat(mod)
        if hasattr(QATModuleExample, "__init__"):
            QATModuleExample.__init__(mod)

        return mod


###############################################################################


class BiasHook:
    _enabled = False
    _input_scale = None

    @classmethod
    def set_input_scale(cls, value):
        cls._input_scale = value

    @classmethod
    def process(cls, mod, bias: Tensor):
        return bias

    @classmethod
    @contextmanager
    def enable(cls):
        try:
            cls._enabled = True
            yield
        finally:
            cls._enabled = False


class WeightHook:
    _enabled = False

    @classmethod
    def process(cls, mod, weight: Tensor):
        if cls._enabled:
            return mod.weight_fake_quant(weight).as_subclass(Tensor)
        else:
            return weight

    @classmethod
    @contextmanager
    def enable(cls):
        try:
            cls._enabled = True
            yield
        finally:
            cls._enabled = False


def weight_getter(obj):
    w = super(type(obj), obj).__getattr__("weight")
    return WeightHook.process(obj, w)


def weight_setter(obj, value):
    super(type(obj), obj).__setattr__("weight", value)


def weight_deler(obj, value):
    super(type(obj), obj).__delattr__("weight", value)


def bias_getter(obj):
    b = super(type(obj), obj).__getattr__("bias")
    return BiasHook.process(obj, b)


def bias_setter(obj, value):
    super(type(obj), obj).__setattr__("bias", value)


def bias_deler(obj, value):
    super(type(obj), obj).__delattr__("bias", value)


def init_qat_conv(obj):
    if obj.qconfig.weight is None:
        raise ValueError("qconfig must include weight")
    if (
        hasattr(obj.qconfig, "activation")
        and obj.qconfig.activation is not None
        and is_float(obj.qconfig.activation())
    ):
        obj.weight_fake_quant = obj.qconfig.weight()
    else:
        obj.weight_fake_quant = obj.qconfig.weight(
            channel_len=obj.out_channels
        )
    if get_march() == March.BERNOULLI:
        obj.register_buffer(
            "bias_scale", torch.ones(obj.out_channels, dtype=torch.float32)
        )


def conv_init(obj, *args, **kwargs):
    common_init(obj, *args, **kwargs)
    init_qat_conv(obj)


def conv_from_float(cls, mod):
    mod = common_from_float(cls, mod)
    init_qat_conv(mod)
    return mod


def conv_forward(obj, input: Union[Tensor, QTensor]):
    if obj.activation_pre_process is not None:
        input = obj.activation_pre_process(input)

    if get_march() == March.BERNOULLI and obj.bias is not None:
        BiasHook.set_input_scale(input.q_scale())
    with WeightHook.enable(), BiasHook.enable():
        output = super(type(obj), obj).forward(input.as_subclass(Tensor))

    if obj.activation_post_process is None:
        return output
    else:
        return obj.activation_post_process(output)


def conv_add_forward(
    obj, x: Union[Tensor, QTensor], y: Union[Tensor, QTensor]
):
    if obj.activation_pre_process is not None:
        x = obj.activation_pre_process[0](x)
        y = obj.activation_pre_process[1](y)

    if get_march() == March.BERNOULLI and obj.bias is not None:
        BiasHook.set_input_scale(x.q_scale())
    with WeightHook.enable(), BiasHook.enable():
        output = super(type(obj), obj).forward(
            x.as_subclass(Tensor), y.as_subclass(Tensor)
        )

    if obj.activation_post_process is None:
        return output
    else:
        return obj.activation_post_process(output)


class QATConvMeta(type):
    def __new__(cls, clsname, bases, attrs, input_num=1):
        if "__init__" in attrs:
            # append class defined init after common_init and common_from_float
            custom_init = attrs["__init__"]

            def init(obj, *args, **kwargs):
                conv_init(obj, *args, **kwargs)
                custom_init(obj)

            def from_float(cls, *args, **kwargs):
                mod = conv_from_float(cls, *args, **kwargs)
                custom_init(mod)

                return mod

        else:
            init = conv_init
            from_float = conv_from_float

        if "forward" not in attrs:
            # define forward according to input_num to improve performance
            if input_num == 1:
                forward = conv_forward
            elif input_num == 2:
                forward = conv_add_forward
            else:
                raise ValueError("QATConvMeta only support input_num = 1 or 2")

            attrs["forward"] = forward

        if "_FLOAT_MODULE" not in attrs:
            attrs["_FLOAT_MODULE"] = bases[0]
        attrs["__init__"] = init
        attrs["weight"] = property(weight_getter, weight_setter, weight_deler)
        attrs["bias"] = property(bias_getter, bias_setter, bias_deler)
        # attrs["prepare_convert"] = prepare_convert
        attrs["from_float"] = classmethod(from_float)
        attrs["_input_num"] = input_num

        new_cls = type(clsname, bases, attrs)

        register_float_to_qat_mapping(new_cls._FLOAT_MODULE, new_cls)

        return new_cls


def pre_process(input_pre_process, *args):
    """Transform the input Tensor according to input_pre_process.

    1. If the length of input_pre_process is the same as the length of args,
    then directly apply the corresponding transformation to each input.
    2. If the length of input_pre_process is less than the length of args,
    then apply the transformation to the preceding inputs and leave the
    remaining inputs unchanged.

    Args:
        input_pre_process: The instance of FakeCast or FakeQuantize initialized
            from the qconfig
        args: the input tensor

    Returns:
        The transformed input
    """

    if input_pre_process is None:
        # return original args
        return args if len(args) > 1 else args[0]

    def impl(pre_process, x):
        if isinstance(x, Tensor):
            return pre_process(x)
        else:
            logger.warning(
                "The input is not Tensor, return the identity directly"
            )
            return x

    if len(args) == 1:
        return impl(input_pre_process, args[0])
    else:
        ret = []
        if isinstance(input_pre_process, torch.nn.ModuleList):
            for i in range(min(len(args), len(input_pre_process))):
                ret.append(impl(input_pre_process[i], args[i]))
            for i in range(len(input_pre_process), len(args)):
                logger.warning(
                    "The length of input_pre_process is less than the input, "
                    "return the identity for the rest of inputs",
                    extra={"call_times_context": ("message")},
                )
                ret.append(args[i])
            return ret
        else:
            logger.warning(
                "Only one qconfig of input is provided while several inputs are given, FakeCast or FakeQuantize the first input and return the rest directly"  # noqa: E501
            )
            ret.append(impl(input_pre_process, args[0]))
            for i in range(1, len(args)):
                ret.append(args[i])
            return ret


def init_input_preprocess(qconfig):
    """Init the input qconfig from the op qconfig.

    1. The length of input qconfig is equal to the length of input,
    then init the input qconfig directly.
    2. The length of input qconfig is less than the length of input, then:
        a. Init the qconfig for the fronter inputs
        b. Use nn.Identity for the rest inputs


    Args:
        qconfig: the qconfig of ops

    Returns:
        The input preprocess for the op
    """
    input_pre_process = None
    if hasattr(qconfig, "input") and qconfig.input is not None:
        if isinstance(qconfig.input, Iterable):
            input_pre_process = torch.nn.ModuleList(
                [nn.Identity() if q is None else q() for q in qconfig.input]
            )
        else:
            input_pre_process = qconfig.input()
    return input_pre_process


def is_float(activation_post_process):
    from horizon_plugin_pytorch.quantization.fake_cast import FakeCast

    return isinstance(activation_post_process, FakeCast)
