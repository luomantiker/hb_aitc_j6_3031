import sys
from collections import namedtuple
from functools import partial
from typing import Dict, Optional, Type, Union

import torch
from torch import nn
from torch.quantization import QConfig as TorchQconfig

from horizon_plugin_pytorch.dtype import (
    QuantDType,
    qinfo,
    qint8,
    qint16,
    quint4,
)
from horizon_plugin_pytorch.utils import deprecated_module_attr_warning
from horizon_plugin_pytorch.utils.typeguard import typechecked
from ._learnable_fake_quantize import (
    _LearnableFakeQuantize,
    default_4bit_lsq_quant,
    default_8bit_lsq_quant,
    default_16bit_lsq_quant,
    default_uint4_lsq_quant,
    default_weight_4bit_lsq_quant,
    default_weight_8bit_lsq_quant,
    default_weight_16bit_lsq_quant,
)
from .fake_cast import FakeCast
from .fake_quantize import (
    CalibFakeQuantize,
    FakeQuantize,
    default_4bit_fake_quant,
    default_8bit_fake_quant,
    default_16bit_fake_quant,
    default_calib_fake_quant,
    default_uint4_fake_quant,
    default_weight_4bit_fake_quant,
    default_weight_8bit_fake_quant,
    default_weight_16bit_fake_quant,
    default_weight_calib_fake_quant,
    per_channel_8bit_fake_quant,
)
from .observer_v2 import (
    ClipObserver,
    ClipStdObserver,
    FixedScaleObserver,
    KLObserver,
    MinMaxObserver,
    MixObserver,
    MSEObserver,
    ObserverBase,
    PercentileObserver,
)
from .pact_fake_quantize import (
    PACTFakeQuantize,
    default_4bit_pact_quant,
    default_8bit_pact_quant,
    default_16bit_pact_quant,
    default_uint4_pact_quant,
)


class QConfig(
    namedtuple("QConfig", ["activation", "weight", "input", "output"]),
    TorchQconfig,
):
    def __new__(cls, activation=None, weight=None, input=None, output=None):
        # catch common mistakes
        if (
            isinstance(activation, nn.Module)
            or isinstance(weight, nn.Module)
            or isinstance(input, nn.Module)
            or isinstance(output, nn.Module)
        ):
            raise ValueError(
                "QConfig received observer instance, "
                "please pass observer class instead."
            )
        assert (
            output is None or activation is None
        ), "Only one can be set for `activation` and `output` of QConfig"
        return super(QConfig, cls).__new__(
            cls, activation if output is None else output, weight, input, None
        )


# fake_quantize
default_qat_8bit_qconfig = QConfig(
    activation=default_8bit_fake_quant, weight=default_weight_8bit_fake_quant
)


# This qconfig will make the OP OUTPUT per channel quantized activation.
# Please note that only depthwise_conv, interpolate and add support per channel
# quantized INPUT and OUTPUT. Take care of the next OP when using this qconfig!
# Model trained with this qconfig cannot be compiled now.
per_channel_qat_8bit_qconfig = QConfig(
    activation=per_channel_8bit_fake_quant,
    weight=default_weight_8bit_fake_quant,
)

default_qat_4bit_qconfig = QConfig(
    activation=default_4bit_fake_quant, weight=default_weight_4bit_fake_quant
)
default_qat_out_8bit_qconfig = QConfig(
    activation=None, weight=default_weight_8bit_fake_quant
)

default_qat_out_4bit_qconfig = QConfig(
    activation=None, weight=default_weight_4bit_fake_quant
)

default_calib_qconfig = QConfig(
    activation=default_calib_fake_quant, weight=default_weight_calib_fake_quant
)


def _get_fake_quant(dtype, fake_quant_name, fake_quant_mapping, qkwargs):
    if fake_quant_name is None:
        return None
    assert fake_quant_name in fake_quant_mapping.keys(), (
        "unsupport fake_quant_name" + fake_quant_name
    )
    if qkwargs is not None:
        if "dtype" in qkwargs:
            dtype = qkwargs["dtype"]
        if "quant_min" in qkwargs:
            min = qkwargs["quant_min"]
            assert (
                qinfo(dtype).min == min
            ), f"expect quant_min = {qinfo(dtype).min} but get {min}"
        if "quant_max" in qkwargs:
            max = qkwargs["quant_max"]
            assert (
                qinfo(dtype).max == max
            ), f"expect quant_max = {qinfo(dtype).max} but get {max}"

    assert dtype in fake_quant_mapping[fake_quant_name].keys(), (
        "unsupport dtype " + dtype + " for " + fake_quant_name
    )
    fake_quant = fake_quant_mapping[fake_quant_name][dtype]
    if qkwargs is not None:
        fake_quant = fake_quant.with_args(**qkwargs)
    return fake_quant


def _get_custom_qconfig(
    dtype="qint8",
    weight_dtype="qint8",
    activation_fake_quant="fake_quant",
    weight_fake_quant="fake_quant",
    activation_qkwargs=None,
    weight_qkwargs=None,
    backend="",
):
    activation_fake_quant_mapping = {
        "fake_quant": {
            "qint16": default_16bit_fake_quant,
            "qint8": default_8bit_fake_quant,
            "qint4": default_4bit_fake_quant,
            "quint4": default_uint4_fake_quant,
        },
        "lsq": {
            "qint16": default_16bit_lsq_quant,
            "qint8": default_8bit_lsq_quant,
            "qint4": default_4bit_lsq_quant,
            "quint4": default_uint4_lsq_quant,
        },
        "pact": {
            "qint16": default_16bit_pact_quant,
            "qint8": default_8bit_pact_quant,
            "qint4": default_4bit_pact_quant,
            "quint4": default_uint4_pact_quant,
        },
    }
    weight_fake_quant_mapping = {
        "fake_quant": {
            "qint8": default_weight_8bit_fake_quant,
            "qint4": default_weight_4bit_fake_quant,
            "qint16": default_weight_16bit_fake_quant,
        },
        "lsq": {
            "qint8": default_weight_8bit_lsq_quant,
            "qint4": default_weight_4bit_lsq_quant,
            "qint16": default_weight_16bit_lsq_quant,
        },
        "pact": {
            "qint8": default_weight_8bit_lsq_quant,
            "qint4": default_weight_4bit_lsq_quant,
            "qint16": default_weight_16bit_lsq_quant,
        },
    }
    activation = _get_fake_quant(
        dtype,
        activation_fake_quant,
        activation_fake_quant_mapping,
        activation_qkwargs,
    )
    weight = _get_fake_quant(
        weight_dtype,
        weight_fake_quant,
        weight_fake_quant_mapping,
        weight_qkwargs,
    )
    return QConfig(activation=activation, weight=weight)


def get_default_qat_qconfig(
    dtype="qint8",
    weight_dtype="qint8",
    activation_fake_quant="fake_quant",
    weight_fake_quant="fake_quant",
    activation_qkwargs=None,
    weight_qkwargs=None,
    backend="",
):
    """Get default qat qconfig.

    Args:
        dtype (str): Activation quantization type, the allowable values is
                     qint8 and qint16
        weight_dtype (str): Weight quantization type, the allowable values
                     is qint8 and qint16
        activation_fake_quant (str): FakeQuantize type of activation, default
                                     is fake_quant. Avaliable items is
                                     fake_quant, lsq, pact
        weight_fake_quant (str): FakeQuantize type of weight, default is
                                 fake_quant.Avaliable items is fake_quant, lsq
                                 and pact
        activation_qkwargs(dict): A dict contain activation Observer type, args
                                  of activation FakeQuantize and args of
                                  activation Observer.
        weight_qkwargs(dict): A dict contain weight Observer type, args of
                              weight FakeQuantize and args of weight Observer.
        backend (str): backend implementation
    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.qconfig.get_default_qat_qconfig"
    )
    assert dtype in (
        "qint4",
        "qint8",
        "qint16",
        "quint4",
    ), f"unsupported activation dtype: {dtype}"
    assert weight_dtype in (
        "qint4",
        "qint8",
        "qint16",
    ), f"unsupported weight dtype: {dtype}"
    if activation_qkwargs is not None:
        assert isinstance(activation_qkwargs, dict), (
            "activation qkwargs must be a dict, but get a "
            + type(activation_qkwargs).__name__
        )
    if weight_qkwargs is not None:
        assert isinstance(weight_qkwargs, dict), (
            "activation qkwargs must be a dict, but get a "
            + type(weight_qkwargs).__name__
        )
    return _get_custom_qconfig(
        dtype=dtype,
        weight_dtype=weight_dtype,
        activation_fake_quant=activation_fake_quant,
        weight_fake_quant=weight_fake_quant,
        activation_qkwargs=activation_qkwargs,
        weight_qkwargs=weight_qkwargs,
        backend=backend,
    )


def get_default_qat_out_qconfig(
    dtype="qint8",
    weight_fake_quant="fake_quant",
    weight_qkwargs=None,
    backend="",
):
    """Get default qat out qconfig.

    Args:
        dtype (str): quantization type, the allowable value is qint8 and qint16
        weight_fake_quant (str): FakeQuantize type of weight, default is
                                 fake_quant.Avaliable items is fake_quant, lsq
                                 and pact
        weight_qkwargs(dict): A dict contain weight Observer type, args of
                              weight FakeQuantize and args of weight Observer.
        backend (str): backend implementation
    """
    assert dtype in (
        "qint4",
        "qint8",
        "qint16",
        "quint4",
    ), f"unsupported dtype: {dtype}"
    if weight_qkwargs is not None:
        assert isinstance(weight_qkwargs, dict), (
            "weight qkwargs must be a dict, but get a "
            + type(weight_qkwargs).__name__
        )
    return _get_custom_qconfig(
        dtype=dtype,
        activation_fake_quant=None,
        weight_fake_quant=weight_fake_quant,
        weight_qkwargs=weight_qkwargs,
        backend="",
    )


def get_default_calib_qconfig(dtype="qint8", calib_qkwargs=None, backend=""):
    """Get default calibration qconfig.

    Args:
        dtype (str): quantization type, the allowable value is qint8 and qint16
        calib_qkwargs(dict): A dict that contains args of CalibFakeQuantize and
            args of calibration observer.
        backend (str): backend implementation
    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.qconfig.get_default_calib_qconfig"
    )
    assert dtype in (
        "qint8",
        "qint16",
    ), f"unsupported dtype: {dtype}"
    if calib_qkwargs is not None:
        assert isinstance(calib_qkwargs, dict)
        calib_qconfig = QConfig(
            activation=CalibFakeQuantize.with_args(
                dtype=dtype,
                **calib_qkwargs,
            ),
            weight=default_weight_calib_fake_quant,
        )
    else:
        calib_qconfig = QConfig(
            activation=default_calib_fake_quant.with_args(
                dtype=dtype,
            ),
            weight=default_weight_calib_fake_quant,
        )
    return calib_qconfig


@typechecked
def get_default_qconfig(
    activation_fake_quant: Optional[str] = "fake_quant",
    weight_fake_quant: Optional[str] = "fake_quant",
    activation_observer: Optional[str] = "min_max",
    weight_observer: Optional[str] = "min_max",
    activation_qkwargs: Optional[Dict] = None,
    weight_qkwargs: Optional[Dict] = None,
):
    """Get default qconfig.

    Args:
        activation_fake_quant: FakeQuantize type of activation, default is
            fake_quant. Avaliable items are fake_quant, lsq, pact.
        weight_fake_quant: FakeQuantize type of weight, default is fake_quant.
            Avaliable items are fake_quant, lsq and pact.
        activation_observer: Observer type of activation, default is min_max.
            Avaliable items are min_max, fixed_scale, clip, percentile,
            clip_std, mse, kl.
        weight_observer: Observer type of weight, default is min_max. Avaliable
            items are min_max, fixed_scale, clip, percentile, clip_std, mse.
        activation_qkwargs: A dict contain activation Observer type, args of
            activation FakeQuantize and args of activation Observer.
        weight_qkwargs: A dict contain weight Observer type, args of weight
            FakeQuantize and args of weight Observer.
    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.qconfig.get_default_qconfig"
    )
    if weight_qkwargs is None:
        weight_qkwargs = {}

    if "qscheme" not in weight_qkwargs:
        weight_qkwargs["qscheme"] = torch.per_channel_symmetric
    if ("ch_axis" not in weight_qkwargs) and (
        weight_qkwargs["qscheme"] == torch.per_channel_symmetric
    ):
        weight_qkwargs["ch_axis"] = 0

    def _get_fake_quant(fake_quant_name, observer_name, qkwargs):
        if fake_quant_name is None:
            return None
        fake_quant_dict = {
            "fake_quant": FakeQuantize,
            "lsq": _LearnableFakeQuantize,
            "pact": PACTFakeQuantize,
        }
        assert fake_quant_name in fake_quant_dict.keys(), (
            "unsupport fake_quant_name" + fake_quant_name
        )
        fake_quant = fake_quant_dict.get(fake_quant_name)

        observer_dict = {
            "min_max": MinMaxObserver,
            "fixed_scale": FixedScaleObserver,
            "clip": ClipObserver,
            "percentile": PercentileObserver,
            "clip_std": ClipStdObserver,
            "mse": MSEObserver,
            "kl": KLObserver,
            "mix": MixObserver,
        }
        assert observer_name in observer_dict.keys(), (
            "unsupport observer_name" + observer_name
        )
        observer = observer_dict.get(observer_name)
        fake_quant = fake_quant.with_args(observer=observer)
        if qkwargs is not None:
            fake_quant = fake_quant.with_args(**qkwargs)
        return fake_quant

    activation = _get_fake_quant(
        activation_fake_quant,
        activation_observer,
        activation_qkwargs,
    )
    weight = _get_fake_quant(
        weight_fake_quant,
        weight_observer,
        weight_qkwargs,
    )
    return QConfig(activation=activation, weight=weight)


default_calib_8bit_fake_quant_qconfig = get_default_qconfig(
    activation_observer="mse",
)

default_qat_8bit_fake_quant_qconfig = get_default_qconfig()

default_qat_8bit_fixed_act_fake_quant_qconfig = get_default_qconfig(
    activation_qkwargs={
        "averaging_constant": 0,
    },
)

default_calib_8bit_weight_16bit_act_fake_quant_qconfig = get_default_qconfig(
    activation_observer="mse",
    activation_qkwargs={
        "dtype": qint16,
    },
)

default_qat_8bit_weight_16bit_act_fake_quant_qconfig = get_default_qconfig(
    activation_qkwargs={
        "dtype": qint16,
    },
)

default_qat_8bit_weight_16bit_fixed_act_fake_quant_qconfig = (
    get_default_qconfig(
        activation_qkwargs={
            "dtype": qint16,
            "averaging_constant": 0,
        },
    )
)

default_qat_8bit_weight_32bit_out_fake_quant_qconfig = get_default_qconfig(
    activation_fake_quant=None,
    activation_observer=None,
)

default_calib_8bit_weight_32bit_out_fake_quant_qconfig = (
    default_qat_8bit_weight_32bit_out_fake_quant_qconfig
)

default_qat_8bit_pact_quant_qconfig = get_default_qconfig(
    activation_fake_quant="pact",
    weight_fake_quant="pact",
    activation_observer="min_max",
    weight_observer="min_max",
    activation_qkwargs=None,
    weight_qkwargs={
        "qscheme": torch.per_channel_symmetric,
        "ch_axis": 0,
    },
)

default_qat_16bit_pact_quant_qconfig = get_default_qconfig(
    activation_fake_quant="pact",
    weight_fake_quant="pact",
    activation_observer="min_max",
    weight_observer="min_max",
    activation_qkwargs={
        "dtype": qint16,
    },
    weight_qkwargs={
        "qscheme": torch.per_channel_symmetric,
        "ch_axis": 0,
    },
)

default_qat_8bit_lsq_quant_qconfig = get_default_qconfig(
    activation_fake_quant="lsq",
    weight_fake_quant="lsq",
    activation_observer="min_max",
    weight_observer="min_max",
    activation_qkwargs={
        "use_grad_scaling": True,
        "averaging_constant": 1.0,
    },
    weight_qkwargs={
        "qscheme": torch.per_channel_symmetric,
        "ch_axis": 0,
        "use_grad_scaling": True,
        "averaging_constant": 1.0,
    },
)

default_qat_16bit_lsq_quant_qconfig = get_default_qconfig(
    activation_fake_quant="lsq",
    weight_fake_quant="lsq",
    activation_observer="min_max",
    weight_observer="min_max",
    activation_qkwargs={
        "dtype": qint16,
        "use_grad_scaling": True,
        "averaging_constant": 1.0,
    },
    weight_qkwargs={
        "qscheme": torch.per_channel_symmetric,
        "ch_axis": 0,
        "use_grad_scaling": True,
        "averaging_constant": 1.0,
    },
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "default_qat_out_8bit_fake_quant_qconfig",
    "1.6.2",
    "1.9.0",
    "default_qat_8bit_weight_32bit_out_fake_quant_qconfig",
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "default_calib_out_8bit_fake_quant_qconfig",
    "1.6.2",
    "1.9.0",
    "default_calib_8bit_weight_32bit_out_fake_quant_qconfig",
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "default_qat_16bit_fake_quant_qconfig",
    "1.7.0",
    "1.9.0",
    "default_qat_8bit_weight_16bit_act_fake_quant_qconfig",
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "default_calib_16bit_fake_quant_qconfig",
    "1.7.0",
    "1.9.0",
    "default_calib_8bit_weight_16bit_act_fake_quant_qconfig",
)


def replace_qconfig_dtype(
    qconfig: QConfig, activation_dtype, weight_dtype=None
):
    activation_config = qconfig.activation
    if activation_config is None:
        activation_config = qconfig.output

    if (
        activation_config is None
        or activation_config().get_dtype() == activation_dtype
    ):
        new_activation = activation_config
    else:
        new_activation = partial(
            activation_config,
            quant_min=qinfo(activation_dtype).min,
            quant_max=qinfo(activation_dtype).max,
            dtype=activation_dtype,
            qscheme=(
                torch.per_tensor_affine
                if activation_dtype == quint4
                else torch.per_tensor_symmetric
            ),
        )
    if (
        weight_dtype is None
        or qconfig.weight is None
        or qconfig.weight().get_dtype() == weight_dtype
    ):
        new_weight = qconfig.weight
    else:
        new_weight = partial(
            qconfig.weight,
            quant_min=qinfo(weight_dtype).min,
            quant_max=qinfo(weight_dtype).max,
            dtype=weight_dtype,
            qscheme=(
                torch.per_channel_affine
                if weight_dtype == quint4
                else torch.per_channel_symmetric
            ),
        )

    return QConfig(
        activation=new_activation,
        weight=new_weight,
        input=qconfig.input,
    )


def promote_int8_activation_to_int16(qconfig):
    if qconfig is None:
        return None
    elif qconfig.activation().get_dtype() == qint8:
        return replace_qconfig_dtype(qconfig, qint16)
    else:
        return qconfig


@typechecked
def get_qconfig(
    observer: Type[ObserverBase] = MinMaxObserver,
    in_dtype: Optional[Union[torch.dtype, QuantDType]] = None,
    weight_dtype: Optional[Union[torch.dtype, QuantDType]] = qint8,
    out_dtype: Optional[Union[torch.dtype, QuantDType]] = qint8,
    fix_scale: bool = False,
):
    """
    Get qconfig.

    Args:
        observer: observer type for input and output. Support
            `MinMaxObserver` and `MSEObserver`
        in_dtype: input dtype.
        weight_dtype: weight dtype.
        out_dtype: output dtype.
        fix_scale: Whether fix input/output scale.
    """
    assert observer in (MinMaxObserver, MSEObserver), (
        f"{observer} is not supported by `get_qconfig`, define qconfig with"
        " horizon_plugin_pytorch.quantization.QConfig"
    )

    def get_fake_quant(is_weight, dtype):
        if dtype in (torch.float16, torch.float32):
            return FakeCast.with_args(dtype=dtype)
        elif dtype in (qint8, qint16):
            if is_weight:
                return FakeQuantize.with_args(
                    observer=MinMaxObserver,
                    dtype=dtype,
                    qscheme=torch.per_channel_symmetric,
                    ch_axis=0,
                )
            if fix_scale:
                return FakeQuantize.with_args(
                    observer=observer, dtype=dtype, averaging_constant=0
                )
            else:
                return FakeQuantize.with_args(observer=observer, dtype=dtype)
        elif dtype is None:
            return None
        else:
            raise ValueError(f"unsupported qconfig dtype {dtype}")

    return QConfig(
        input=get_fake_quant(False, in_dtype),
        weight=get_fake_quant(True, weight_dtype),
        output=get_fake_quant(False, out_dtype),
    )
