from enum import Enum

import horizon_plugin_pytorch as horizon
import torch

__all__ = [
    "set_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qat_out_qconfig",
    "get_default_calibration_qconfig",
    "get_qconfig_mode",
    "set_qconfig_mode",
    "get_qconfig",
    "get_default_qconfig",
    "get_default_out_qconfig",
    "QconfigMode",
]

global_qat_qconfig = horizon.quantization.get_default_qat_qconfig()
global_qat_out_qconfig = horizon.quantization.get_default_qat_out_qconfig()

try:
    global_calibration_qconfig = (
        horizon.quantization.get_default_calib_qconfig()
    )  # noqa E501
except Exception:
    global_calibration_qconfig = None


class QconfigMode(Enum):
    COMPATIBLE = "compatible"
    QAT = "qat"
    CALIBRATION = "calibration"


_qconfig_mode = QconfigMode.COMPATIBLE


def get_qconfig_mode():
    return _qconfig_mode


def set_qconfig_mode(mode):
    assert type(mode) == QconfigMode, "only support mode in `QconfigMode`"
    global _qconfig_mode
    _qconfig_mode = mode


try:
    global_qat_qconfig_v2 = horizon.quantization.get_default_qconfig(
        activation_fake_quant="fake_quant",
        weight_fake_quant="fake_quant",
        activation_observer="min_max",
        weight_observer="min_max",
        activation_qkwargs=None,
        weight_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
        },
    )
    global_qat_out_qconfig_v2 = horizon.quantization.get_default_qconfig(
        activation_fake_quant=None,
        weight_fake_quant="fake_quant",
        activation_observer=None,
        weight_observer="min_max",
        activation_qkwargs=None,
        weight_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
        },
    )
    global_calibration_qconfig_v2 = horizon.quantization.get_default_qconfig(
        activation_fake_quant="fake_quant",
        weight_fake_quant="fake_quant",
        activation_observer="mse",
        weight_observer="min_max",
        activation_qkwargs=None,
        weight_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
        },
    )
    global_calibration_out_qconfig_v2 = (
        horizon.quantization.get_default_qconfig(
            activation_fake_quant=None,
            weight_fake_quant="fake_quant",
            activation_observer=None,
            weight_observer="min_max",
            activation_qkwargs=None,
            weight_qkwargs={
                "qscheme": torch.per_channel_symmetric,
                "ch_axis": 0,
            },
        )
    )
except Exception:
    global_qat_qconfig_v2 = global_qat_qconfig
    global_qat_out_qconfig_v2 = global_qat_out_qconfig
    global_calibration_qconfig_v2 = global_calibration_qconfig
    global_calibration_out_qconfig_v2 = None


def set_default_qconfig(
    activation_fake_quant: str = "fake_quant",
    weight_fake_quant: str = "fake_quant",
    activation_qat_observer: str = "min_max",
    weight_qat_observer: str = "min_max",
    activation_calibration_observer: str = "mse",
    weight_calibration_observer: str = "min_max",
    activation_qat_qkwargs: dict = None,
    weight_qat_qkwargs: dict = None,
    activation_calibration_qkwargs: dict = None,
    weight_calibration_qkwargs: dict = None,
):
    """Set default qat qconfig.

    Args:
        activation_fake_quant: FakeQuantize type of activation, default is
            "fake_quant". Avaliable items are "fake_quant", "lsq", "pact".

        weight_fake_quant: FakeQuantize type of weight, default is fake_quant.
            Avaliable items are "fake_quant", "lsq" and "pact".

        activation_qat_observer: observer type of qat activation, default is
            "min_max". Avaliable items are "min_max", "fixed_scale", "clip",
            "percentile", "clip_std", "mse", "kl", "mix".

        weight_qat_observer: observer type of qat weight, default is "min_max".
            Avaliable items are "min_max", "fixed_scale", "clip","percentile",
            "clip_std", "mse", "kl", "mix".

        activation_calibration_observer: observer type of calibration
            activation, default is "mse". Avaliable items are "min_max",
            "fixed_scale", "clip","percentile", "clip_std", "mse", "kl", "mix".

        weight_calibration_observer: observer type of calibration weight,
            default is "min_max". Avaliable items are "min_max", "fixed_scale",
            "clip","percentile", "clip_std", "mse", "kl", "mix".

        activation_qat_qkwargs: A dict contain args of qat activation
            FakeQuantize and Observer.

        weight_qat_qkwargs: A dict contain args of qat weight FakeQuantize and
            Observer.

        activation_calibration_qkwargs: A dict contain args of calibration
            activation FakeQuantize and Observer.

        weight_calibration_qkwargs: A dict contain args of calibration weight
            FakeQuantize and Observer.

    """

    global global_qat_qconfig_v2
    global global_qat_out_qconfig_v2
    global global_calibration_qconfig_v2
    global global_calibration_out_qconfig_v2

    try:
        if weight_qat_qkwargs is None:
            weight_qat_qkwargs = {}
        if weight_calibration_qkwargs is None:
            weight_calibration_qkwargs = {}

        for wdict in (weight_qat_qkwargs, weight_calibration_qkwargs):
            if "qscheme" not in wdict:
                wdict["qscheme"] = torch.per_channel_symmetric
            if (
                "ch_axis" not in wdict
                and wdict["qscheme"] == torch.per_channel_symmetric
            ):
                wdict["ch_axis"] = 0
        global_qat_qconfig_v2 = horizon.quantization.get_default_qconfig(
            activation_fake_quant=activation_fake_quant,
            weight_fake_quant=weight_fake_quant,
            activation_observer=activation_qat_observer,
            weight_observer=weight_qat_observer,
            activation_qkwargs=activation_qat_qkwargs,
            weight_qkwargs=weight_qat_qkwargs,
        )
        global_qat_out_qconfig_v2 = horizon.quantization.get_default_qconfig(
            activation_fake_quant=None,
            weight_fake_quant=weight_fake_quant,
            activation_observer=None,
            weight_observer=weight_qat_observer,
            activation_qkwargs=None,
            weight_qkwargs=weight_qat_qkwargs,
        )
        global_calibration_qconfig_v2 = (
            horizon.quantization.get_default_qconfig(
                activation_fake_quant=activation_fake_quant,
                weight_fake_quant=weight_fake_quant,
                activation_observer=activation_calibration_observer,
                weight_observer=weight_calibration_observer,
                activation_qkwargs=activation_calibration_qkwargs,
                weight_qkwargs=weight_calibration_qkwargs,
            )
        )
        global_calibration_out_qconfig_v2 = (
            horizon.quantization.get_default_qconfig(
                activation_fake_quant=None,
                weight_fake_quant=weight_fake_quant,
                activation_observer=None,
                weight_observer=weight_calibration_observer,
                activation_qkwargs=None,
                weight_qkwargs=weight_calibration_qkwargs,
            )
        )
    except AttributeError:
        pass


def get_default_qat_qconfig():
    if get_qconfig_mode() == QconfigMode.COMPATIBLE:
        return global_qat_qconfig
    elif get_qconfig_mode() == QconfigMode.CALIBRATION:
        return global_calibration_qconfig_v2
    elif get_qconfig_mode() == QconfigMode.QAT:
        return global_qat_qconfig_v2


def get_default_qat_out_qconfig():
    if get_qconfig_mode() == QconfigMode.COMPATIBLE:
        return global_qat_out_qconfig
    elif get_qconfig_mode() == QconfigMode.CALIBRATION:
        return global_calibration_out_qconfig_v2
    elif get_qconfig_mode() == QconfigMode.QAT:
        return global_qat_out_qconfig_v2


def get_default_calibration_qconfig():
    return global_calibration_qconfig


def get_default_qconfig():
    if get_qconfig_mode() == QconfigMode.COMPATIBLE:
        raise ValueError(
            "`get_default_qconfig` is designed for new qconfig, "
            "make sure `QconfigMode` is not `COMPATIBLE`"
        )
    elif get_qconfig_mode() == QconfigMode.CALIBRATION:
        return global_calibration_qconfig_v2
    elif get_qconfig_mode() == QconfigMode.QAT:
        return global_qat_qconfig_v2


def get_default_out_qconfig():
    if get_qconfig_mode() == QconfigMode.COMPATIBLE:
        raise ValueError(
            "`get_default_out_qconfig` is designed for new qconfig, "
            "make sure `QconfigMode` is not `COMPATIBLE`"
        )
    elif get_qconfig_mode() == QconfigMode.CALIBRATION:
        return global_calibration_out_qconfig_v2
    elif get_qconfig_mode() == QconfigMode.QAT:
        return global_qat_out_qconfig_v2


def get_qconfig(
    activation_fake_quant: str = "fake_quant",
    weight_fake_quant: str = "fake_quant",
    activation_qat_observer: str = "min_max",
    weight_qat_observer: str = "min_max",
    activation_calibration_observer: str = "mse",
    weight_calibration_observer: str = "min_max",
    activation_qat_qkwargs: dict = None,
    weight_qat_qkwargs: dict = None,
    activation_calibration_qkwargs: dict = None,
    weight_calibration_qkwargs: dict = None,
):
    """Get qconfig.

    Args:
        activation_fake_quant: FakeQuantize type of activation, default is
            "fake_quant". Avaliable items are "fake_quant", "lsq", "pact".

        weight_fake_quant: FakeQuantize type of weight, default is fake_quant.
            Avaliable items are "fake_quant", "lsq" and "pact".

        activation_qat_observer: observer type of qat activation, default is
            "min_max". Avaliable items are "min_max", "percentile", "clip_std".

        weight_qat_observer: observer type of qat weight, default is "min_max".
            Avaliable items are "min_max", "percentile", "clip_std".

        activation_calibration_observer: observer type of calibration
            activation, default is "min_max". Avaliable items are "min_max",
            "percentile", "clip_std".

        weight_calibration_observer: observer type of calibration weight,
            default is "min_max". Avaliable items are "min_max", "percentile",
            "clip_std".

        activation_qat_qkwargs: A dict contain args of qat activation
            FakeQuantize and Observer.

        weight_qat_qkwargs: A dict contain args of qat weight FakeQuantize and
            Observer.

        activation_calibration_qkwargs: A dict contain args of calibration
            activation FakeQuantize and Observer.

        weight_calibration_qkwargs: A dict contain args of calibration weight
            FakeQuantize and Observer.

    """

    if get_qconfig_mode() == QconfigMode.COMPATIBLE:
        raise ValueError(
            "`get_qconfig` is designed for new qconfig, "
            "make sure `QconfigMode` is not `COMPATIBLE`"
        )
    elif get_qconfig_mode() == QconfigMode.CALIBRATION:
        return horizon.quantization.get_default_qconfig(
            activation_fake_quant=activation_fake_quant,
            weight_fake_quant=weight_fake_quant,
            activation_observer=activation_calibration_observer,
            weight_observer=weight_calibration_observer,
            activation_qkwargs=activation_calibration_qkwargs,
            weight_qkwargs=weight_calibration_qkwargs,
        )
    elif get_qconfig_mode() == QconfigMode.QAT:
        return horizon.quantization.get_default_qconfig(
            activation_fake_quant=activation_fake_quant,
            weight_fake_quant=weight_fake_quant,
            activation_observer=activation_qat_observer,
            weight_observer=weight_qat_observer,
            activation_qkwargs=activation_qat_qkwargs,
            weight_qkwargs=weight_qat_qkwargs,
        )
