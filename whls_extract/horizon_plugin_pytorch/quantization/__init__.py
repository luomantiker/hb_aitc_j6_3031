from horizon_plugin_pytorch.dtype import (  # noqa: F401
    qinfo,
    qint8,
    qint16,
    qint32,
)
from horizon_plugin_pytorch.march import March  # noqa: F401
from horizon_plugin_pytorch.qtensor import QTensor  # noqa: F401
from ._learnable_fake_quantize import _LearnableFakeQuantize
from .auto_calibration import auto_calibrate
from .fake_cast import FakeCast, default_fp16_fake_cast
from .fake_quantize import (
    FakeQuantize,
    FakeQuantState,
    default_4bit_fake_quant,
    default_8bit_fake_quant,
    default_16bit_fake_quant,
    default_calib_fake_quant,
    default_uint4_fake_quant,
    default_weight_4bit_fake_quant,
    default_weight_8bit_fake_quant,
    per_channel_8bit_fake_quant,
    set_fake_quantize,
    update_scale_by_qtype,
)
from .fuse_modules import (
    fuse_conv_shared_modules,
    fuse_known_modules,
    fuse_modules,
)
from .hbdk3 import (
    check_model,
    compile_model,
    export_hbir,
    perf_model,
    visualize_model,
)
from .misc import QATMode, get_qat_mode, set_qat_mode
from .observer import (
    ClipObserver,
    FixedScaleObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from .observer_v2 import load_observer_params
from .pact_fake_quantize import PACTFakeQuantize
from .prepare import PrepareMethod, prepare
from .qconfig import (
    QConfig,
    get_default_calib_qconfig,
    get_default_qat_out_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig,
    get_qconfig,
    per_channel_qat_8bit_qconfig,
)
from .quantize import convert, prepare_qat
from .quantize_fx import convert_fx, fuse_fx, prepare_qat_fx
from .stubs import QuantStub
from .weight_reconstruction import weight_reconstruction

__all__ = [
    # qconfig related
    "QConfig",
    "get_default_qconfig",
    "get_qconfig",
    "FakeCast",
    "default_fp16_fake_cast",
    "FakeQuantize",
    "_LearnableFakeQuantize",
    "PACTFakeQuantize",
    # fuse modules
    "fuse_known_modules",
    "fuse_modules",
    "fuse_conv_shared_modules",
    "fuse_fx",
    # prepare and convert
    "PrepareMethod",
    "prepare",
    "prepare_qat",
    "prepare_qat_fx",
    "convert",
    "convert_fx",
    # hbdk utils
    "export_hbir",
    "check_model",
    "compile_model",
    "perf_model",
    "visualize_model",
    "QuantStub",
    "set_qat_mode",
    "get_qat_mode",
    "QATMode",
    "set_fake_quantize",
    "FakeQuantState",
    "update_scale_by_qtype",
    "load_observer_params",
    "weight_reconstruction",
    "auto_calibrate",
    # deprecated
    "get_default_qat_qconfig",
    "get_default_qat_out_qconfig",
    "get_default_calib_qconfig",
    "default_uint4_fake_quant",
    "default_8bit_fake_quant",
    "per_channel_8bit_fake_quant",
    "default_weight_8bit_fake_quant",
    "default_4bit_fake_quant",
    "default_weight_4bit_fake_quant",
    "default_16bit_fake_quant",
    "default_calib_fake_quant",
    "per_channel_qat_8bit_qconfig",
    # todo: replace with observer v2
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "ClipObserver",
    "FixedScaleObserver",
]
