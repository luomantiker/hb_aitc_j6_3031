import horizon_plugin_pytorch as horizon
from .version import __version__  # noqa: F401

horizon.utils.logger.set_logger(
    "horizon_plugin_profiler", file_dir=".horizon_plugin_profiler_logs"
)


def _raise_profiler_import_error(func_name):
    def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"{func_name} in horizon_plugin_profiler {__version__} cannot run "
            f"with horizon_plugin_pytorch {horizon.__version__}. Please "
            "update horizon_plugin_pytorch."
        )

    return wrapper


try:
    from .check_deploy_device import check_deploy_device
except ImportError:
    check_deploy_device = _raise_profiler_import_error("check_deploy_device")

try:
    from .check_qconfig import check_qconfig
except ImportError:
    check_qconfig = _raise_profiler_import_error("check_qconfig")

try:
    from .check_unfused_operations import check_unfused_operations
except ImportError:
    check_unfused_operations = _raise_profiler_import_error(
        "check_unfused_operations"
    )

try:
    from .compare_weights import compare_weights
except ImportError:
    compare_weights = _raise_profiler_import_error("compare_weights")

try:
    from .featuremap_similarity import featuremap_similarity
except ImportError:
    featuremap_similarity = _raise_profiler_import_error(
        "featuremap_similarity"
    )

try:
    from .get_module_called_count import get_module_called_count
except ImportError:
    get_module_called_count = _raise_profiler_import_error(
        "get_module_called_count"
    )

try:
    from .get_raw_features import get_raw_features
except ImportError:
    get_raw_features = _raise_profiler_import_error("get_raw_features")

try:
    from .model_profiler import model_profiler
except ImportError:
    model_profiler = _raise_profiler_import_error("model_profiler")

try:
    from .profile_featuremap import profile_featuremap
except ImportError:
    profile_featuremap = _raise_profiler_import_error("profile_featuremap")

try:
    from .script_profile import compare_script_models, script_profile
except ImportError:
    compare_script_models = _raise_profiler_import_error(
        "compare_script_models"
    )
    script_profile = _raise_profiler_import_error("script_profile")

try:
    from .set_preserve_qat_mode import set_preserve_qat_mode
except ImportError:
    set_preserve_qat_mode = _raise_profiler_import_error(
        "set_preserve_qat_mode"
    )

try:
    from .show_cuda_memory_consumption import show_cuda_memory_consumption
except ImportError:
    show_cuda_memory_consumption = _raise_profiler_import_error(
        "show_cuda_memory_consumption"
    )

try:
    from .model_profilerv2 import ModelProfiler
except ImportError:
    ModelProfiler = _raise_profiler_import_error("ModelProfiler")

try:
    from .model_profilerv2 import QuantAnalysis
except ImportError:
    QuantAnalysis = _raise_profiler_import_error("QuantAnalysis")

try:
    from .hbir_model_profiler import HbirModelProfiler
except ImportError:
    HbirModelProfiler = _raise_profiler_import_error("HbirModelProfiler")

__all__ = [
    "check_deploy_device",
    "check_qconfig",
    "check_unfused_operations",
    "compare_script_models",
    "compare_weights",
    "featuremap_similarity",
    "get_module_called_count",
    "get_raw_features",
    "model_profiler",
    "profile_featuremap",
    "script_profile",
    "set_preserve_qat_mode",
    "show_cuda_memory_consumption",
    "ModelProfiler",
    "QuantAnalysis",
    "HbirModelProfiler",
]
