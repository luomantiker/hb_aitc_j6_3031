def _raise_profiler_import_error(func_name):
    def wrapper(*args, **kwargs):
        raise RuntimeError(
            "Profilers tools are released in a standalone package"
            "`horizon_plugin_profiler` after horizon_plugin_pytorch v1.8.1."
            f"Please install horizon_plugin_profiler to use {func_name}."
        )

    return wrapper


try:
    from horizon_plugin_profiler import (
        check_deploy_device,
        check_qconfig,
        check_unfused_operations,
        compare_script_models,
        compare_weights,
        featuremap_similarity,
        get_module_called_count,
        get_raw_features,
        model_profiler,
        profile_featuremap,
        script_profile,
        set_preserve_qat_mode,
        show_cuda_memory_consumption,
    )
except ImportError:
    check_deploy_device = _raise_profiler_import_error("check_deploy_device")
    check_qconfig = _raise_profiler_import_error("check_qconfig")
    check_unfused_operations = _raise_profiler_import_error(
        "check_unfused_operations"
    )
    compare_weights = _raise_profiler_import_error("compare_weights")
    featuremap_similarity = _raise_profiler_import_error(
        "featuremap_similarity"
    )
    get_module_called_count = _raise_profiler_import_error(
        "get_module_called_count"
    )
    get_raw_features = _raise_profiler_import_error("get_raw_features")
    model_profiler = _raise_profiler_import_error("model_profiler")
    profile_featuremap = _raise_profiler_import_error("profile_featuremap")
    compare_script_models = _raise_profiler_import_error(
        "compare_script_models"
    )
    script_profile = _raise_profiler_import_error("script_profile")
    set_preserve_qat_mode = _raise_profiler_import_error(
        "set_preserve_qat_mode"
    )
    show_cuda_memory_consumption = _raise_profiler_import_error(
        "show_cuda_memory_consumption"
    )
