"""load ops from extension c library."""
import warnings

import torch


def _register_extensions():
    import importlib
    import os

    import torch

    from .version import __version__

    if "cu" in __version__ and not torch.cuda.is_available():
        warnings.warn(
            "CUDA or GPU is not availiable, please check the "
            "environment configuration.",
            ResourceWarning,
        )

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("libhorizon_ops")
    if ext_specs is None:
        raise ImportError(
            "Can not find libhorizon_ops.so in {}, please check the "
            "installation of horizon_plugin_pytorch".format(lib_dir)
        )
    try:
        torch.ops.load_library(ext_specs.origin)
    except Exception as e:
        raise RuntimeError(
            "Fail to load libhorizon_ops, origin msg is\n{}".format(str(e))
        )


_register_extensions()


def _check_cuda_version():
    """Refine this docstring in the future.

    Make sure that CUDA versions match between the pytorch install and
    horizon_plugin_pytorch install
    """

    _version = torch.ops.horizon._cuda_version()
    if _version != -1 and torch.version.cuda is not None:
        horizon_version = str(_version)
        if int(horizon_version) < 10000:
            tv_major = int(horizon_version[0])
            tv_minor = int(horizon_version[2])
        else:
            tv_major = int(horizon_version[0:2])
            tv_minor = int(horizon_version[3])
        t_version = torch.version.cuda
        t_version = t_version.split(".")
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != tv_major or t_minor != tv_minor:
            raise RuntimeError(
                "Detected that PyTorch and horizon_plugin_pytorch were compiled with different CUDA versions. "  # noqa: E501
                "PyTorch has CUDA Version={}.{} and horizon_plugin_pytorch has CUDA Version={}.{}. "  # noqa: E501
                "Please reinstall the horizon_plugin_pytorch that matches your PyTorch install.".format(  # noqa: E501
                    t_major, t_minor, tv_major, tv_minor
                )
            )
    return _version


if not torch.version.hip:
    _check_cuda_version()
