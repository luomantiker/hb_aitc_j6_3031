r"""Serialization.

This module contains functionality for serializing TorchScript modules in
horizon_plugin_pytorch, notably:
    * horizon_plugin_pytorch.jit.save
    * horizon_plugin_pytorch.jit.load
This two functions are same as `torch.jit.save/load`, in addition to saving
with horizon_plugin_pytorch version and loading with version check.

This is not intended to be imported directly; please use the exposed
functionalities in `horizon_plugin_pytorch.jit`.
"""
import logging
from distutils.version import LooseVersion
from zipfile import ZipFile

import torch

import horizon_plugin_pytorch  # noqa F401

logger = logging.getLogger(__name__)


def get_version_from_scriptmodule(model):
    """Get version from a script module.

    Args:
        model(str): model file name
    """
    extra_file = {"horizon-plugin-version": ""}
    torch.jit.load(model, _extra_files=extra_file)
    version = extra_file["horizon-plugin-version"]
    if version.decode("utf-8") == "":
        logger.warning(
            "The model has not plugin version information, "
            "please use horizon_plugin_pytorch.utils.save_with_version "
            "to save model first",
            extra={"call_times_context": ("message")},
        )
        return None
    else:
        return version.decode("utf-8")


def _get_version():
    try:
        from horizon_plugin_pytorch import __version__

        return __version__
    except ImportError:
        return None


def save_with_version(model, f, _extra_files=None):
    """Save torch version.

    Save an offline version of this module and add horizon-plugin-pytorch
    version into the saved module for use in a separate process. The saved
    module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    torch::jit::load(filename) or into the Python API with torch.jit.load.
    And using horizon_plugin_pytorch.utils.get_version.
    get_version_from_scriptmodule to get corresponding horizon-plugin-pytorch
    version of the saved module.

    To be able to save a module, it must not make any calls to native Python
    functions. This means that all submodules must be subclasses of
    ScriptModule as well.

    Args:
        model: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush)
            or a string containing a file name.
        _extra_files: Map from filename to contents which will be stored
            as part of `f`
    """
    version = _get_version()
    extra_file = {"horizon-plugin-version": version}
    if _extra_files is not None:
        assert isinstance(
            _extra_files, dict
        ), "_extra_files should be dict, but get {}".format(type(_extra_files))
        extra_file.update(_extra_files)
    torch.jit.save(model, f, _extra_files=extra_file)


# A dict records pt compatibility info. Defaultly, pt file is both forward and
# backward compatible. And this dict only records incompatible changes to
# existing ops, excluding new ops.
pt_compatibility_map = {
    # pt generated by the plugin version < version_key can not be loaded in
    # plugin version >= version_key
    # versions in order from new to old
    "backward-incompatible": {},
    # pt generated by the plugin version >= version_key can not be loaded in
    # plugin version < version_key
    # versions in order from new to old
    "forward-incompatible": {
        "1.2.3": "meta requantize implementation introduced in quantized requantize",  # noqa E501
        "1.1.2": "meta conv implementation introduced in quantized conv2d",
    },
}


def save(m, f, _extra_files=None):
    """Save ScriptModule.

    In addition to plugin version saved, this function is same as
    torch.jit.save.

    Args:
        m: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush) or a string
            containing a file name.
        _extra_files: Map from filename to contents which will be stored as
            part of `f`.
    """
    save_with_version(m, f, _extra_files)


def load(f, map_location=None, _extra_files=None):
    """Load ScriptModule previously saved with `horizon.jit.save`.

    In addition to loaded plugin version comparsion with current plugin
    version, this function is same as torch.jit.save.

    Args:
        f: a file-like object(has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.

    Returns:
        A :class:`ScriptModule` object.
    """
    current_version = _get_version()
    extra_file = {"horizon-plugin-version": None}
    if _extra_files is not None:
        assert isinstance(
            _extra_files, dict
        ), "_extra_files should be dict, but get {}".format(type(_extra_files))
        _extra_files.update(extra_file)
    else:
        _extra_files = extra_file

    pt_version = None
    raise_error = None
    try:
        model = torch.jit.load(
            f, map_location=map_location, _extra_files=_extra_files
        )
        pt_version = _extra_files["horizon-plugin-version"].decode("utf-8")
        # if no version in pt, loaded pt_version = ""
        pt_version = None if pt_version == "" else pt_version
    except RuntimeError as e:
        # if load error, unzip pt version from origin pt file
        raise_error = e
        with ZipFile(f, "r") as zipf:
            files = zipf.namelist()
            for file in files:
                if file.endswith("extra/horizon-plugin-version"):
                    pt_version = zipf.read(file).decode("utf-8")
                    break
    # catch not RuntimeError, like filename not exist etc.
    except Exception as e:
        raise (e)

    # if no pt_version, skip version check and return directly
    if pt_version is None:
        if raise_error is None:
            return model
        else:
            raise raise_error

    # check version
    loose_pt_version = LooseVersion(pt_version)
    loose_current_version = LooseVersion(current_version)
    if loose_pt_version < loose_current_version:
        # new plugin load old pt, check backward compatibility
        for k, msg in pt_compatibility_map["backward-incompatible"].items():
            loose_k = LooseVersion(k)
            if loose_pt_version < loose_k and loose_current_version >= loose_k:
                if raise_error is None:
                    logger.warning(
                        f"pt generated in plugin {pt_version} but loaded in "
                        f"plugin {current_version}!",
                        extra={"call_times_context": ("message")},
                    )
                    break
                else:
                    logger.error("origin error info: ", raise_error)
                    raise RuntimeError(
                        f"pt generated in plugin {pt_version} can not be "
                        f"loaded in plugin {current_version} becase of {msg}. "
                        f"Please regenerate pt using plugin >= {k} or load pt "
                        f"using plugin < {k}."
                    )
    elif loose_pt_version > loose_current_version:
        # old plugin load new pt, check forward compatibility
        for k, msg in pt_compatibility_map["forward-incompatible"].items():
            loose_k = LooseVersion(k)
            if loose_pt_version >= loose_k and loose_current_version < loose_k:
                if raise_error is None:
                    logger.warning(
                        f"pt generated in plugin {pt_version} but loaded in "
                        f"plugin {current_version}!",
                        extra={"call_times_context": ("message")},
                    )
                    break
                else:
                    logger.error("origin error info: ", raise_error)
                    raise RuntimeError(
                        f"pt generated in plugin {pt_version} can not be "
                        f"loaded in plugin {current_version} becase of {msg}. "
                        f"Please regenerate pt using plugin < {k} or load pt "
                        f"using plugin >= {k}."
                    )

    # if version compatible but still fail, raise origin load error
    if raise_error is not None:
        raise raise_error

    return model
