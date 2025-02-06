# Copyright (c) Horizon Robotics. All rights reserved.

import inspect
import logging
import os
import pickle
import pkgutil
from collections import defaultdict
from importlib import import_module
from typing import Any, Dict, Union

import torch

from hat.utils.apply_func import _as_list
from hat.utils.elastic import elastic_need_resume
from hat.utils.module_patch import TorchModulePatch

logger = logging.getLogger(__name__)

__all__ = [
    "Registry",
    "OBJECT_REGISTRY",
    "build_from_cfg",
    "build_from_registry",
    "RegistryContext",
]


class Registry(object):
    """The registry that provides name -> object mapping.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        class MyBackbone():
            ...

    To register an object with alias :

    .. code-block:: python

        @BACKBONE_REGISTRY.register
        @BACKBONE_REGISTRY.alias('custom')
        class MyBackbone():
            ...

    Args:
        name: The name of this registry
    """

    def __init__(
        self,
        name: str,
    ):
        self._name = name
        self._name_obj_map = {}
        try:
            self.patcher = TorchModulePatch(self._name)
        except ImportError:
            self.patcher = None

    def __contains__(self, name):
        return name in self._name_obj_map

    def keys(self):
        return self._name_obj_map.keys()

    def _do_register(self, name, obj):
        if name in self._name_obj_map:
            # import pdb; pdb.set_trace()
            obj_origin = self._name_obj_map.get(name)
            if obj_origin.__module__ != obj.__module__:
                raise KeyError(
                    f"{name}:{obj_origin} was already registered in "
                    f"{self._name} registry, but get a new object {obj}!"
                )
        else:
            if self.patcher:
                self.patcher.patch_module(obj)
            self._name_obj_map[name] = obj
        return obj

    def register(self, obj=None, *, name=None):
        """Register the given object under `obj.__name__` or given name."""
        if obj is None and name is None:
            raise ValueError("Should provide at least one of obj and name")
        elif obj is not None and name is not None:
            self._do_register(name, obj)
        elif obj is not None and name is None:  # used as decorator
            name = obj.__name__
            self._do_register(name, obj)
            return obj
        else:
            return self.alias(name)

    def register_module(self, *args, **kwargs):  # type: ignore
        return self.register(*args, **kwargs)

    def alias(self, name):
        """Get registrator function that allow aliases.

        Parameters
        ----------
        name: str
            The register name

        Returns
        -------
        a registrator function
        """

        def reg(obj):
            self._do_register(name, obj)
            return obj

        return reg

    def get(self, name):
        ret = self._name_obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object found in {name} registry!")
        return ret


OBJECT_REGISTRY = Registry("HAT_OBJECT_REGISTRY")


class RegistryContext:
    """Store the mapping between object id and object instance."""

    _current: Union[None, Dict] = None

    def __init__(self) -> None:
        try:
            self.patcher = TorchModulePatch("HAT_OBJECT_REGISTRY")
            self.patcher.clear_named_ops()
        except ImportError:
            self.patcher = None
        self._old = None

    def __enter__(self):  # type: ignore
        assert RegistryContext._current is None
        self._old = RegistryContext._current
        if self.patcher:
            self.patcher.__enter__()
        RegistryContext._current = dict()  # noqa
        return self

    def __exit__(self, ptype, value, trace):  # type: ignore
        if self.patcher:
            self.patcher.__exit__(ptype, value, trace)
        RegistryContext._current = self._old

    @classmethod
    def get_current(cls) -> Union[None, Dict]:
        return cls._current


_default_registered = False
_default_registered_error = {}


def _raise_invalid_type_error(obj_type, registry=None):  # type: ignore
    if registry is None:
        registry = [
            OBJECT_REGISTRY,
        ]
    registry_names = [registry_i._name for registry_i in _as_list(registry)]

    err_msg = "{} has not registered in any of registry {} and is not a class, which is not allowed. \n".format(  # noqa
        obj_type, registry_names
    )

    global _default_registered_error
    runtime_error = False
    err_key = []

    for py_file_name in _default_registered_error.keys():
        class_obj = "class " + obj_type
        func_obj = "def " + obj_type
        # path of hat
        dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        file_name = os.path.join(dir_path, py_file_name)

        with open(file_name + ".py", "r", encoding="utf-8") as py_file:
            file_read = py_file.read()
            if file_read.find(class_obj) > 0 or file_read.find(func_obj) > 0:
                err_key.append(py_file_name)
                runtime_error = True

    if runtime_error:
        err_msg += "The reason for not registered may be "
        err_msg += "one of the following situations: \n"
        for key in err_key:
            err_msg += key + ": "
            err_msg += str(_default_registered_error[key]) + " \n"
        raise RuntimeError(err_msg)
    else:
        raise TypeError(err_msg)


def register_default_config(path=None):
    for _, module_name, ispkg in pkgutil.walk_packages(
        [os.path.dirname(__file__)], prefix="hat."
    ):
        try:
            import_module(module_name)
        except Exception as e:
            if not ispkg:
                global _default_registered_error
                py_file_name = os.path.join(*module_name.split("."))
                _default_registered_error[py_file_name] = e


def build_from_cfg(registry: Registry, cfg: dict) -> Any:
    global _default_registered
    if not _default_registered:
        register_default_config()
        _default_registered = True

    if not isinstance(registry, Registry):
        raise TypeError("Expected Registry, but get {}".format(type(registry)))
    if not isinstance(cfg, dict):
        raise TypeError("Expected dict, but get {}".format(type(cfg)))
    if "type" not in cfg:
        raise KeyError("Required has key `type`, but not")

    cfg = cfg.copy()
    obj_type = cfg.pop("type")
    if obj_type in registry:
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        _raise_invalid_type_error(obj_type, registry)

    try:
        instance = obj_cls(**cfg)
    except TypeError as te:
        raise TypeError("%s: %s" % (obj_cls, te))

    return instance


def _build_dataset(cfg: dict) -> Any:
    from hat.core.compose_transform import Compose

    if "transforms" in cfg and cfg["transforms"] is not None:
        if isinstance(cfg["transforms"], (list, tuple)):
            cfg["transforms"] = Compose(cfg["transforms"])  # noqa
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
    obj = pickle.loads(pickle.dumps(obj))
    return obj


def _build_optimizer(cfg: dict) -> Any:
    def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
        if "params" in cfg:
            loc_name = {}  # type: ignore
            for k, v in cfg["params"].items():
                loc_name[k] = {"params": []}
                loc_name[k].update(v)

            loc_name["others"] = {
                "params": [],
                "weight_decay": (
                    cfg["weight_decay"] if "weight_decay" in cfg else 0
                ),
            }
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    pass
                flag = False
                for k, _v in cfg["params"].items():
                    if k in name:
                        loc_name[k]["params"].append(p)
                        flag = True
                        break
                if not flag:
                    loc_name["others"]["params"].append(p)

            res = []
            for _k, v in loc_name.items():
                res.append(v)
            cfg["params"] = res
        else:
            cfg["params"] = filter(
                lambda p: p.requires_grad, model.parameters()
            )
        return build_from_cfg(OBJECT_REGISTRY, cfg)

    if "model" in cfg:
        model = cfg.pop("model")
        assert isinstance(model, torch.nn.Module)
        return build_optimizer(model)
    else:
        return build_optimizer


def _modify_pytorch_dataloader_config(cfg: dict) -> dict:
    if "sampler" in cfg:
        if (
            isinstance(cfg["sampler"], dict)
            and "dataset" not in cfg["sampler"]
        ):  # noqa
            cfg["sampler"]["dataset"] = cfg["dataset"]

        if cfg["sampler"] is not None:
            cfg["shuffle"] = False

        if elastic_need_resume():
            if cfg["sampler"]["type"] is torch.utils.data.DistributedSampler:
                cfg["sampler"]["type"] = "StatefulDistributedSampler"
                if "batch_size" in cfg:
                    cfg["sampler"]["batch_size"] = cfg["batch_size"]

    return cfg


def _is_dataloader(object_type: Any) -> bool:
    custom_loader_types = ["RankSplitDataLoader"]
    if (
        object_type is torch.utils.data.DataLoader
        or object_type in custom_loader_types
    ):
        return True
    else:
        return False


def build_from_registry(x: Any) -> Any:
    """
    Build object from registry.

    This function will recursively visit all elements, if an object is dict
    and has the key `type`, which is considered as an object that should be
    build.
    """

    def _impl(x):  # type: ignore
        id2object = RegistryContext.get_current()
        if isinstance(x, (list, tuple)):
            x = type(x)((_impl(x_i) for x_i in x))
            return x
        elif isinstance(x, dict):
            if "__lazy_build__" in x and x["__lazy_build__"]:
                x.pop("__lazy_build__")
                return x
            object_id = id(x)
            has_type = "type" in x
            object_type = x.get("type", None)
            if has_type and object_id in id2object:
                return id2object[object_id]
            if has_type and _is_dataloader(object_type):
                x = _modify_pytorch_dataloader_config(x)

            if x.pop("__build_recursive", True):
                # TODO(hongyu.xie): add unittest
                # fmt: off
                build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
                # fmt: on
            else:
                build_x = x

            if type(x) is defaultdict:
                x = defaultdict(x.default_factory, build_x)
            else:
                x = type(x)(build_x)

            if has_type:
                if object_type in OBJECT_REGISTRY:
                    object_type = OBJECT_REGISTRY.get(object_type)
                    isclass = inspect.isclass(object_type)
                elif inspect.isclass(object_type):
                    isclass = True
                else:
                    _raise_invalid_type_error(object_type)
                if isclass and issubclass(
                    object_type, torch.utils.data.Dataset
                ):  # noqa
                    obj = _build_dataset(x)
                elif isclass and issubclass(
                    object_type, torch.optim.Optimizer
                ):  # noqa
                    obj = _build_optimizer(x)
                else:
                    obj = build_from_cfg(OBJECT_REGISTRY, x)
                id2object[object_id] = obj
                return obj
            else:
                return x
        else:
            return x

    global _default_registered
    if not _default_registered:
        register_default_config()
        _default_registered = True
    current = RegistryContext.get_current()
    if current is None:
        with RegistryContext():
            return _impl(x)
    else:
        return _impl(x)
