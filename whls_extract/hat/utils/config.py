# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import json
import logging
import os
import os.path as osp
import sys
from argparse import Action, ArgumentParser, Namespace
from ast import literal_eval
from collections.abc import Mapping
from enum import Enum
from functools import reduce
from importlib import import_module
from pathlib import PurePath
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
import yaml

from .jsonable import is_jsonable, strify_keys

__all__ = [
    "Config",
    "filter_configs",
    "crop_configs",
]
logger = logging.getLogger(__name__)

_VALID_TYPES = {tuple, list, str, int, float, bool, type(None), dict}


class ConfigVersion(Enum):
    v1 = 1
    v2 = 2


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_jsonable(obj):
            return super(JSONEncoder, self).default(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return str(obj)


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    """

    @staticmethod
    def fromfile(filename):
        if isinstance(filename, PurePath):
            filename = filename.as_posix()
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise KeyError("file {} does not exist".format(filename))
        if filename.endswith(".py"):
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)

            old_module = None
            if module_name in sys.modules:
                old_module = sys.modules.pop(module_name)

            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }
            # IMPORTANT: pop to avoid `import_module` from cache, to avoid the
            # cfg sharing by multiple processes or functions, which may cause
            # interference and get unexpected result.
            sys.modules.pop(module_name)

            if old_module is not None:
                sys.modules[module_name] = old_module

        elif filename.endswith((".yml", ".yaml")):
            with open(filename, "r") as fid:
                cfg_dict = yaml.load(fid, Loader=yaml.Loader)
        else:
            raise IOError(
                "Only py/yml/yaml type are supported now, "
                f"but found {filename}!"
            )
        return Config(cfg_dict, filename=filename)

    def __init__(self, cfg_dict=None, filename=None, encoding="utf-8"):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", cfg_dict)
        super(Config, self).__setattr__("_filename", filename)
        if filename:
            with open(filename, "r", encoding=encoding) as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    def merge_from_list_or_dict(self, cfg_opts, overwrite=False):
        """Merge config (keys, values) in a list or dict into this cfg.

        Examples:
            cfg_opts is a list:
            >>> cfg_opts = [
                                'model.backbone.type', 'ResNet18',
                                'model.backbone.num_classes', 10,
                            ]
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet50'))))
            >>> cfg.merge_from_list_or_dict(cfg_opts)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...    model=dict(backbone=dict(type="ResNet18", num_classes=10)))

            cfg_opts is a dict:
            >>> cfg_opts = {'model.backbone.type': "ResNet18",
            ...            'model.backbone.num_classes':10}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet50'))))
            >>> cfg.merge_from_list_or_dict(cfg_opts)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...    model=dict(backbone=dict(type="ResNet18", num_classes=10)))
        Args:
            cfg_opts (list or dict): list or dict of configs to merge from.
            overwrite (bool): Weather to overwrite existing (keys, values).
        """

        if isinstance(cfg_opts, list):
            assert len(cfg_opts) % 2 == 0, (
                "Override list has odd length: "
                f"{cfg_opts}; it must be a list of pairs"
            )
            opts_dict = {}
            for k, v in zip(cfg_opts[0::2], cfg_opts[1::2]):
                opts_dict[k] = v
        elif isinstance(cfg_opts, dict):
            opts_dict = cfg_opts
        else:
            raise ValueError(
                f"cfg_opts should be list or dict, but is {type(cfg_opts)}"
            )

        for full_key, v in opts_dict.items():
            d = self
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, {})
                d = d[subkey]
            subkey = key_list[-1]

            try:
                v = literal_eval(v)
            except Exception:
                pass

            if type(v) not in _VALID_TYPES:
                raise ValueError(
                    f"The incoming value of key `{full_key}` should be str, "
                    f"list, tuple or dict, but get {v}"
                )
            else:
                value = v

            if subkey in d:
                if overwrite:
                    value = _check_and_coerce_cfg_value_type(
                        value, d[subkey], subkey, full_key
                    )
                    logger.debug(
                        f"'{full_key}: {d[subkey]}' will be overwritten "
                        f"with '{full_key}: {value}'"
                    )
                    d[subkey] = value
                else:
                    logger.warning(
                        f"The incoming `{full_key}` already exists in config, "
                        f"but the obtained `overwrite = false`, which will "
                        f"still use the `{full_key}: {value}` in config."
                    )
            else:
                d[subkey] = value

    def dump_json(self, skip_keys=False):
        if skip_keys:
            logger.warning(
                "Some non compliant keys will be changed to null and saved."
            )
            cfg_dict = self._cfg_dict
        else:
            cfg_dict = strify_keys(self._cfg_dict)
        return json.dumps(
            cfg_dict, cls=JSONEncoder, sort_keys=False, skipkeys=skip_keys
        )

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return "Config (path: {}): {}".format(
            self.filename, self._cfg_dict.__repr__()
        )

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        try:
            return getattr(self._cfg_dict, name)
        except AttributeError as e:
            if isinstance(self._cfg_dict, dict):
                try:
                    return self.__getitem__(name)
                except KeyError:
                    raise AttributeError(name)
            raise e

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __setitem__(self, name, value):
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)


def crop_configs(cfg: Config):
    cfg = filter_configs(cfg, return_dict=False)
    cfg_dict = json.loads(cfg.dump_json())

    def _to_flat_list(obj):
        out_lst = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out_lst.append(k)
                out_lst.append(_to_flat_list(v))
        elif isinstance(obj, list):
            for v in obj:
                out_lst.append(_to_flat_list(v))
        else:
            out_lst.append(obj)

        if len(out_lst) == 1:
            return out_lst[0]
        else:
            return out_lst

    def _flat_cfg(obj, prefix="", sep="."):
        out = {}

        def _flat():
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_predix = prefix + sep + key if prefix else key
                    out.update(_flat_cfg(value, prefix=new_predix))
            elif isinstance(obj, list):
                out[prefix] = _to_flat_list(obj)
            else:
                out[prefix] = str(obj) if obj is None else obj

        _flat()
        return out

    def _filter_dict_by_keys(cfg_dict, filter_keys=None):

        if not filter_keys:
            return cfg_dict

        output_dict = {}
        for key in filter_keys:
            value = cfg_dict.get(key, None)
            if isinstance(value, bool) or not value:
                value = str(value)
            output_dict[key] = value

        return output_dict

    output_dict = {}
    main_keys_list = [
        "device_ids",
        "march",
        "compile_cfg",
        "seed",
        "cudnn_benchmark",
    ]

    trainer_keys_list = [
        "optimizer",
        "stop_by",
        "num_steps",
        "num_epochs",
        "sync_bn",
        "batch_processor",
        "model_convert_pipeline",
    ]

    # {stage}_trainer
    stage = os.environ.get("HAT_TRAINING_STEP", None)
    assert stage

    trainer = f"{stage}_trainer"
    main_keys_list.append(trainer)

    output_dict = _filter_dict_by_keys(cfg_dict, main_keys_list)

    # {stage}_trainer
    output_dict[trainer] = _filter_dict_by_keys(
        output_dict[trainer],
        trainer_keys_list,
    )

    return _flat_cfg(output_dict)


class ParseAction(Action):
    """ParseAction.

    Argparse action to split an argument into KEY VALUE form
    on the blank and append to a dictionary.
    List options can be passed as comma separated values,
    i.e 'KEY V1,V2,V3', or with explicit brackets, i.e. 'KEY [V1,V2,V3]'.
    It also support nested brackets to build
    list/tuple values. e.g. 'KEY [(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """Parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' or '{}' are treated
        as iterable values.

        Args:
            val : Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> ParseAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> ParseAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> ParseAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
            >>> ParseAction._parse_iterable("{'type'='SimpileProfiler', \
                'interval'='10'}")
            {type='SimpileProfiler', interval=10}
            >>> ParseAction._parse_iterable("dict(type='SimpileProfiler', \
                interval=10)")
            {type='SimpileProfiler', interval=10}
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (
                (string.count("(") == string.count(")"))
                and (string.count("[") == string.count("]"))
                and (string.count("{") == string.count("}"))
                and (
                    string.count("dict(") == 0
                    or string.count("dict(") == string.count(")")
                )
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                    (
                        (char == ",")
                        and (pre.count("(") == pre.count(")"))
                        and (pre.count("[") == pre.count("]"))
                    )
                    and (pre.count("{") == pre.count("}"))
                    and (
                        pre.count("dict(") == 0
                        or pre.count("dict(") == pre.count(")")
                    )
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        is_dict = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif val.startswith("{") and val.endswith("}"):
            val = val[1:-1]
            is_dict = True
        elif val.startswith("dict(") and val.endswith(")"):
            val = val[5:-1]
            is_dict = True
        elif ":" in val:
            # val is a dict
            kv_pairs = val.split(":")
            if len(kv_pairs) > 2:
                kv_pairs = [kv_pairs[0], ":".join(kv_pairs[1:])]
            return {
                ParseAction._parse_iterable(
                    kv_pairs[0]
                ): ParseAction._parse_iterable(kv_pairs[1])
            }
        elif "=" in val:
            # val is a dict
            kv_pairs = val.split("=")
            if len(kv_pairs) > 2:
                kv_pairs = [kv_pairs[0], ":".join(kv_pairs[1:])]
            return {
                ParseAction._parse_iterable(
                    kv_pairs[0]
                ): ParseAction._parse_iterable(kv_pairs[1])
            }
        elif "," not in val:
            # val is a single value
            return ParseAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = ParseAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]

        if is_tuple:
            return tuple(values)
        if is_dict:
            res = {}
            for d in values:
                res.update(d)
            return res

        return values

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: str = None,
    ):
        """Parse Variables in string and add them into argparser.

        Args:
            parser : Argument parser.
            namespace : Argument namespace.
            values : Argument string.
            option_string : Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or [])
        if values is not None:
            use_values = []
            # Deal with the situation where the value is split miss by dict
            if len(values) % 2 != 0:
                tmp = []
                for v in values:
                    if (
                        "[" in v
                        or "(" in v
                        or "{" in v
                        or "]" in v
                        or ")" in v
                        or "}" in v
                        or "," in v
                        or ":" in v
                    ):
                        tmp[-1] += v
                    else:
                        tmp.append(v + "=")
                for v in tmp:
                    use_values.extend(v.strip("=").split("="))
            else:
                use_values = values

            assert len(use_values) % 2 == 0, (
                "Override list has odd length: "
                f"{use_values}; it must be a list of pairs"
            )

            for full_key, v in zip(use_values[0::2], use_values[1::2]):
                options.append(full_key)
                options.append(ParseAction._parse_iterable(v))

        setattr(namespace, self.dest, options)


def filter_configs(
    configs: Config,
    threshold: int = 100,
    return_dict: bool = True,
) -> Dict or Config:
    """
    Filter config to for pprint.format.

    Inplace Tensor with Tensor.shape and inplace numpy with numpy.shape.

    Args:
        configs (Config): configs for pprint.
        threshold (int): threshold of filter to convert shape.
        return_dict (bool): Whether to return dict or config.
    Returns:
        configs (Dict or Config): configs to pprint.
    """

    def filter_elem(cfg):
        if isinstance(cfg, (torch.Tensor, np.ndarray)):
            n_elem = reduce(lambda x, y: x * y, cfg.shape)
            if n_elem >= threshold:
                return cfg.shape
            else:
                return cfg
        elif isinstance(cfg, Mapping):
            return {key: filter_elem(cfg[key]) for key in cfg.keys()}
        elif isinstance(cfg, (list, tuple)):
            return [filter_elem(c) for c in cfg]
        else:
            return cfg

    new_configs = {}
    for k, v in configs._cfg_dict.items():
        new_configs[k] = filter_elem(v)
    if return_dict:
        return new_configs
    else:
        return Config(cfg_dict=new_configs)


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Check that `replacement`, which is intended to replace `original` is \
    of the right type. The type is correct if it matches exactly or is one of \
    a few cases in which the type can be easily coerced.

    Copied from `yacs <https://github.com/rbgirshick/yacs>`.

    """

    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # If either of them is None, allow type convert to one of the valid types
    if (replacement is None and original_type in _VALID_TYPES) or (
        original is None and replacement_type in _VALID_TYPES
    ):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and
    # original types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )
