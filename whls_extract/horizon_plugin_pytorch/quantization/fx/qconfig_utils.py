import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Union

import torch

from .utils import _parent_name

QConfigAny = Union[torch.quantization.QConfig, None]


def get_flattened_qconfig_dict(qconfig_dict):
    """Refine this docstring in the future.

    Flatten the global, module_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "module_type": [
        (torch.nn.Conv2d, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened = {}
    if "" in qconfig_dict:
        flattened[""] = qconfig_dict[""]

    def flatten_key(key):
        if key in qconfig_dict:
            for (obj, qconfig) in qconfig_dict[key].items():
                flattened[obj] = qconfig

    flatten_key("module_type")
    flatten_key("module_name")
    return flattened


def convert_dict_to_ordered_dict(
    qconfig_dict: Dict[str, Any],
) -> Dict[str, Dict[Any, Any]]:
    """Convert dict in qconfig_dict to ordered dict."""
    # convert a qconfig list for a type to OrderedDict
    def _convert_to_ordered_dict(key, qconfig_dict):
        qconfig_dict[key] = OrderedDict(qconfig_dict.get(key, []))

    _convert_to_ordered_dict("module_type", qconfig_dict)
    _convert_to_ordered_dict("module_name", qconfig_dict)
    return qconfig_dict


def get_module_type_qconfig(
    qconfig_dict: Dict[str, Any],
    module_type: Union[Callable, str],
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    return qconfig_dict["module_type"].get(module_type, fallback_qconfig)


def get_module_name_regex_qconfig(qconfig_dict, module_name, fallback_qconfig):
    for regex_pattern, qconfig in qconfig_dict["module_name_regex"].items():
        if re.match(regex_pattern, module_name):
            # first match wins
            return qconfig
    return fallback_qconfig


def get_module_name_qconfig(qconfig_dict, module_name, fallback_qconfig):
    if module_name == "":
        # module name qconfig not found
        return fallback_qconfig
    if module_name in qconfig_dict["module_name"]:
        return qconfig_dict["module_name"][module_name]
    else:
        parent, _ = _parent_name(module_name)
        return get_module_name_qconfig(qconfig_dict, parent, fallback_qconfig)


# get qconfig for module_name,
# fallback to module_name_regex_qconfig, module_type_qconfig,
# global_qconfig if necessary
def get_qconfig(qconfig_dict, module_type, module_name, global_qconfig):
    module_type_qconfig = get_module_type_qconfig(
        qconfig_dict, module_type, global_qconfig
    )
    module_name_regex_qconfig = get_module_name_regex_qconfig(
        qconfig_dict, module_name, module_type_qconfig
    )
    module_name_qconfig = get_module_name_qconfig(
        qconfig_dict, module_name, module_name_regex_qconfig
    )
    return module_name_qconfig
