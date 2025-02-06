import copy
import logging
from collections import OrderedDict
from functools import partial
from typing import Dict, Tuple, Type, Union

import torch.nn as nn
import torch.optim as optim
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _NormBase

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["custom_param_optimizer"]


@OBJECT_REGISTRY.register
def custom_param_optimizer(
    optim_cls: Type[optim.Optimizer],
    optim_cfgs: Dict,
    custom_param_mapper: Dict[
        Union[str, Type[nn.Module], Tuple], Dict[str, float]
    ],
):
    """Return optimizer with custom params setting.

    Args:
        optim_cls: The wrapped optimizer class, e.g., `torch.optim.SGD`.
        model: The model instance that will be trained.
        optim_cfgs: The configuration for the basic optimizer, e.g., `lr`.
        custom_param_mapper: A dictionary for custom mapping between model
            parameters and optimizer parameters.

    The `custom_param_mapper` has the following key characteristics:

    **Key Matching**:

        1. Class of `torch.nn.Module`: The keys can directly match the
           corresponding parameters of the model.

        2. Predefined types: Keys can be chosen from predefined types.
           Supported types include ["norm_types", ].

        3. String match: Keys can be matched based on param_names.

        4. Tuple of previous 3 kinds of keys.

    **Value Setting**:
        Optimizer parameters can be set, e.g.,
        {"weight_decay": 1e-4, "lr": 0.01}.

    Example::

           >>> custom_param_mapper = {
           ...     "norm_types": {"weight_decay": 0},
           ...     nn.Conv2d: {"weight_decay": 1e-4, "lr": 0.01},
           ...     "bias": {"weight_decay": 1e-5, "lr": 0.1},
           ...     (nn.Conv2d, "bias"): {"weight_decay": 0,}
           ... }
    """

    def _wrapped_optimizer(model, optim_cls, optim_cfgs, custom_param_mapper):
        assert issubclass(
            optim_cls, optim.Optimizer
        ), "optim_cls should be a subclass of torch.optim.Optimizer"
        custom_type_dict = {
            "norm_types": (_NormBase, GroupNorm, LayerNorm),
        }

        custom_param_mapper_ = {}
        for k, v in custom_param_mapper.items():
            if isinstance(k, tuple):
                custom_param_mapper_[frozenset(k)] = v
            else:
                custom_param_mapper_[k] = v

        params = _custom_set_param(
            model, custom_param_mapper_, custom_type_dict
        )

        wrapped_optimizer = optim_cls(params, **optim_cfgs)
        return wrapped_optimizer

    func = partial(
        _wrapped_optimizer,
        optim_cls=optim_cls,
        optim_cfgs=optim_cfgs,
        custom_param_mapper=custom_param_mapper,
    )

    return func


def _custom_match(
    param_name: str,
    module: nn.Module,
    match_key: Union[frozenset, type, str],
    custom_type_dict: dict,
):
    if isinstance(match_key, frozenset):
        matched = True
        for key in match_key:
            matched = matched and _custom_match(
                param_name, module, key, custom_type_dict
            )
        return matched
    elif isinstance(match_key, type) and issubclass(match_key, nn.Module):
        if isinstance(module, match_key):
            return True
    elif isinstance(match_key, str):
        if match_key in custom_type_dict:
            if isinstance(module, custom_type_dict[match_key]):
                return True
        elif match_key in param_name:
            return True
    else:
        raise TypeError(
            "The type of the key of custom_param_mapper should be"
            + " either str, type of module, or tuple of both type."
        )
    return False


def _merge_set(A, B):
    A = A if isinstance(A, frozenset) else frozenset([A])
    B = B if isinstance(B, frozenset) else frozenset([B])
    return A.union(B)


def _custom_set_param(
    model: nn.Module,
    custom_param_mapper: dict,
    custom_type_dict: dict,
):
    special_params = {}
    for match_key in custom_param_mapper:
        special_params[match_key] = []
    other_params_list = []
    extra_param_mapper = {}
    param_module_name_dict = OrderedDict()
    for name, module in model.named_modules():
        for sub_name, p in module.named_parameters(recurse=False):
            param_name = ".".join([name, sub_name])
            if not p.requires_grad:
                continue
            if p not in param_module_name_dict:
                param_module_name_dict[p] = [(module, param_name)]
            else:
                param_module_name_dict[p].append((module, param_name))

    for p in param_module_name_dict.keys():
        matched_value = None
        matched_key = None
        for module, param_name in param_module_name_dict[p]:
            for match_key in custom_param_mapper:
                if _custom_match(
                    param_name, module, match_key, custom_type_dict
                ):
                    if matched_key is None:
                        matched_value = custom_param_mapper[match_key]
                        matched_key = match_key
                    else:
                        for k, v in custom_param_mapper[match_key].items():
                            if k in matched_value.keys():
                                assert v == matched_value[k], (
                                    f"{param_name} is contained in multiple custom_param_mapper, "  # noqa
                                    + "but params setting are inconsistent, get "  # noqa
                                    + f"{k}:{v} vs. {k}:{matched_value[k]} in {matched_key}"  # noqa
                                )  # noqa

                        new_key = _merge_set(matched_key, match_key)
                        matched_value = copy.deepcopy(matched_value)
                        matched_value.update(custom_param_mapper[match_key])
                        extra_param_mapper[new_key] = matched_value
                        matched_key = new_key
        if not matched_key:
            other_params_list.append(p)
        elif matched_key in special_params:
            special_params[matched_key].append(p)
        else:
            special_params[matched_key] = [p]

    param_groups = []
    for match_key in special_params:
        if len(special_params[match_key]) == 0:
            logger.warning(f"Key '{match_key}' found no matched parameters.")
            continue
        params = {"params": special_params[match_key]}
        if match_key in extra_param_mapper:
            params.update(extra_param_mapper[match_key])
        else:
            params.update(custom_param_mapper[match_key])
        param_groups.append(params)
    other_params = {"params": other_params_list}
    param_groups.append(other_params)
    _check_param_groups(param_groups, param_module_name_dict)
    return param_groups


def _check_param_groups(param_groups: dict, param_module_name_dict: dict):
    total_params_list = []
    for param_group in param_groups:
        params = param_group["params"]
        for p in params:
            total_params_list.append(p)
    assert len(total_params_list) == len(
        param_module_name_dict
    ), "The number of parameters in param_groups is not equal to the number of total_params_set, maybe some parameters exist in multiple groups"  # noqa
    assert (
        set(total_params_list) == param_module_name_dict.keys()
    ), "The parameters in param_groups is not equal to the parameters of params_set."  # noqa
