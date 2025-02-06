# Quant pad is supported by QTensor (F.pad), do patch on torch modules to
# handle the loading of old state_dict.

from torch import nn


def _load_state_dict_ignore_act(
    obj: nn.Module,
    state_dict: dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    ignored_submod = "activation_post_process"
    ignored_prefix = prefix + ignored_submod
    ignored_buffers = []

    for k in state_dict:
        if k.startswith(ignored_prefix):
            ignored_buffers.append(k)

    for k in ignored_buffers:
        state_dict.pop(k)

    return nn.Module._load_from_state_dict(
        obj,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )


def _patch_torch_modules():
    if nn.ConstantPad1d._load_from_state_dict is _load_state_dict_ignore_act:
        return
    nn.ConstantPad1d._load_from_state_dict = _load_state_dict_ignore_act
    nn.ConstantPad2d._load_from_state_dict = _load_state_dict_ignore_act
    nn.ConstantPad3d._load_from_state_dict = _load_state_dict_ignore_act
    nn.ZeroPad2d._load_from_state_dict = _load_state_dict_ignore_act


_patch_torch_modules()
