import logging
from typing import Tuple, Type

from horizon_plugin_profiler.utils.typeguard import typechecked

import torch

from horizon_plugin_pytorch import nn as horizon_nn
from horizon_plugin_pytorch.quantization.quantization_mappings import (
    get_qat_module_mappings,
)

logger = logging.getLogger(__name__)


@typechecked
def set_preserve_qat_mode(
    model: torch.nn.Module,
    prefixes: Tuple[str, ...] = (),
    types: Tuple[Type[torch.nn.Module], ...] = (),
    value: bool = True,
):
    """Set preserve qat mode.

    Make modules in the model to preserve qat mode in convert by setting
    mod.preserve_qat_mode attribute. It can be used on float model or qat
    model.
    Note:
        1) For fused module, only conv.preserve_qat_mode = True,
        fused.preserve_qat_mode = True. So setting the fused.preserve_qat_mode
        = True is same as setting conv.preserve_qat_mode = True. For example,

        .. code-block:: python

            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv = torch.nn.Conv2d()
                    self.bn = torch.nn.BatchNorm2d()
                    self.add = FloatFunctional()
                    self.relu = torch.nn.Relu()

            float_model = Model()

            # set float conv is OK
            set_preserve_qat_mode(float_model, types=(torch.nn.Conv2d,))

            # set float bn does not work
            set_preserve_qat_mode(float_model, types=(torch.nn.BatchNorm2d,))

            float_model.fuse_modules()
            float_model.qconfig = get_default_qat_qconfig()
            qat_model = prepare_qat(float_model)

            # After fuse and convert, set conv via float type is also OK.
            # All conv modules and fused modules(convbn, convbnadd, ...)
            # will set preserve_qat_mode = True
            set_preserve_qat_mode(qat_model, types=(torch.nn.Conv2d,))

            # To set exactly one fused module, use 'prefixes' arg.
            # convbnaddrelu is fused on "add" position
            set_preserve_qat_mode(qat_model, prefixes=("add",))

        2) If float model uses torch functions(torch.add, torch.pow, ...) and
        is converted by fx, this functions will be converted to horizon ops
        automatically. To set these functions preserve_qat_mode = True, please
        set corresponding horizon ops preserve_qat_mode = True in qat model.
        For example,

        .. code-block:: python

            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.add = torch.add

            float_model = Model()
            # convert by fx
            qat_model = prepare_qat_fx(float_model)

            # set by types is OK. All FloatFunctional in qat model will
            # be set preserve_qat_mode = True
            set_preserve_qat_mode(qat_model, types=(FloatFunctional,))

            # To set exactly this add, use 'prefixes' arg
            # "add_generated_add_0" is the generated add module name
            set_preserve_qat_mode(qat_model, prefixes=("add_generated_add_0",))

    Args:
        model (nn.Module): The model to modify.
        prefixes (tuple, optional):
            Set preserve_qat_mode by the prefix of qualified name.
            Defaults to tuple().
        types (tuple, optional):
            Set preserve_qat_mode by module type. Defaults to tuple().
            If float model, types must be float module types
            If QAT model, types can be float or qat module types
        value (bool, optional):
            Set preserve_qat_mode to this value.
            Defaults to True.
    """
    is_qat_stage = False
    float_qat_map = get_qat_module_mappings()
    qat_modules = set(float_qat_map.values())
    for _, mod in model.named_modules():
        if type(mod) in qat_modules:
            is_qat_stage = True
            break

    if is_qat_stage:
        # if conv.perserve_qat_mode = True, then fused.preserve_qat_mode = True
        qat_types = []
        for qat_mod_class in float_qat_map.values():
            if issubclass(qat_mod_class, types):
                qat_types.append(qat_mod_class)
        for t in types:
            if t in float_qat_map:
                qat_types.append(float_qat_map[t])

        types += type(types)(qat_types)

    def _delete_submodule_qat_mode(mod):
        if hasattr(mod, "preserve_qat_mode"):
            del mod.preserve_qat_mode
        for _, child in mod.named_children():
            _delete_submodule_qat_mode(child)

    for name, mod in model.named_modules():
        if type(mod) in types:
            if type(mod) in (torch.nn.Identity, horizon_nn.Identity):
                logger.warning(
                    f"The module {name} is Identity, which may be fused in "
                    f"conv. Please find the fused conv module to set "
                    f"preserve_qat_mode"
                )
                continue
            _delete_submodule_qat_mode(mod)
            mod.preserve_qat_mode = value
        elif name in prefixes:
            if type(mod) in (torch.nn.Identity, horizon_nn.Identity):
                logger.warning(
                    f"The module {name} is Identity, which may be fused in "
                    f"conv. Please find the fused conv module to set "
                    f"preserve_qat_mode"
                )
                continue
            _delete_submodule_qat_mode(mod)
            mod.preserve_qat_mode = value
