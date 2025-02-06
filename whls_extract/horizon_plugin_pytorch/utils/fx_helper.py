r"""Extended tracer and wrap of torch.fx.

This file defines a inherit tracer of torch.fx.Tracer and a extended wrap to
allow wrapping of user-defined Module or method, which help users do some
optimization of their own module by torch.fx
"""
import sys

from horizon_plugin_pytorch.fx import CustomTracer
from horizon_plugin_pytorch.fx.fx_helper import (
    convert_fx_node_name,
    get_fx_node_input_output,
    get_supported_method,
    is_fx_node_name_match_module_name,
    replace_torch_op,
    wrap,
)
from horizon_plugin_pytorch.utils import deprecated_module_attr_warning

__all__ = [
    "wrap",
    "CustomTracer",
    "get_supported_method",
    "replace_torch_op",
    "convert_fx_node_name",
    "is_fx_node_name_match_module_name",
    "get_fx_node_input_output",
]


def default_msg(old_interface, new_interface):
    msg = (
        f"{__name__}.{old_interface} will be deprecated after 2.5.1. "
        f"Use {new_interface} instead of {__name__}.{old_interface}."
    )
    return msg


sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "wrap",
    msg=default_msg(
        "wrap",
        "horizon_plugin_pytorch.fx.fx_helper.wrap",
    ),
)


sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "CustomTracer",
    msg=default_msg(
        "CustomTracer",
        "horizon_plugin_pytorch.fx.tracer.CustomTracer",
    ),
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "get_supported_method",
    msg=default_msg(
        "get_supported_method",
        "horizon_plugin_pytorch.fx.fx_helper.get_supported_method",
    ),
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "replace_torch_op",
    msg=default_msg(
        "replace_torch_op",
        "horizon_plugin_pytorch.fx.fx_helper.replace_torch_op",
    ),
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "convert_fx_node_name",
    msg=default_msg(
        "convert_fx_node_name",
        "horizon_plugin_pytorch.fx.fx_helper.convert_fx_node_name",
    ),
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "is_fx_node_name_match_module_name",
    msg=default_msg(
        "is_fx_node_name_match_module_name",
        (
            "horizon_plugin_pytorch.fx.fx_helper."
            "is_fx_node_name_match_module_name"
        ),
    ),
)

sys.modules[__name__] = deprecated_module_attr_warning(
    sys.modules[__name__],
    "get_fx_node_input_output",
    msg=default_msg(
        "get_fx_node_input_output",
        "horizon_plugin_pytorch.fx.fx_helper.get_fx_node_input_output",
    ),
)
