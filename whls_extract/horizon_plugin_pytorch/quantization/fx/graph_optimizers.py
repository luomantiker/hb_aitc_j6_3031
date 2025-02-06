import copy
import inspect
import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.fx import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.nn import functional as F  # noqa: N812
from torch.utils._pytree import tree_flatten

from horizon_plugin_pytorch import nn as nnf
from horizon_plugin_pytorch._torchvision_wrapper import ops as nnv
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.fx.jit_scheme import GraphModule as JitGraphModule
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.nn.qat import SegmentLUT
from horizon_plugin_pytorch.quantization.qconfig import (
    QConfig,
    replace_qconfig_dtype,
)
from horizon_plugin_pytorch.quantization.qconfig_template import (
    ModuleNameQconfigSetter,
    QconfigSetterBase,
)
from horizon_plugin_pytorch.utils.model_helper import (
    get_qconfig_useless_modules,
    is_ff_node,
    is_fused_ff_node,
)

logger = logging.getLogger(__name__)

# Mapping from ops to their input indices.
# keys are ops which only support int8 input.
# values are input indices which only support int8 dtype.
DEFAULT_INPUT_INT8_ONLY = {
    nn.Upsample: [0],
    nn.UpsamplingBilinear2d: [0],
    nn.UpsamplingNearest2d: [0],
    nn.MaxPool1d: [0],
    nn.MaxPool2d: [0],
    nn.MaxPool3d: [0],
    nn.AdaptiveAvgPool1d: [0],
    nn.AdaptiveAvgPool2d: [0],
    nn.AdaptiveAvgPool3d: [0],
    intrinsic.ConvAdd2d: [0, 1],
    intrinsic.ConvAddReLU2d: [0, 1],
    intrinsic.ConvAddReLU62d: [0, 1],
    intrinsic.ConvBNAdd2d: [0, 1],
    intrinsic.ConvBNAddReLU2d: [0, 1],
    intrinsic.ConvBNAddReLU62d: [0, 1],
    intrinsic.LinearAdd: [0, 1],
    intrinsic.LinearAddReLU: [0, 1],
    intrinsic.LinearAddReLU6: [0, 1],
    intrinsic.DeformConvAdd2d: [0, 1],
    intrinsic.DeformConvAddReLU2d: [0, 1],
    intrinsic.DeformConvAddReLU62d: [0, 1],
    nnf.Interpolate: [0],
    nnf.interpolate.autocasted_interpolate: [0],
    nnf.interpolate.autocasted_interpolate_outer: [0],
    nnf.grid_sample.autocasted_grid_sample: [0],
    nnf.grid_sample.autocasted_grid_sample_outer: [0],
    nnf.grid_sample.warp: [0],
    nnf.GridSample: [0],
    nnf.DetectionPostProcess: [0, 1, 2],
    nnf.RcnnPostProcess: [0, 1, 2],
    nnf.MultiScaleRoIAlign: [0],
    nnv.RoIAlign: [0, 1],
    nnf.GridSample: [0],
    F.max_pool2d: [0],
    F.adaptive_max_pool1d: [0],
    F.adaptive_max_pool2d: [0],
    F.avg_pool2d: [0],
}

# Set of ops which only support int8 output.
DEFAULT_OUTPUT_INT8_ONLY = {
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nnf.Interpolate,
    nn.Upsample,
    nn.UpsamplingBilinear2d,
    nn.UpsamplingNearest2d,
    nnf.GridSample,
    nnf.MultiScaleRoIAlign,
    nnf.DetectionPostProcess,
    nnf.RcnnPostProcess,
    nnv.RoIAlign,
    nn.MultiheadAttention,
}


def matmul_int16_fallback(
    mod_name, mod, used_node_target_pos, all_input_fallback=True, rank=None
):
    """
    Fallback qconfig of matmul input from int16 to int8.

    1. all_input_fallback is True: force both inputs fallback to int8.
    2. all_input_fallback is False:
        a. rank is None: fallback input with lower topological order to int8.
        b. rank is provided: fallback input with lower rank or without rank
            to int8.

    Args:
        mod_name: Input module name.
        mod: Input module.
        used_node_target_pos: List of user node, module called by user node,
            input index of user node.
        all_input_fallback: Force all input of matmul fallback to int8.
            Defaults to True.
        rank: List of module name. The smaller the index, the more important
            it is. Defaults to None.
    """
    for user_node, target, _ in used_node_target_pos:
        if (
            type(target) == nnf.quantized.FloatFunctional
            and hasattr(target, "_last_called_method_name")
            and target._last_called_method_name == "matmul"
        ):
            if user_node.op != "call_method":
                raise RuntimeError(
                    "Unexpected matmul node type {}".format(user_node.op)
                )
            user_mod_name = user_node.args[0].target

            if all_input_fallback:
                # force input fallback to int8.
                logger.info(
                    f"{user_mod_name} doesn't support int16 input, "
                    f"the output dtype of {mod_name} will fallback to int8."
                )
                mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)
                return
            if not hasattr(target, "_int16_input"):
                # record current input module as '_int16_input' attr
                target._int16_input = (mod_name, mod)
            elif rank is None:
                # keep module in '_int16_input' attr as int16 and
                # fallback current input to int8.
                logger.info(
                    f"{user_mod_name} doesn't support two int16 inputs, "
                    f"the output dtype of {mod_name} will fallback to int8."
                )
                mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)
                return
            else:
                # fallback input with lower rank or without rank to int8.
                first_mod_name, first_mod = target._int16_input
                delattr(target, "_int16_input")
                try:
                    first_rank = rank.index(first_mod_name)
                except ValueError:
                    logger.info(
                        f"{user_mod_name} doesn't support two int16 inputs, "
                        f"the output dtype of {first_mod_name} will fallback "
                        f"to int8."
                    )
                    first_mod.qconfig = replace_qconfig_dtype(
                        first_mod.qconfig, qint8
                    )
                    return

                try:
                    second_rank = rank.index(mod_name)
                except ValueError:
                    logger.info(
                        f"{user_mod_name} doesn't support two int16 inputs, "
                        f"the output dtype of {mod_name} will fallback to "
                        f"int8."
                    )
                    mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)
                    return

                if first_rank < second_rank:
                    logger.info(
                        f"{user_mod_name} doesn't support two int16 inputs, "
                        f"the output dtype of {mod_name} will fallback to "
                        f"int8."
                    )
                    mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)
                else:
                    logger.info(
                        f"{user_mod_name} doesn't support two int16 inputs, "
                        f"the output dtype of {first_mod_name} will fallback "
                        f"to int8."
                    )
                    first_mod.qconfig = replace_qconfig_dtype(
                        first_mod.qconfig, qint8
                    )
                return


def default_march_specific_func(mod_name, mod, used_node_target_pos, rank):
    """
    March specific fallback function.

    Used to handle operators which cannot be distinguished by module type.
    e.g. quantized.FloatFunctional.

    Args:
        mod_name: Input module name.
        mod: Input module.
        used_node_target_pos: List of user node, module called by user node,
            input index of user node.
        rank: List of module name. The smaller the index, the more important
            it is. Defaults to None.
    """
    matmul_int16_fallback(mod_name, mod, used_node_target_pos)


def nash_m_specific_func(mod_name, mod, used_node_target_pos, rank):
    matmul_int16_fallback(
        mod_name,
        mod,
        used_node_target_pos,
        all_input_fallback=False,
        rank=rank,
    )


MARCH_LIMIT_MAPPING = {
    March.NASH_E: (
        DEFAULT_INPUT_INT8_ONLY,
        DEFAULT_OUTPUT_INT8_ONLY,
        nash_m_specific_func,
    ),
    March.NASH_M: (
        DEFAULT_INPUT_INT8_ONLY,
        DEFAULT_OUTPUT_INT8_ONLY,
        nash_m_specific_func,
    ),
    March.NASH_P: (
        DEFAULT_INPUT_INT8_ONLY,
        DEFAULT_OUTPUT_INT8_ONLY,
        nash_m_specific_func,
    ),
}


class BaseGraphPass:
    def __call__(*args, **kwargs):
        msg = "Graph pass function is not implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)

    def _get_module_called_by_node(
        self, model: Union[GraphModule, JitGraphModule], node: Node
    ):
        """Get module called by node. Return path and object of the module."""
        if node.op == "call_module":
            return node.target, model.get_submodule(node.target)
        if is_ff_node(node, model) or is_fused_ff_node(node, model):
            return (
                node.args[0].target,
                model.get_submodule(node.args[0].target),
            )
        return None, None

    def _is_node_in_arg(self, node, arg):
        """Check if node is in arg."""
        flat_args = [i for i in tree_flatten(arg)[0] if isinstance(i, Node)]
        return node in flat_args

    def _get_users_to_next_module(
        self, model: Union[GraphModule, JitGraphModule], node: Node
    ) -> List[Tuple[Node, torch.nn.Module, int]]:
        """
        Get users of which input dtypes are affected by current node.

        Traverse users of the node, return list of user node, module
        called by user node, input index of user node.
        """

        rets = []
        for user in node.users:
            if isinstance(user, Node):
                _, user_mod = self._get_module_called_by_node(model, user)
                if user_mod is not None or user.op == "call_function":
                    target = user.target if user_mod is None else user_mod
                    is_node_in_args = False
                    for idx, arg in enumerate(user.args):
                        if self._is_node_in_arg(node, arg):
                            is_node_in_args = True
                            rets.append(
                                (
                                    user,
                                    target,
                                    idx - 1
                                    if is_ff_node(user, model)
                                    or is_fused_ff_node(user, model)
                                    else idx,
                                )
                            )

                    if len(user.kwargs) > 0:
                        if isinstance(target, nn.Module):
                            # function will convert kwargs to args
                            # automatically, the logic is only used by module.
                            if isinstance(
                                target, nnf.quantized.FloatFunctional
                            ):
                                func = getattr(target, user.target, None)
                                sig = (
                                    []
                                    if func is None
                                    else inspect.signature(func).parameters
                                )
                            else:
                                sig = inspect.signature(
                                    target.forward
                                ).parameters

                            args = [
                                user.kwargs[i]
                                if i in user.kwargs
                                else sig[i].default
                                for i in sig
                            ]
                            for idx, arg in enumerate(args):
                                if self._is_node_in_arg(node, arg):
                                    is_node_in_args = True
                                    rets.append((user, target, idx))
                        else:
                            # build-in method can't get kwargs by signature
                            # check with given order and log warning
                            for idx, arg in enumerate(user.kwargs.values()):
                                if self._is_node_in_arg(node, arg):
                                    logger.warning(
                                        f"{target} doesn't have signature. "
                                        f"Please use positional args instead "
                                        f"of keyword args. Otherwise, qconfig "
                                        f"canonicalize may not work as "
                                        f"expected."
                                    )
                                    is_node_in_args = True
                                    rets.append((user, target, idx))

                    if not is_node_in_args:
                        raise ValueError(
                            f"Invalid graph detected, attachment "
                            f"conflicts between \n{node}\n and "
                            f"\n{user}\n"
                        )

                if (
                    isinstance(user_mod, get_qconfig_useless_modules())
                    or user.op == "call_function"
                ):
                    rets.extend(self._get_users_to_next_module(model, user))

        return rets


class QconfigCanonicalizer(BaseGraphPass):
    """Qconfig Canonicalizer.

    1. Check compiler restriction and fallback unsupported int16 to int8
    2. Reduce unnecessary fake cast/quant.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        model: Union[GraphModule, JitGraphModule],
        qconfig_setter: Optional[Tuple[QconfigSetterBase, ...]] = None,
    ):
        module_name_to_qconfig = {}
        if qconfig_setter is not None:
            for setter in qconfig_setter:
                if isinstance(setter, ModuleNameQconfigSetter):
                    for (
                        mod_name,
                        qconfig,
                    ) in setter.module_name_to_qconfig.items():
                        if mod_name not in module_name_to_qconfig:
                            module_name_to_qconfig[mod_name] = qconfig
        rank = (
            list(module_name_to_qconfig.keys())
            if len(module_name_to_qconfig) > 0
            else None
        )

        graph: Graph = model.graph
        for node in graph.nodes:
            node: Node
            mod_name, mod = self._get_module_called_by_node(model, node)
            if mod is not None and not isinstance(
                mod, (torch.nn.Identity, nnf.Identity)
            ):
                used_node_target_pos = self._get_users_to_next_module(
                    model, node
                )
                self._dtype_fallback(mod_name, mod, used_node_target_pos, rank)
                self._reduce_qconfig(mod, used_node_target_pos)

    def _dtype_fallback(self, mod_name, mod, used_node_target_pos, rank):
        """Fallback unsupported int16 to int8."""
        qconfig: QConfig = getattr(mod, "qconfig", None)
        # If current mod is qconfig_useless_modules, its output should already
        # handled by previous mod.
        if (
            qconfig is None
            or qconfig.activation is None
            or qconfig.activation().get_dtype() != qint16
            or isinstance(mod, get_qconfig_useless_modules())
        ):
            return

        march = get_march()
        (
            input_int8_only,
            output_int8_only,
            march_specific_func,
        ) = MARCH_LIMIT_MAPPING.get(
            march,
            (
                DEFAULT_INPUT_INT8_ONLY,
                DEFAULT_OUTPUT_INT8_ONLY,
                default_march_specific_func,
            ),
        )

        if type(mod) in output_int8_only:
            logger.info(
                f"{mod_name} doesn't support int16 output, "
                f"the output dtype of {mod_name} will fallback to int8."
            )
            mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)

        for user_node, target, pos in used_node_target_pos:
            if (
                user_node.op == "call_function"
                and target in input_int8_only
                and pos in input_int8_only[target]
            ) or (
                type(target) in input_int8_only
                and pos in input_int8_only[type(target)]
            ):
                logger.info(
                    f"{user_node.name} doesn't support int16 input, "
                    f"the output dtype of {mod_name} will fallback to int8."
                )
                mod.qconfig = replace_qconfig_dtype(mod.qconfig, qint8)

        if march_specific_func is not None:
            march_specific_func(mod_name, mod, used_node_target_pos, rank)

    def _reduce_qconfig(self, mod, used_node_target_pos):
        """Reduce unnecessary fake cast/quant."""
        qconfig: QConfig = getattr(mod, "qconfig", None)
        if qconfig is None or qconfig.activation is None:
            output_dtype = torch.float
        else:
            output_dtype = qconfig.activation().get_dtype()

        for _, mod, input_idx in used_node_target_pos:
            if (
                hasattr(mod, "qconfig")
                and mod.qconfig is not None
                and hasattr(mod.qconfig, "input")
                and mod.qconfig.input is not None
            ):
                if isinstance(mod.qconfig.input, (list, tuple)):
                    if (
                        mod.qconfig.input[input_idx] is not None
                        and mod.qconfig.input[input_idx]().get_dtype()
                        == output_dtype
                    ):
                        mod.qconfig.input[input_idx] = None
                else:
                    if mod.qconfig.input().get_dtype() == output_dtype:
                        mod.qconfig = QConfig(
                            mod.qconfig.activation, mod.qconfig.weight, None
                        )


class LUTInputMinMax(BaseGraphPass):
    """Find SegmentLUT input observer."""

    def __init__(self) -> None:
        pass

    def __call__(
        self, model: Union[GraphModule, JitGraphModule], example_inputs
    ):
        trace_model = copy.deepcopy(model)
        graph_module = Tracer().trace(trace_model, example_inputs)
        graph: Graph = graph_module.graph
        graph.print_tabular()
        lut_input_mod = {}

        # find lut input
        for node in graph.nodes:
            node: Node
            mod_name, mod = self._get_module_called_by_node(model, node)
            if (
                mod is not None
                and not isinstance(mod, (torch.nn.Identity, nnf.Identity))
                and not isinstance(mod, get_qconfig_useless_modules())
            ):
                used_node_target_pos = self._get_users_to_next_module(
                    model, node
                )
                for user_node, _, _ in used_node_target_pos:
                    user_mod_name, user_mod = self._get_module_called_by_node(
                        model, user_node
                    )
                    if isinstance(user_mod, SegmentLUT) and hasattr(
                        mod, "activation_post_process"
                    ):
                        if user_mod_name not in lut_input_mod:
                            lut_input_mod[user_mod_name] = [
                                mod_name,
                            ]
                        else:
                            lut_input_mod[user_mod_name].append(mod_name)

        # attach input observer to lut
        for lut_name, input_name_list in lut_input_mod.items():
            input_num = len(set(input_name_list))
            if input_num != 1:
                # segment inputs come from two different mods
                logger.warning(
                    f"{lut_name} shared between {input_num} different modules "
                    f"{input_name_list}"
                )
            else:
                lut_mod = model.get_submodule(lut_name)
                input_mod = model.get_submodule(input_name_list[0])
                observer = (
                    input_mod.activation_post_process.activation_post_process
                )
                if hasattr(observer, "min_val") and hasattr(
                    observer, "max_val"
                ):
                    lut_mod._input_observer = (input_name_list[0], observer)
