import logging
import operator
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn
from torch.fx.graph import Node
from torch.quantization.fuser_method_mappings import get_fuser_method

# from torchvision import ops
from horizon_plugin_pytorch._torchvision_wrapper import ops
from horizon_plugin_pytorch.nn import Identity
from horizon_plugin_pytorch.qat_mode import QATMode, get_qat_mode, tricks
from horizon_plugin_pytorch.utils.model_helper import (
    get_float_functional_classes,
)
from ..fuse_modules import get_op_list_to_fuser_mapping
from ..qconfig import QConfig
from .pattern_utils import MatchAllNode, register_fusion_pattern
from .quantization_types import QuantizerCls
from .utils import _parent_name

logger = logging.getLogger(__name__)
# ---------------------
# Fusion Pattern Registrations
# ---------------------


# Not used currently
# TODO: refine the order of fuse, qconfig_tamplate, propagate_qconfig
# and swap_module
def get_fused_qconfig(mod_list, add_input_idx=0):
    if hasattr(mod_list[0], "qconfig"):
        weight_config = getattr(mod_list[0].qconfig, "weight", None)
    else:
        weight_config = None

    if hasattr(mod_list[-1], "qconfig"):
        activation_config = getattr(mod_list[-1].qconfig, "activation", None)
    else:
        activation_config = None

    if hasattr(mod_list[0], "qconfig"):
        input_qconfig = getattr(mod_list[0].qconfig, "input", None)
    else:
        input_qconfig = None

    for mod in mod_list:
        if isinstance(mod, get_float_functional_classes()):
            if (
                hasattr(mod, "qconfig")
                and hasattr(mod.qconfig, "input")
                and mod.qconfig.input is not None
            ):
                add_input_config = list(mod.qconfig.input)
            else:
                add_input_config = [None, None]
            add_input_config[add_input_idx] = input_qconfig
            if add_input_config[0] is None and add_input_config[1] is None:
                input_qconfig = None
            else:
                input_qconfig = add_input_config

    if (
        weight_config is None
        and activation_config is None
        and input_qconfig is None
    ):
        return None
    else:
        return QConfig(activation_config, weight_config, input_qconfig)


# Base Pattern Handler
class FuseHandler(ABC):
    """Base handler class for the fusion patterns."""

    _common_relu_name = "_common_relu_name_in_fuse"

    def __init__(self, quantizer: QuantizerCls, node: Node):
        pass

    @abstractmethod
    def fuse(
        self,
        quantizer: QuantizerCls,
        env: Dict[str, Node],
        load_arg: Callable,
        module_called_times: Dict[str, int],
        fuse_custom_config_dict: Dict[str, Any] = None,
    ):
        pass


class ConvBNAddReLUFusion(FuseHandler):
    """This handler only fuse add with its first input node currently."""

    def __init__(
        self, quantizer: QuantizerCls, node: Node, reverse_add_input=False
    ):
        super().__init__(quantizer, node)
        from horizon_plugin_pytorch import nn as horizon_nn

        self.reverse_add_input = reverse_add_input
        self.quantizer = quantizer
        self.root_name = node.name
        self.node_list = []
        self.relu_node = None
        self.add_node = None
        self.bn_node = None
        self.preserve_qat_mode = False

        # get relu node
        if (
            (node.op == "call_function" and node.target is nn.functional.relu)
            or (
                node.op == "call_module"
                and type(quantizer.modules[node.target]) == nn.ReLU
            )
            or (node.op == "call_method" and node.target == "relu")
        ):
            self.relu_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            self.relu_class = nn.ReLU
        elif (
            (node.op == "call_function" and node.target is nn.functional.relu6)
            or (
                node.op == "call_module"
                and type(quantizer.modules[node.target]) == nn.ReLU6
            )
            or (node.op == "call_method" and node.target == "relu6")
        ):
            self.relu_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            self.relu_class = nn.ReLU6

        # get add node
        if (
            node.op == "call_function"
            and node.target in (torch.add, operator.add)
        ) or (
            node.op == "call_method" and node.target == "add"
        ):  # Tensor.add or FloatFunctional.add
            self.add_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)

            add_input_idx = 1 if self.reverse_add_input else 0

            if node.args[0].op == "get_attr":
                assert type(quantizer.modules[node.args[0].target]) in (
                    torch.nn.quantized.FloatFunctional,
                    horizon_nn.quantized.FloatFunctional,
                )
                add_input_idx += 1  # skip get_attr node

            node = node.args[add_input_idx]

        # get bn node
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            nn.SyncBatchNorm,
            horizon_nn.ChannelScale2d,
        ]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.node_list.append(node)

        self.conv = quantizer.modules[self.conv_node.target]

        # process preserve qat mode
        # only conv.preserve_qat_mode = True, fused.preserve_qat_mode = True
        self.preserve_qat_mode = getattr(self.conv, "preserve_qat_mode", False)

    def fuse(
        self,
        quantizer: QuantizerCls,
        env: Dict[str, Node],
        load_arg: Callable,
        module_called_times: Dict[str, int],
        fuse_custom_config_dict: Dict[str, Any] = None,
        fused_map: Dict[Tuple, torch.nn.Module] = None,
    ):
        duplicate_shared_convbn = tricks.fx_force_duplicate_shared_convbn
        additional_fuser_method_mapping = get_op_list_to_fuser_mapping()
        if fuse_custom_config_dict is not None:
            additional_fuser_method_mapping.update(
                fuse_custom_config_dict.get(
                    "additional_fuser_method_mapping", {}
                )
            )

        op_list = []
        op_name_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to
            # create a relu module for each match
            if self.relu_node.op == "call_module":
                relu = self.relu_class(
                    quantizer.modules[self.relu_node.target].inplace
                )
                op_name_list.append(self.relu_node.target)
                module_called_times[self.relu_node.target] -= 1
                if hasattr(
                    quantizer.modules[self.relu_node.target], "qconfig"
                ):
                    relu.qconfig = quantizer.modules[
                        self.relu_node.target
                    ].qconfig
            else:
                inplace = False
                if len(self.relu_node.args) > 1:
                    inplace = self.relu_node.args[1]
                relu = self.relu_class(inplace)
            relu.training = self.conv.training
            op_list.append(relu)
        if self.add_node is not None:
            if self.add_node.op == "call_method":
                add = quantizer.modules[self.add_node.args[0].target]
                op_name_list.append(self.add_node.args[0].target)
                module_called_times[self.add_node.args[0].target] -= 1
            else:
                from horizon_plugin_pytorch import nn as horizon_nn

                add = horizon_nn.quantized.FloatFunctional()
            add.training = self.conv.training
            op_list.append(add)
        if self.bn_node is not None:
            op_list.append(self.bn)
            op_name_list.append(self.bn_node.target)
            module_called_times[self.bn_node.target] -= 1
        op_list.append(self.conv)
        op_name_list.append(self.conv_node.target)
        module_called_times[self.conv_node.target] -= 1

        assert len(op_list) > 1

        op_list.reverse()
        op_name_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        fuser_method = get_fuser_method(
            op_type_list, additional_fuser_method_mapping
        )
        if fuser_method is None:
            raise RuntimeError("Cannot fuse modules: {}".format(op_type_list))

        sig = signature(fuser_method)
        key = tuple(
            [id(m) for m in op_list if not isinstance(m, (nn.ReLU, nn.ReLU6))]
        )
        if not duplicate_shared_convbn and key in fused_map:
            fused = fused_map[key]
        else:
            if len(sig.parameters) > len(op_list):
                fused = fuser_method(
                    *op_list,
                    fuse_bn=get_qat_mode()
                    not in (QATMode.WithBN, QATMode.WithBNReverseFold)
                )
            else:
                fused = fuser_method(*op_list)

        if self.preserve_qat_mode:
            fused.preserve_qat_mode = True

        # set qconfig of fused mod
        if hasattr(op_list[-1], "qconfig"):
            fused.qconfig = op_list[-1].qconfig

        if self.add_node is not None and self.add_node.op == "call_method":
            module_called_times[self.add_node.args[0].target] += 1
            # replace add if fuse FloatFunctional.add
            getattr_node = self.add_node.args[0]
            add_parent_name, add_name = _parent_name(getattr_node.target)
            setattr(quantizer.modules[add_parent_name], add_name, fused)

            env[getattr_node.name] = quantizer.fused_graph.node_copy(
                getattr_node, load_arg
            )

            new_node = torch.fx.Node(
                self.add_node.graph,
                self.add_node.name,
                "call_module",
                getattr_node.target,
                (
                    (self.conv_node.args[0], self.add_node.args[-2])
                    if self.reverse_add_input
                    else (
                        self.conv_node.args[0],
                        self.add_node.args[-1],
                    )
                ),
                {},
            )

        elif (
            module_called_times[self.conv_node.target] > 0
            and self.bn_node is not None
            and module_called_times[self.bn_node.target] == 0
            and (duplicate_shared_convbn or key not in fused_map)
        ):
            # replace bn if conv is shared and bn is not
            module_called_times[self.bn_node.target] += 1
            bn_parent_name, bn_name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[bn_parent_name], bn_name, fused)

            new_node = torch.fx.Node(
                self.bn_node.graph,
                self.bn_node.name,
                "call_module",
                self.bn_node.target,
                (self.conv_node.args[0],),
                {},
            )

        else:
            module_called_times[self.conv_node.target] += 1
            # replace conv in other cases
            convs = (
                nn.Conv1d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.Conv3d,
                nn.Linear,
                ops.DeformConv2d,
            )
            if type(fused) not in convs and type(
                quantizer.modules[self.conv_node.target]
            ) is type(fused):
                # already fused (shared head for example)
                fused_dict = quantizer.modules[
                    self.conv_node.target
                ].state_dict()
                fusing_dict = fused.state_dict()
                for k in fused_dict:
                    try:
                        torch.testing.assert_allclose(
                            fused_dict[k], fusing_dict[k]
                        )
                    except AssertionError:
                        raise RuntimeError(
                            "find fused op in {} when trying to fuse {}, "
                            "and params are conflict".format(
                                self.conv_node.target, op_name_list
                            )
                        )
            else:
                conv_parent_name, conv_name = _parent_name(
                    self.conv_node.target
                )
                setattr(quantizer.modules[conv_parent_name], conv_name, fused)

            new_node = self.conv_node

        fused_map[key] = fused
        env[self.root_name] = quantizer.fused_graph.node_copy(
            new_node, load_arg
        )

        for qualified_name, times in module_called_times.items():
            if times == 0:
                parent_name, name = _parent_name(qualified_name)
                setattr(quantizer.modules[parent_name], name, Identity())

    def set_mod(self, mod_name, new_mod):
        parent_name, mod_name = _parent_name(mod_name)
        setattr(self.quantizer.modules[parent_name], mod_name, new_mod)

    def set_to_identity(self, mod_name, relu_mod_name=None):
        if mod_name == self._common_relu_name:
            assert relu_mod_name is not None
            self.set_mod(relu_mod_name, Identity())
        else:
            self.set_mod(mod_name, Identity())

    def fuse_on_module(
        self,
        quantizer: QuantizerCls,
        module_called_times: Dict[str, int],
        fuse_custom_config_dict: Dict[str, Any] = None,
        fused_map: Dict[Tuple[str, ...], torch.nn.Module] = None,
        fused_modules: Dict[str, int] = None,
    ):
        """Apply fuse by swap module and preserve origin forward code.

        Args:
            quantizer (QuantizerCls):
                The Quantizer instance which hold need info.
            module_called_times (Dict[str, int]):
                A mapping from module's name to its called time in one forward.
            fuse_custom_config_dict (Dict[str, Any], optional):
                A dict with key 'additional_fuser_method_mapping' to give
                custom pair of module seq to their fuse function.
                Not used now.
                Defaults to None.
            fused_map (Dict[Tuple[str, ...], torch.nn.Module], optional):
                A mapping to hold a mapping from already fused module sequence
                (record by their name) to corresponding fused module.
                Defaults to None.
            fused_modules (Dict[str, int], optional):
                A mapping from module's name to its fused time.
                For shared Conv un-shared Bn, we only replce Conv with
                Identity after its fused time reach
                Conv-called-time // Bn-called-time.
                Defaults to None.
        """
        if tricks.fx_force_duplicate_shared_convbn:
            raise ValueError(
                "fx_force_duplicate_shared_convbn is not supported "
                "in fuse_on_module"
            )

        additional_fuser_method_mapping = get_op_list_to_fuser_mapping()
        if fuse_custom_config_dict is not None:
            additional_fuser_method_mapping.update(
                fuse_custom_config_dict.get(
                    "additional_fuser_method_mapping", {}
                )
            )

        # get fuse lists
        op_list: List[torch.nn.Module] = []
        op_name_list: List[str] = []
        op_list.append(self.conv)
        op_name_list.append(self.conv_node.target)
        if self.bn_node is not None:
            op_list.append(self.bn)
            op_name_list.append(self.bn_node.target)
        if self.add_node is not None:
            assert self.add_node.op == "call_method"
            add_mod_name = self.add_node.args[0].target
            op_list.append(quantizer.modules[add_mod_name])
            op_name_list.append(add_mod_name)
        relu_mod_name = None
        if self.relu_node is not None:
            assert self.relu_node.op == "call_module"
            relu_mod_name = self.relu_node.target
            if (
                module_called_times[relu_mod_name]
                <= module_called_times[op_name_list[0]]
            ):
                op_list.append(quantizer.modules[relu_mod_name])
                # use a common name 'relu' because we allow fuse multi relu
                # into one conv
                op_name_list.append(self._common_relu_name)
            else:
                logger.warning(
                    "ReLU in fuse pattern called more than conv in a forward:"
                    " {}({}) vs {}({})\nReLU will not be fused".format(
                        relu_mod_name,
                        module_called_times[relu_mod_name],
                        op_name_list[0],
                        module_called_times[op_name_list[0]],
                    )
                )
                if len(op_list) == 1:
                    # only fuse conv+relu
                    return

        assert len(op_list) > 1

        op_name_list = tuple(op_name_list)
        op_type_list = tuple(type(m) for m in op_list)

        # check if already fused, shared head for example
        if op_name_list in fused_map:
            if self.relu_node is not None:
                self.set_to_identity(relu_mod_name)
            return

        # get fused module
        fuser_method = get_fuser_method(
            op_type_list, additional_fuser_method_mapping
        )
        if fuser_method is None:
            raise RuntimeError("Cannot fuse modules: {}".format(op_type_list))

        sig = signature(fuser_method)
        if len(sig.parameters) > len(op_list):
            fused = fuser_method(
                *op_list,
                fuse_bn=get_qat_mode()
                not in (QATMode.WithBN, QATMode.WithBNReverseFold)
            )
        else:
            fused = fuser_method(*op_list)

        if self.preserve_qat_mode:
            fused.preserve_qat_mode = True

        # set qconfig of fused mod
        if hasattr(op_list[-1], "qconfig"):
            fused.qconfig = op_list[-1].qconfig

        called_times_list = tuple(
            module_called_times[n]
            for n in op_name_list
            if n != self._common_relu_name
        )
        for i in range(1, len(called_times_list)):
            if called_times_list[i] != called_times_list[1]:
                logger.warning(
                    "Mod in fuse pattern called different times in a forward:"
                    " {}({}) vs {}({})\nFusion is skipped".format(
                        op_name_list[1],
                        called_times_list[1],
                        op_name_list[i],
                        called_times_list[i],
                    )
                )
                return

        if (
            len(called_times_list) > 1
            and called_times_list[0] > called_times_list[1]
        ):
            # shared conv
            for mod_name in op_name_list[1:]:
                if mod_name in fused_modules:
                    for k in fused_map:
                        if mod_name in k:
                            raise RuntimeError(
                                "Mod {} which is already fused in {} "
                                "occurs in another fuse pattern {}".format(
                                    mod_name, k, op_name_list
                                )
                            )

            # the "shared conv" pattern may be called multi times
            if called_times_list[0] % called_times_list[1] != 0:
                logger.warning(
                    "Conv called times is not divisible by other mod "
                    "called times: {}({}) vs {}({})\nFusion is skipped".format(
                        op_name_list[0],
                        called_times_list[0],
                        op_name_list[1],
                        called_times_list[1],
                    )
                )
                return

            conv_shared_times = called_times_list[0] // called_times_list[1]

            if self.add_node is not None:
                # fuse to add
                self.set_mod(add_mod_name, fused)
                for mod_name in op_name_list[1:]:
                    if mod_name == add_mod_name:
                        continue
                    self.set_to_identity(mod_name, relu_mod_name)
            else:
                # fuse to last module (except generated relu),
                # same as eager mode
                replace_mod_name = op_name_list[-1]
                if replace_mod_name == self._common_relu_name:
                    replace_mod_name = op_name_list[-2]
                self.set_mod(replace_mod_name, fused)
                for mod_name in op_name_list[1:]:
                    if mod_name == replace_mod_name:
                        continue
                    self.set_to_identity(mod_name, relu_mod_name)
            if fused_modules.get(op_name_list[0], 0) + 1 >= conv_shared_times:
                # if this is the last fuse of conv, replace it with identity
                self.set_to_identity(op_name_list[0])
        else:
            for mod_name in op_name_list:
                if mod_name in fused_modules:
                    for k in fused_map:
                        if mod_name in k:
                            raise RuntimeError(
                                "Mod {} which is already fused in {} "
                                "occurs in another fuse pattern {}".format(
                                    mod_name, k, op_name_list
                                )
                            )

            # replace all mods with identity
            for mod_name in op_name_list:
                self.set_to_identity(mod_name, relu_mod_name)

            if self.add_node is not None:
                # replace add if fuse FloatFunctional.add
                self.set_mod(add_mod_name, fused)
            else:
                # replace conv in other cases
                self.set_mod(op_name_list[0], fused)

        if self.add_node is not None and self.reverse_add_input:
            assert hasattr(fused, "swap_inputs")
            fused.swap_inputs()

        fused_map[op_name_list] = fused
        for op_name in op_name_list:
            if op_name != self._common_relu_name:
                fused_modules[op_name] = fused_modules.get(op_name, 0) + 1


class ConvBNAddedReLUFusion(ConvBNAddReLUFusion):
    """This handler only fuse add with its second input node currently."""

    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node, True)


def register_conv_bn_add_relu_fusion_patterns():
    from horizon_plugin_pytorch import nn as horizon_nn

    convs = (
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.Conv3d,
        nn.Linear,
        ops.DeformConv2d,
    )
    bns = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        horizon_nn.ChannelScale2d,
    )
    adds = (
        nn.quantized.FloatFunctional.add,
        horizon_nn.quantized.FloatFunctional.add,
        torch.add,
        operator.add,
    )
    relus = (nn.ReLU, nn.ReLU6, nn.functional.relu, nn.functional.relu6)
    for conv in convs:
        for bn in bns:
            for add in adds:
                for relu in relus:
                    # conv bn add relu
                    register_fusion_pattern(
                        (relu, (add, (bn, conv), MatchAllNode))
                    )(ConvBNAddReLUFusion)
                    register_fusion_pattern(
                        (relu, (add, MatchAllNode, (bn, conv)))
                    )(ConvBNAddedReLUFusion)
                    # conv bn add
                    register_fusion_pattern((add, (bn, conv), MatchAllNode))(
                        ConvBNAddReLUFusion
                    )
                    register_fusion_pattern((add, MatchAllNode, (bn, conv)))(
                        ConvBNAddedReLUFusion
                    )
                    # conv bn relu
                    register_fusion_pattern((relu, (bn, conv)))(
                        ConvBNAddReLUFusion
                    )
                    # conv add relu
                    register_fusion_pattern((relu, (add, conv, MatchAllNode)))(
                        ConvBNAddReLUFusion
                    )
                    register_fusion_pattern((relu, (add, MatchAllNode, conv)))(
                        ConvBNAddedReLUFusion
                    )
                    # conv bn
                    register_fusion_pattern((bn, conv))(ConvBNAddReLUFusion)
                    # conv add
                    register_fusion_pattern((add, conv, MatchAllNode))(
                        ConvBNAddReLUFusion
                    )
                    register_fusion_pattern((add, MatchAllNode, conv))(
                        ConvBNAddedReLUFusion
                    )
                    # conv relu
                    register_fusion_pattern((relu, conv))(ConvBNAddReLUFusion)


register_conv_bn_add_relu_fusion_patterns()
