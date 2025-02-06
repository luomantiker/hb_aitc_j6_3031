import inspect
import logging
import math
import re
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Union

import torch
import torch.nn.intrinsic as nni
from tabulate import tabulate
from torch import fx, nn
from torch.fx import GraphModule, Node
from torch.quantization import DeQuantStub
from torch.utils._pytree import tree_flatten

from horizon_plugin_pytorch.dtype import QuantDType, qint8, qint16
from horizon_plugin_pytorch.fx import HookAndTorchFunctionTracer
from horizon_plugin_pytorch.fx.jit_scheme import TensorInfo
from horizon_plugin_pytorch.nn import GridSample, Identity
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn import intrinsic, quantized
from horizon_plugin_pytorch.nn.grid_sample import warp
from horizon_plugin_pytorch.quantization import FakeCast
from horizon_plugin_pytorch.quantization.qconfig import (
    QConfig,
    default_calib_8bit_fake_quant_qconfig,
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    default_qat_8bit_fake_quant_qconfig,
    default_qat_8bit_fixed_act_fake_quant_qconfig,
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
    default_qat_8bit_weight_16bit_fixed_act_fake_quant_qconfig,
)
from horizon_plugin_pytorch.quantization.stubs import QuantStub
from horizon_plugin_pytorch.utils.model_helper import (
    find_leaf_modules,
    get_model_training_state,
    is_ff_node,
    is_fused_ff_node,
    set_model_training_state,
)
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)


class SpecialNodeValue(Enum):
    # all node
    ALL_NODE = "all"
    # input node
    PLACEHOLDER = "placeholder"
    # gemm
    GEMM = "gemm"


GEMM_MODULE = (
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nni.LinearReLU,
    intrinsic.ConvReLU2d,
    intrinsic.ConvReLU62d,
    intrinsic.ConvReLU63d,
    intrinsic.ConvAdd2d,
    intrinsic.ConvAdd3d,
    intrinsic.ConvAddReLU2d,
    intrinsic.ConvAddReLU3d,
    intrinsic.ConvAddReLU62d,
    intrinsic.ConvAddReLU63d,
    intrinsic.ConvBN2d,
    intrinsic.ConvBNAdd2d,
    intrinsic.ConvBNAddReLU2d,
    intrinsic.ConvBNReLU2d,
    intrinsic.ConvBNReLU62d,
    intrinsic.ConvBNAddReLU62d,
    intrinsic.DeformConvAdd2d,
    intrinsic.DeformConvAddReLU2d,
    intrinsic.DeformConvAddReLU62d,
    intrinsic.DeformConvReLU2d,
    intrinsic.DeformConvReLU62d,
    intrinsic.LinearAdd,
    intrinsic.LinearReLU,
    intrinsic.LinearReLU6,
    intrinsic.LinearAddReLU,
    intrinsic.LinearAddReLU6,
)


def is_gemm_node(node):
    """Judge if a fx node is gemm node."""
    if node.op == "call_module":
        module = node.graph.owning_module.get_submodule(node.target)
        if isinstance(module, GEMM_MODULE):
            return True
        if isinstance(module, quantized.FloatFunctional):
            return module._last_called_method_name == "matmul"
    if (
        node.op == "call_method"
        and len(node.args) > 0
        and isinstance(node.args[0], Node)
        and node.args[0].op == "get_attr"
    ):
        try:
            module = node.graph.owning_module.get_submodule(
                node.args[0].target
            )
            if isinstance(module, quantized.FloatFunctional):
                return node.target == "matmul"
            else:
                return isinstance(module, GEMM_MODULE)
        except AttributeError:
            pass

    return False


def is_dequant_node(node):
    """Judge if a fx node is dequant node."""
    if node.op == "call_module":
        module = node.graph.owning_module.get_submodule(node.target)
        if isinstance(module, DeQuantStub):
            return True
    return False


def is_node_only_used_by_dequant_node(node):
    """Judge if a fx node is only used by dequant node."""
    for user in node.users:
        if isinstance(user, (nn.Identity, Identity)):
            if not is_node_only_used_by_dequant_node(user):
                return False
        elif not is_dequant_node(user):
            return False
    return True if len(node.users) > 0 else False


class PatternNode:
    """Template pattern node."""

    def __init__(self, value, pre=None, next=None):
        self.value = value
        self.pre = pre if pre else {}
        self.next = next if next else {}

    def _add_pre(self, node):
        """Add node to previous node set."""
        self.pre[node] = None

    def _add_next(self, node):
        """Add node to next node set."""
        self.next[node] = None

    @typechecked
    def link_node(self, node: "PatternNode") -> None:
        """
        Link node after current node.

        Args:
            node (PatternNode): The node for linking.
        """
        self._add_next(node)
        node._add_pre(self)

    def __repr__(self):
        return f"node_{id(self)}_value: {self.value}"


class PatternGraph:
    """Template pattern graph."""

    def __init__(self, entry_nodes=None, nodes=None):
        self.entry_nodes = entry_nodes if entry_nodes else {}
        self.nodes = nodes if nodes else {}

    def __repr__(self):
        node_specs = [
            [
                id(n),
                n in self.entry_nodes,
                n.value,
                [id(p) for p in n.pre],
                [id(p) for p in n.next],
            ]
            for n in self.nodes
        ]
        return tabulate(
            node_specs,
            headers=["node_id", "is_entry", "value", "pre", "next"],
        )

    def _add_node(self, node):
        """Add node to the graph node list."""
        if node not in self.nodes:
            self.nodes[node] = None

    def _add_entry_node(self, node):
        """Add entry node to the graph entry node list."""
        if node not in self.entry_nodes:
            self.entry_nodes[node] = None

    def _remove_entry_node(self, node):
        """Remove entry node of the graph."""
        if node in self.entry_nodes:
            del self.entry_nodes[node]

    @typechecked
    def add_node(self, node: PatternNode) -> None:
        """
        Add node to the graph.

        Args:
            node (PatternNode): node to be added.
        """
        self._add_node(node)
        self._add_entry_node(node)

    @typechecked
    def add_edge(self, node1: PatternNode, node2: PatternNode) -> None:
        """
        Add edge to the graph.

        Args:
            node1 (PatternNode): Start node of the edge.
            node2 (PatternNode): End node of the edge.
        """
        node1.link_node(node2)
        if node1 not in self.nodes:
            self._add_entry_node(node1)
        self._add_node(node1)
        if node2 in self.entry_nodes:
            self._remove_entry_node(node2)
        self._add_node(node2)

    @staticmethod
    def _is_node_pattern_match(node, pattern_node):
        """If fx node match pattern node."""

        def is_node_pattern_value_match(value):
            if value == SpecialNodeValue.ALL_NODE:
                return True
            if value == SpecialNodeValue.GEMM:
                return is_gemm_node(node)
            if node.op == "placeholder":
                return value == SpecialNodeValue.PLACEHOLDER
            if node.op == "call_module" and type(value) is type:
                return isinstance(
                    node.graph.owning_module.get_submodule(node.target),
                    value,
                )
            if node.op == "call_function":
                return value == node.target

            return False

        if isinstance(pattern_node.value, list):
            for value in pattern_node.value:
                if is_node_pattern_value_match(value):
                    return True
            return False
        else:
            return is_node_pattern_value_match(pattern_node.value)

    @staticmethod
    def _find_subgraphs(node, pattern_node, subgraph_nodes=None):
        """Find matched subgraphs, start from the node."""
        if subgraph_nodes is None:
            subgraph_nodes = []

        subgraph_nodes.append(node)

        if len(pattern_node.next) == 0:
            return [subgraph_nodes]

        ret = []
        for pattern_next_node in pattern_node.next:
            for next_node in node.users:
                if PatternGraph._is_node_pattern_match(
                    next_node, pattern_next_node
                ):
                    subgraphs = PatternGraph._find_subgraphs(
                        next_node,
                        pattern_next_node,
                        list(subgraph_nodes),
                    )
                    if len(subgraphs) > 0:
                        ret += subgraphs
        return ret

    @typechecked
    def pattern_match(self, graph: fx.Graph) -> List[List[fx.node.Node]]:
        """
        Find subgraphs which match the pattern.

        Args:
            graph (Graph): Graph to match.

        Returns:
            List[List[fx.node.Node]]: matched subgraphs.
        """
        # currently, we only have single line pattern match. Complex isomorphic
        # graph matching problems are not considered because this will slow
        # down the matching efficiency.
        # check to ensure pattern is single line pattern.
        for node in self.nodes:
            if len(node.next) > 1:
                msg = "Only single line pattern is supported now"
                logger.error(msg)
                raise NotImplementedError(msg)

        logger.debug(f"Finding matches of {self}...")

        pattern_entry_node = list(self.entry_nodes)[0]
        matches = []
        for node in graph.nodes:
            if PatternGraph._is_node_pattern_match(node, pattern_entry_node):
                subgraphs = PatternGraph._find_subgraphs(
                    node, pattern_entry_node
                )
                if len(subgraphs) > 0:
                    matches += subgraphs

        logger.debug(f"Find {len(matches)} matches of {self}.")
        return matches


class QconfigTemplateBase:
    """Base qconfig template."""

    def __init__(self, mode=None):
        self.mode = mode
        self.pattern = self.pattern()

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        msg = "pattern of template is not implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        msg = "run_transform of template is not implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        msg = "set_default_qconfig of template is not implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)

    @staticmethod
    def _set_qconfig_for_module_and_submodule(module_name, module, qconfig):
        """Set qconfig for module and submodule."""

        if not hasattr(module, "qconfig"):
            module.qconfig = qconfig
            if hasattr(module, "propagate_qconfig"):
                module.propagate_qconfig(qconfig)
        else:
            logger.debug(f"Qconfig of {module_name} has been set before.")

        for child_name, child in module.named_children():
            QconfigTemplateBase._set_qconfig_for_module_and_submodule(
                f"{module_name}.{child_name}", child, qconfig
            )

    @staticmethod
    def _get_module_called_by_node(node):
        """Get module called by node."""
        if node.op in ("call_module", "get_attr"):
            mod_name = node.target
        elif (
            node.op == "call_method"
            and isinstance(node.args[0], Node)
            and node.args[0].op == "get_attr"
        ):
            mod_name = node.args[0].target
        else:
            mod_name = None

        try:
            module = node.graph.owning_module.get_submodule(mod_name)
        except AttributeError:
            module = None

        return mod_name, module

    @staticmethod
    def _set_qconfig_for_module_called_by_node(nodes, qconfig):
        """Set qconfig for module called by node."""
        for node in nodes:
            mod_name, module = QconfigTemplateBase._get_module_called_by_node(
                node
            )

            if module is None or mod_name is None:
                continue

            QconfigTemplateBase._set_qconfig_for_module_and_submodule(
                mod_name, module, qconfig
            )


class DefaultTemplate(QconfigTemplateBase):
    """Template to set default qconfig for unset module."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode(SpecialNodeValue.ALL_NODE)
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = qconfig

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."
        self._set_qconfig_for_module_called_by_node(subgraph, self.qconfig)


class AllInt8Template(DefaultTemplate):
    """Set the qconfig of all modules to int8."""

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint8),
            weight=qconfig.weight,
        )


class AllInt16Template(DefaultTemplate):
    """Set the qconfig of all modules to int16."""

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint16),
            weight=qconfig.weight,
        )


class HighPrecisionOutputTemplate(QconfigTemplateBase):
    """Set the qconfig of output nodes to high precision."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode(DeQuantStub)
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=None,
            weight=qconfig.weight,
        )

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."

        def set_qconfig_for_high_precision_output_module(node):
            for input_node in node.all_input_nodes:
                if is_gemm_node(
                    input_node
                ) and is_node_only_used_by_dequant_node(input_node):
                    logger.info(
                        f"The qconfig of {input_node.target} will be set to "
                        f"default_qat_8bit_weight_32bit_out_fake_quant_qconfig"
                    )
                    self._set_qconfig_for_module_called_by_node(
                        [input_node],
                        self.qconfig,
                    )
                elif input_node.op == "call_module":
                    module = input_node.graph.owning_module.get_submodule(
                        input_node.target
                    )
                    if isinstance(module, (nn.Identity, Identity)):
                        set_qconfig_for_high_precision_output_module(
                            input_node
                        )
                else:
                    logger.info(
                        f"{input_node.name} doesn't support high precision"
                        "output."
                    )

        set_qconfig_for_high_precision_output_module(subgraph[0])


class GEMMInt8Template(QconfigTemplateBase):
    """Set the qconfig of gemm nodes to int8."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode(SpecialNodeValue.GEMM)
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint8),
            weight=qconfig.weight,
        )

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."
        self._set_qconfig_for_module_called_by_node(subgraph, self.qconfig)


class GEMMToLayerNormInt16Template(QconfigTemplateBase):
    """Set the qconfig of modules from gemm to layernorm to int16."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode(LayerNorm2d)
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint16),
            weight=qconfig.weight,
        )

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."

        cur_node = subgraph[0]
        if cur_node.op == "call_module":
            self._set_qconfig_for_module_called_by_node(
                [cur_node],
                self.qconfig,
            )
            if is_gemm_node(cur_node):
                return

        for node in cur_node.all_input_nodes:
            self.run_transform([node])


class GEMMToSoftmaxInt16Template(QconfigTemplateBase):
    """Set the qconfig of modules from gemm to softmax to int16."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode(nn.Softmax)
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint16),
            weight=qconfig.weight,
        )

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."

        cur_node = subgraph[0]
        if cur_node.op == "call_module":
            self._set_qconfig_for_module_called_by_node(
                [cur_node],
                self.qconfig,
            )
            if is_gemm_node(cur_node):
                return

        for node in cur_node.all_input_nodes:
            self.run_transform([node])


class GEMMToGridSampleGridInt16Template(QconfigTemplateBase):
    """Set the qconfig of modules from gemm to gridsample grid to int16."""

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        node = PatternNode([GridSample, nn.functional.grid_sample, warp])
        pattern_graph = PatternGraph()
        pattern_graph.add_node(node)
        return pattern_graph

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = QConfig(
            activation=qconfig.activation.with_args(dtype=qint16),
            weight=qconfig.weight,
        )

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."

        grid_input_node = subgraph[0].args[1]
        vis_node = {grid_input_node.name}

        def set_qconfig_for_grid_related_node(node):
            self._set_qconfig_for_module_called_by_node(
                [node],
                self.qconfig,
            )
            if is_gemm_node(node):
                return
            # single arg of some ops may be a list. e.g. cat
            for arg in tree_flatten(node.args)[0]:
                if isinstance(arg, fx.node.Node) and arg.name not in vis_node:
                    vis_node.add(arg.name)
                    set_qconfig_for_grid_related_node(arg)

        set_qconfig_for_grid_related_node(grid_input_node)


class ConstTensorQuantTemplate(QconfigTemplateBase):
    """Quant const tensor automatically."""

    class ConstTensorInputPatternGraph(PatternGraph):
        """Pattern graph for const tensor input."""

        @typechecked
        def pattern_match(self, graph: fx.Graph) -> List[List[fx.node.Node]]:
            """
            Find nodes which have constant tensor input.

            Args:
                graph (Graph): Graph to match.

            Returns:
                List[List[fx.node.Node]]: matched subgraphs.
            """
            matches = []
            for node in graph.nodes:
                flatten_args = tree_flatten((node.args, node.kwargs))[0]
                (
                    mod_name,
                    module,
                ) = QconfigTemplateBase._get_module_called_by_node(node)
                if (
                    mod_name is not None
                    and module is not None
                    and any(
                        isinstance(arg, TensorInfo)
                        and arg.dtype == torch.float
                        for arg in flatten_args
                    )
                    and not isinstance(
                        module, (QuantStub, torch.quantization.QuantStub)
                    )
                ):
                    matches.append([node])
            return matches

    @staticmethod
    def pattern():
        """Pattern of qconfig template."""
        return ConstTensorQuantTemplate.ConstTensorInputPatternGraph()

    def set_default_qconfig(self, qconfig):
        """Set default qconfig."""
        self.qconfig = qconfig

    def run_transform(self, subgraph):
        """Run transform of qconfig template."""
        assert hasattr(
            self, "qconfig"
        ), "call set_default_qconfig before run_transform."

        node = subgraph[0]
        mod_name, module = self._get_module_called_by_node(node)

        if (
            hasattr(module, "qconfig")
            and hasattr(module.qconfig, "input")
            and module.qconfig.input is not None
        ):
            logger.warning(
                f"input of {mod_name} contains const tensor without "
                f"quantstub, please check if quantstub is needed."
            )
            return

        def get_qconfig_output(qconfig):
            if qconfig is None:
                return None
            if qconfig.output is not None:
                return qconfig.output
            return qconfig.activation

        if hasattr(module, "qconfig"):
            single_input_qconfig = get_qconfig_output(module.qconfig)
        else:
            single_input_qconfig = get_qconfig_output(self.qconfig)

        is_ff_or_fused_ff = is_ff_node(
            node, node.graph.owning_module
        ) or is_fused_ff_node(node, node.graph.owning_module)

        def get_kwarg_idx(kw):
            if is_ff_node(node, node.graph.owning_module):
                func = getattr(module, node.target, None)
                sig = (
                    [] if func is None else inspect.signature(func).parameters
                )
            else:
                sig = inspect.signature(module.forward).parameters

            return list(sig.keys()).index(kw)

        len_args = len(node.args) - 1 if is_ff_or_fused_ff else len(node.args)
        len_kwargs = len(node.kwargs)

        qconfig_input = [None for _ in range(len_args + len_kwargs)]
        for idx, arg in enumerate(node.args):
            if isinstance(arg, TensorInfo) and arg.dtype == torch.float:
                if is_ff_or_fused_ff:
                    idx = idx - 1
                logger.info(
                    f"input {idx} of {node.name} is a const tensor without "
                    f"quantstub, will set input qconfig to insert quantstub "
                    f"automatically."
                )
                qconfig_input[idx] = single_input_qconfig
        for kw, arg in enumerate(node.kwargs):
            if isinstance(arg, TensorInfo) and arg.dtype == torch.float:
                logger.info(
                    f"input {kw} of {node.name} is a const tensor without "
                    f"quantstub, will set input qconfig to insert quantstub "
                    f"automatically."
                )
                qconfig_input[
                    len_args + get_kwarg_idx(kw)
                ] = single_input_qconfig

        if hasattr(module, "qconfig"):
            qconfig = QConfig(
                activation=get_qconfig_output(module.qconfig),
                weight=module.qconfig.weight,
                input=qconfig_input,
            )
        else:
            qconfig = QConfig(
                activation=get_qconfig_output(self.qconfig),
                weight=self.qconfig.weight,
                input=qconfig_input,
            )
        module.qconfig = qconfig


class QconfigSetterBase:
    def set_qconfig(
        self, model, example_inputs: Any = None, example_kw_inputs: Any = None
    ):
        msg = "set_qconfig of QconfigSetter is not implemented!"
        logger.error(msg)
        raise NotImplementedError(msg)


class TemplateQconfigSetter(QconfigSetterBase):
    def __init__(self, default_qconfig, templates):
        self.default_qconfig = default_qconfig
        self.templates = templates
        # add default qconfig template
        self.templates.append(DefaultTemplate())
        for template in self.templates:
            template.set_default_qconfig(default_qconfig)

    @typechecked
    def set_qconfig(
        self,
        model: nn.Module,
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
    ) -> None:
        """
        Set template qconfig.

        Args:
            model (nn.Module): Model to set template qconfig.
            example_inputs (Any): Inputs for tracing graph.
            example_kw_inputs (Any): Keyword inputs for tracing graph.
        """
        logger.debug("Setting template qconfig...")
        if example_inputs is None and example_kw_inputs is None:
            raise ValueError(
                "example_inputs or example_kw_inputs must be provided "
                "to use {}".format(self.__class__.__name__)
            )

        # trace model in eval mode to focus on deploy graph.
        state = get_model_training_state(model)
        model.eval()
        from horizon_plugin_pytorch.fx.jit_scheme import (
            GraphModule as JitGraphModule,
        )

        if isinstance(model, (GraphModule, JitGraphModule)):
            graph = model.graph
        else:
            tracer = HookAndTorchFunctionTracer()
            graph = tracer.trace(model, example_inputs, example_kw_inputs)
        set_model_training_state(model, state)

        for template in self.templates:
            subgraphs = template.pattern.pattern_match(graph)
            for subgraph in subgraphs:
                template.run_transform(subgraph)

        logger.info("Template qconfig has been set!")


class ModuleNameQconfigSetter(QconfigSetterBase):
    """Module name qconfig setter.

    Set Qconfig by module name.

    Args:
        module_name_to_qconfig: mapping for module name and their qconfig.
    """

    def __init__(
        self,
        module_name_to_qconfig: Dict[str, QConfig],
    ):
        self.module_name_to_qconfig = module_name_to_qconfig

    @typechecked
    def set_qconfig(
        self,
        model: nn.Module,
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
    ):
        float_leaf_modules = find_leaf_modules(model)
        for mod_name, qconfig in self.module_name_to_qconfig.items():
            try:
                mod = model.get_submodule(mod_name)
            except AttributeError:
                # submodule in qat module
                names = mod_name.split(".")
                op_path = None
                for i in range(-1, -len(names), -1):
                    if ".".join(names[:i]) in float_leaf_modules:
                        op_path = ".".join(names[:i])
                        break
                if op_path is None:
                    logger.warning(
                        f"Can not find module {mod_name} in the model, "
                        + "skip qconfig set."
                    )
                    continue
                mod = model.get_submodule(op_path)
                mod_name = op_path
            QconfigTemplateBase._set_qconfig_for_module_and_submodule(
                mod_name, mod, qconfig
            )


def get_sensitive_op_qconfig_setter(
    sensitive_type: str,
    origin_dtype: Union[QuantDType, torch.dtype],
    default_qconfig: QConfig,
    sensitive_table: List,
    topk: int = None,
    ratio: float = None,
):
    """Set qconfig of sensitive ops obtained by QuantAnalysis.sensitivity.

    Args:
        sensitive_type: sensitive_type in sensitive.
            value should be "activation", "weight".
        origin_dtype: origin quant dtype in model.
        default_qconfig: custom default qconfig.
        sensitive_table: QuantAnalysis.sensitivity results
        topk: topk ops to be set custom qconfig. If set, ratio must be None.
        ratio: top ratio ops to be set custom qconfig. If set, topk must be
            None.

    Returns:
        ModuleNameQconfigSetter

    Examples:
    .. code-block:: python

        # get sensitive results from QuantAnalysis
        qa = QuantAnalysis(float_net, qat_net, "fake_quant", out_dir)
        table = qa.sensitive()

        # set activation qint16 and weight qint16 in top 20% sensitive ops
        activation_setter = get_sensitive_op_qconfig_setter(
            "activation",
            8bit_weight_16bit_act_qconfig,
            table,
            topk=None,
            ratio=0.2,
        )
        weight_setter = get_sensitive_op_qconfig_setter(
            "weight",
            16bit_weight_8bit_act_qconfig,
            table,
            topk=None,
            ratio=0.2,
        )
    """
    assert (topk or ratio) and (
        topk is None or ratio is None
    ), "Only one of topk or ratio can be set."
    if topk is not None:
        assert topk > 0, "Topk value must be positive."
    if ratio is not None:
        assert ratio > 0 and ratio <= 1, "Ratio must be in (0, 1]."
        topk = math.ceil(len(sensitive_table) * ratio)

    assert sensitive_type in (
        "activation",
        "weight",
    ), "sensitive_type must be activation/weight."

    # list of [op_name, sensitive_type, op_type, metric, dtype]
    sensitive_table = sensitive_table[:topk]

    assert len(sensitive_table) > 0, "sensitive_table can't be empty"

    if len(sensitive_table[0]) == 4:
        # old version sensitive_table
        ops = [x[0] for x in sensitive_table if x[1] == sensitive_type]
        logger.info(
            f"{len(ops)} ops will be set custom {sensitive_type} qconfig "
            f"in top {topk} ops:"
        )
    elif len(sensitive_table[0]) == 5:
        # new version sensitive_table
        ops = [
            x[0]
            for x in sensitive_table
            if x[1] == sensitive_type and x[-1] == origin_dtype
        ]
        logger.info(
            f"{len(ops)} ops will be set custom {sensitive_type} qconfig "
            f"in top {topk} ops:"
        )
    elif len(sensitive_table[0]) == 6:
        sensitive_table = [
            x
            for x in sensitive_table
            if x[1] == sensitive_type and x[4] == origin_dtype
        ]
        ops = [x[0] for x in sensitive_table]

        flops = [
            re.match(r"(\d+)\(([0-9.]+)%\)", x[-1]) for x in sensitive_table
        ]
        total_flops = sum([int(x.group(1)) for x in flops])
        total_ratio = sum([float(x.group(2)) for x in flops])

        logger.info(
            f"{len(ops)} ops with {total_flops}({total_ratio:.2f}%) flops will"
            f" be set custom {sensitive_type} qconfig in top {topk} ops:"
        )
    else:
        raise ValueError("illegal sensitivity table!")

    logger.info("\n".join(ops))

    module_name_to_qconfig = {op: default_qconfig for op in ops}

    return ModuleNameQconfigSetter(module_name_to_qconfig)


# activation qint16 weight qint8
sensitive_op_qat_8bit_weight_16bit_act_qconfig_setter = partial(
    get_sensitive_op_qconfig_setter,
    "activation",
    qint8,
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
)

sensitive_op_qat_8bit_weight_16bit_fixed_act_qconfig_setter = partial(
    get_sensitive_op_qconfig_setter,
    "activation",
    qint8,
    default_qat_8bit_weight_16bit_fixed_act_fake_quant_qconfig,
)

sensitive_op_calibration_8bit_weight_16bit_act_qconfig_setter = partial(
    get_sensitive_op_qconfig_setter,
    "activation",
    qint8,
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
)

# 1. high precision output.
# 2. grid sample grid int16.
# 3. other int8.
default_qat_qconfig_setter = TemplateQconfigSetter(
    default_qat_8bit_fake_quant_qconfig,
    [
        HighPrecisionOutputTemplate(),
        GEMMToGridSampleGridInt16Template(),
        ConstTensorQuantTemplate(),
    ],
)

default_qat_fixed_act_qconfig_setter = TemplateQconfigSetter(
    default_qat_8bit_fixed_act_fake_quant_qconfig,
    [
        HighPrecisionOutputTemplate(),
        GEMMToGridSampleGridInt16Template(),
        ConstTensorQuantTemplate(),
    ],
)

default_calibration_qconfig_setter = TemplateQconfigSetter(
    default_calib_8bit_fake_quant_qconfig,
    [
        HighPrecisionOutputTemplate(),
        GEMMToGridSampleGridInt16Template(),
        ConstTensorQuantTemplate(),
    ],
)

# 1. high precision output.
# 2. other int16.
qat_8bit_weight_16bit_act_qconfig_setter = TemplateQconfigSetter(
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
    [HighPrecisionOutputTemplate(), ConstTensorQuantTemplate()],
)

qat_8bit_weight_16bit_fixed_act_qconfig_setter = TemplateQconfigSetter(
    default_qat_8bit_weight_16bit_fixed_act_fake_quant_qconfig,
    [HighPrecisionOutputTemplate(), ConstTensorQuantTemplate()],
)

calibration_8bit_weight_16bit_act_qconfig_setter = TemplateQconfigSetter(
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    [HighPrecisionOutputTemplate(), ConstTensorQuantTemplate()],
)

all_fp16_qconfig_setter = TemplateQconfigSetter(
    QConfig(
        FakeCast.with_args(dtype=torch.float16),
        FakeCast.with_args(dtype=torch.float16),
        FakeCast.with_args(dtype=torch.float16),
    ),
    [],
)

# 1. high precision output.
# 2. gemm int8.
# 3. other int16.
gemm_int8_qat_qconfig_setter = TemplateQconfigSetter(
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
    [
        HighPrecisionOutputTemplate(),
        GEMMInt8Template(),
    ],
)

gemm_int8_calibration_qconfig_setter = TemplateQconfigSetter(
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    [
        HighPrecisionOutputTemplate(),
        GEMMInt8Template(),
    ],
)
