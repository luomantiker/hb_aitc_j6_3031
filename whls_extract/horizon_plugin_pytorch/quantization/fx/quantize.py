import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule
from torch.fx.graph import Node

from horizon_plugin_pytorch.nn import quantized
from horizon_plugin_pytorch.nn.linear import Identity
from horizon_plugin_pytorch.nn.qat import DeQuantStub
from horizon_plugin_pytorch.quantization.quantize import (
    convert,
    propagate_qconfig_,
)
from horizon_plugin_pytorch.quantization.stubs import QuantStub
from ..hybrid_ops import (
    get_hbdk4_float_supported_functions,
    get_hbdk4_float_supported_methods,
    get_hbdk4_float_supported_modules,
    get_hbdk4_quantized_supported_functions,
    get_hbdk4_quantized_supported_methods,
    get_hbdk4_quantized_supported_modules,
    get_hybrid_qat_module_mappings,
    get_hybrid_quantized_module_mappings,
    get_hybrid_supported_functions,
    get_hybrid_supported_methods,
    get_hybrid_supported_modules,
)
from ..qconfig_template import QconfigSetterBase
from ..quantization_mappings import (
    get_qat_module_mappings,
    get_quantized_operator_mappings,
)
from .graph_module import (
    ObservedGraphModule,
    QuantizedGraphModule,
    is_observed_module,
)
from .graph_optimizers import QconfigCanonicalizer
from .pattern_utils import Pattern
from .qconfig_utils import (
    QConfigAny,
    convert_dict_to_ordered_dict,
    get_flattened_qconfig_dict,
)

logger = logging.getLogger(__name__)
# Define helper types
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QConfigAny]


class Quantizer:
    def __init__(self):
        self.prepare_custom_config_dict: Dict[str, Any] = {}

    def save_state(self, observed: GraphModule) -> None:
        pass

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(
            observed
        ), "incoming model must be produced by prepare_fx"
        pass

    def _qat_swap_modules(
        self,
        root: torch.nn.Module,
        hybrid: bool,
        opset_version: str = "hbdk3",
        swapable_names: List[str] = None,
    ) -> None:
        mapping = (
            get_hybrid_qat_module_mappings()
            if hybrid and opset_version == "hbdk3"
            else get_qat_module_mappings()
        )

        convert(
            root,
            mapping=mapping,
            inplace=True,
            remove_qconfig=False,
            swapable_names=swapable_names,
        )

    def _insert_dequant_and_act_fq_for_cpu_op(
        self,
        model: GraphModule,
        node_name_to_module: Dict[str, torch.nn.Module],
        node_name_to_qconfig: Dict[str, QConfigAny],
        node_name_to_hybrid: Dict[str, bool],
    ) -> None:
        nodes = list(model.graph.nodes)
        identity_type_set = {torch.nn.Identity, Identity}

        def _find_real_users(node, start_user=None):
            # Find non-identity users and starting users.
            # E.g. node -> identity 1 -> identity 2 -> non-identity 1
            # will return [(identity 1, non-identity 1)]
            real_users = []
            for user in node.users:
                replace_user = start_user if start_user else user
                if (
                    type(node_name_to_module.get(user.name))
                    in identity_type_set
                ):
                    real_users += _find_real_users(user, replace_user)
                else:
                    real_users.append((replace_user, user))
            return real_users

        def _find_real_providers(node):
            # Find non-identity providers and starting providers.
            # E.g. non-identity 1 -> identity 1 -> identity 2 -> node
            # will return [(identity 2, non-identity 1)]
            real_providers = []
            for input_node in node.all_input_nodes:
                real_provider = input_node
                while (
                    type(node_name_to_module.get(real_provider.name))
                    in identity_type_set
                ):
                    real_provider = real_provider.all_input_nodes[0]
                real_providers.append((input_node, real_provider))
            return real_providers

        def _should_insert_act_fq(node):
            # A node should insert act fq if it meets all these conditions:
            # 1. It has activation qconfig.
            # 2. It is a cpu node.
            # 3. It isn't Identity.
            # 4. It has at least one non-QuantStub bpu user node.
            if (
                not node_name_to_qconfig.get(node.name, None)
                or not node_name_to_qconfig[node.name].activation
                or not node_name_to_hybrid.get(node.name, False)
                or (
                    node.op == "call_module"
                    and type(node_name_to_module[node.name])
                    in identity_type_set
                )
            ):
                return False

            for _, real_user in _find_real_users(node):
                if not node_name_to_hybrid.get(real_user.name, True) and (
                    type(node_name_to_module.get(real_user.name, None))
                    not in {
                        QuantStub,
                        torch.quantization.QuantStub,
                        DeQuantStub,
                        torch.quantization.DeQuantStub,
                    }
                ):
                    return True

            return False

        def _should_insert_dequant(node):
            # A node should insert dequant if it meets all these conditions:
            # 1. It is a cpu node.
            # 2. It isn't Identity.
            # 3. It has at least one non-DeQuantStub bpu input node.
            if not node_name_to_hybrid.get(node.name, False) or (
                node.op == "call_module"
                and type(node_name_to_module[node.name]) in identity_type_set
            ):
                return False

            for _, real_provider in _find_real_providers(node):
                if not node_name_to_hybrid.get(real_provider.name, True) and (
                    type(node_name_to_module.get(real_provider.name, None))
                    not in {DeQuantStub, torch.quantization.DeQuantStub}
                ):
                    return True

            return False

        for node in nodes:
            if _should_insert_act_fq(node):
                # insert act fq after
                act_fq = node_name_to_qconfig[node.name].activation()
                act_fq_name = f"{node.name}_activation_post_process"
                model.add_module(act_fq_name, act_fq)
                with model.graph.inserting_after(node):
                    act_fq_node = model.graph.create_node(
                        "call_module", act_fq_name, (node,), {}
                    )
                for replace_user, real_user in _find_real_users(node):
                    if real_user is act_fq_node:
                        continue
                    if not node_name_to_hybrid.get(real_user.name, True) and (
                        type(node_name_to_module.get(real_user.name, None))
                        not in {
                            QuantStub,
                            torch.quantization.QuantStub,
                            DeQuantStub,
                            torch.quantization.DeQuantStub,
                        }
                    ):
                        replace_user.replace_input_with(node, act_fq_node)

            if _should_insert_dequant(node):
                # insert dequant before
                dequant = DeQuantStub()
                dequant_module_name = f"{node.name}_input_dequant"
                model.add_module(dequant_module_name, dequant)
                for idx, (replace_node, real_provider) in enumerate(
                    _find_real_providers(node)
                ):
                    if not node_name_to_hybrid.get(
                        real_provider.name, True
                    ) and (
                        type(node_name_to_module.get(real_provider.name, None))
                        not in {DeQuantStub, torch.quantization.DeQuantStub}
                    ):
                        dequant_node_name = (
                            f"{node.name}_input_dequant_{str(idx)}"
                        )
                        with model.graph.inserting_before(node):
                            dequant_node = model.graph.create_node(
                                "call_module",
                                dequant_module_name,
                                (replace_node,),
                                {},
                                dequant_node_name,
                            )
                        node.replace_input_with(replace_node, dequant_node)

    def _get_node_name_to_module(
        self,
        model: GraphModule,
    ) -> Dict[str, torch.nn.Module]:
        node_name_to_modules = {}
        for node in model.graph.nodes:
            if node.op == "call_module":
                node_name_to_modules[node.name] = model.get_submodule(
                    node.target
                )
        return node_name_to_modules

    def _get_node_name_to_qconfig(
        self,
        model: GraphModule,
        node_name_to_scope: Dict[str, str],
    ) -> Dict[str, QConfigAny]:
        node_name_to_qconfig = {}
        for node in model.graph.nodes:
            if node.op not in {"get_attr", "output", "placeholder"}:
                # nodes which cannot find in node_name_to_scope are
                # auto-inserted node after calibration, they are children
                # of the model
                node_path = node_name_to_scope.get(node.name, "")
                module = model.get_submodule(node_path)
                node_name_to_qconfig[node.name] = getattr(
                    module, "qconfig", None
                )
        return node_name_to_qconfig

    def _node_in_opset(
        self, node, node_name_to_module, dtype: str, opset_version: str
    ) -> bool:
        if opset_version == "hbdk4":
            if dtype == "quantized":
                supported_modules = get_hbdk4_quantized_supported_modules()
                supported_functions = get_hbdk4_quantized_supported_functions()
                supported_methods = get_hbdk4_quantized_supported_methods()
            else:
                supported_modules = get_hbdk4_float_supported_modules()
                supported_functions = get_hbdk4_float_supported_functions()
                supported_methods = get_hbdk4_float_supported_methods()
        else:
            if dtype == "quantized":
                supported_modules = get_hybrid_supported_modules()
                supported_functions = get_hybrid_supported_functions()
                supported_methods = get_hybrid_supported_methods()
            else:
                # assume all ops support float in hbdk3
                return True
        return (
            (
                node.op == "call_module"
                and type(node_name_to_module[node.name]) in supported_modules
            )
            or (
                node.op == "call_function"
                and node.target in supported_functions
            )
            or (node.op == "call_method" and node.target in supported_methods)
        )

    def _get_node_name_to_hybrid(
        self,
        model: GraphModule,
        node_name_to_module: Dict[str, torch.nn.Module],
        node_name_to_scope: Dict[str, str],
        hybrid_dict: Dict[str, List] = None,
        opset_version: str = "hbdk3",
    ) -> Dict[str, bool]:
        # true is CPU node while false is BPU node.
        node_name_to_hybrid = {}

        if hybrid_dict is None:
            hybrid_dict = {}
        hybrid_module_type = hybrid_dict.get("module_type", [])
        hybrid_module_name = hybrid_dict.get("module_name", [])
        for node in model.graph.nodes:
            if node.op not in {"get_attr", "output", "placeholder"}:
                # assume all op are bpu op.
                node_name_to_hybrid[node.name] = False
                # nodes which cannot find in node_name_to_scope are
                # auto-inserted node after calibration, they are children
                # of the model
                node_path = node_name_to_scope.get(node.name, "")

                ancestor_name = None
                ancestor_modules = [model]
                if len(node_path) > 0:
                    for i in node_path.split("."):
                        ancestor_name = (
                            f"{ancestor_name}.{i}" if ancestor_name else i
                        )
                        ancestor_modules.append(
                            model.get_submodule(ancestor_name)
                        )

                # User-specified cpu node
                for ancestor_module in ancestor_modules:
                    if type(ancestor_module) in hybrid_module_type:
                        node_name_to_hybrid[node.name] = True
                for prefix in hybrid_module_name:
                    if node_path.startswith(prefix):
                        node_name_to_hybrid[node.name] = True

                if not self._node_in_opset(
                    node,
                    node_name_to_module,
                    "quantized",
                    opset_version,
                ):
                    # not supported for quantized but supported for float,
                    # marked as hybrid (float) op.
                    if self._node_in_opset(
                        node,
                        node_name_to_module,
                        "float",
                        opset_version,
                    ):
                        node_name_to_hybrid[node.name] = True

                    # Neither does the op support float, do nothing.
                    else:
                        logger.warning(
                            f"{node.op} `{node.target}` does not have "
                            f"float implementation in {opset_version}! "
                            "It will not be marked as hybrid op. "
                            "If it is user-defined module/function wrapped by "
                            "`fx_helper.wrap()`, please be aware that torch.fx"
                            " would consider it as leaf node and won't parse "
                            "its internal implementation."
                        )
        return node_name_to_hybrid

    def _remove_hybrid_qconfig(
        self,
        model: GraphModule,
        node_name_to_module: Dict[str, torch.nn.Module],
        node_name_to_scope: Dict[str, str],
        node_name_to_hybrid: Dict[str, bool],
    ):
        for node in model.graph.nodes:
            # remove qconfig of unsupported functional
            if node.op == "call_method" and node_name_to_hybrid.get(
                node.name, False
            ):
                for input_node in node.all_input_nodes:
                    if input_node.op == "get_attr":
                        try:
                            module = model.get_submodule(input_node.target)
                            if isinstance(
                                module,
                                (
                                    torch.nn.quantized.FloatFunctional,
                                    quantized.FloatFunctional,
                                ),
                            ):
                                module.qconfig = None
                        except AttributeError:
                            pass

            # remove qconfig of user specified cpu module
            if node.op == "call_module" and node_name_to_hybrid.get(
                node.name, False
            ):

                def set_qconfig_to_none(module):
                    for child in module.children():
                        set_qconfig_to_none(child)
                    if hasattr(module, "qconfig"):
                        module.qconfig = None

                set_qconfig_to_none(node_name_to_module[node.name])

    def _prepare(
        self,
        model: GraphModule,
        qconfig_dict: Dict[str, Any],
        prepare_custom_config_dict: Optional[Dict[str, Any]],
        check_qconfig: callable,
        optimize_graph,
        hybrid: bool = False,
        hybrid_dict: Dict[str, List] = None,
        opset_version: str = "hbdk3",
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
        qconfig_setter: Optional[Tuple[QconfigSetterBase, ...]] = None,
        only_swap_used_mods: bool = False,
    ) -> ObservedGraphModule:
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        if qconfig_dict is None:
            qconfig_dict = {}
        self.prepare_custom_config_dict = prepare_custom_config_dict

        convert_dict_to_ordered_dict(qconfig_dict)
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)

        if only_swap_used_mods:
            swapable_names = []
            for n in model.graph.nodes:
                if n.op in ("call_module", "get_attr"):
                    swapable_names.append(n.target)
        else:
            swapable_names = None

        # set qconfig for modules
        propagate_qconfig_(
            model, flattened_qconfig_dict, white_list=swapable_names
        )

        if qconfig_setter is not None:
            for setter in qconfig_setter:
                setter.set_qconfig(model, example_inputs, example_kw_inputs)

        QconfigCanonicalizer()(model, qconfig_setter)

        if check_qconfig is not None:
            check_qconfig(model)

        self._qat_swap_modules(model, hybrid, opset_version)
        self.save_state(model)
        preserved_attributes = prepare_custom_config_dict.get(
            "preserved_attributes", []
        )

        if isinstance(model, GraphModule):
            model = ObservedGraphModule(
                model, model.graph, preserved_attributes
            )

        return model

    def prepare(
        self,
        model: GraphModule,
        qconfig_dict: Dict[str, Any] = None,
        prepare_custom_config_dict: Dict[str, Any] = None,
        check_qconfig: callable = None,
        optimize_graph: bool = False,
        hybrid: bool = False,
        hybrid_dict: Dict[str, List] = None,
        opset_version: str = "hbdk3",
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
        qconfig_setter: Optional[Tuple[QconfigSetterBase, ...]] = None,
        only_swap_used_mods: bool = False,
    ) -> ObservedGraphModule:
        return self._prepare(
            model,
            qconfig_dict,
            prepare_custom_config_dict,
            check_qconfig,
            optimize_graph,
            hybrid,
            hybrid_dict,
            opset_version,
            example_inputs,
            example_kw_inputs,
            qconfig_setter,
            only_swap_used_mods,
        )

    def _convert_swap_modules(
        self,
        root: torch.nn.Module,
        remove_qconfig,
        inplace: bool = False,
        hybrid: bool = False,
    ) -> None:
        mapping = (
            get_hybrid_quantized_module_mappings()
            if hybrid
            else get_quantized_operator_mappings()
        )
        return convert(
            root,
            mapping=mapping,
            inplace=inplace,
            remove_qconfig=remove_qconfig,
        )

    def _convert(
        self,
        model: GraphModule,
        inplace: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True,
        hybrid: bool = False,
    ) -> QuantizedGraphModule:
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        if isinstance(model, GraphModule):
            self.restore_state(model)

        quantized_model = self._convert_swap_modules(
            model, _remove_qconfig, inplace, hybrid=hybrid
        )

        if isinstance(model, GraphModule):
            self.save_state(quantized_model)
            preserved_attributes = convert_custom_config_dict.get(
                "preserved_attributes", []
            )
            quantized_model = QuantizedGraphModule(
                quantized_model, quantized_model.graph, preserved_attributes
            )

        return quantized_model

    def convert(
        self,
        model: GraphModule,
        inplace: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True,
        hybrid: bool = False,
    ) -> QuantizedGraphModule:
        opset_version = getattr(model, "_opset_version", "hbdk3")
        if opset_version == "hbdk4":
            raise RuntimeError(
                'Opset version is "hbdk4", '
                "`convert_fx` is not supposed to be used."
            )
        quantized = self._convert(
            model,
            inplace,
            convert_custom_config_dict,
            _remove_qconfig=_remove_qconfig,
            hybrid=hybrid,
        )

        return quantized
