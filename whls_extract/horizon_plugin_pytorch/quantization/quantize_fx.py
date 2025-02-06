import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule

from horizon_plugin_pytorch.fx.jit_scheme import FunctionReplacer
from horizon_plugin_pytorch.fx.jit_scheme import GraphModule as JitGraphModule
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from horizon_plugin_pytorch.fx.tracer import CustomTracer
from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
    DispatchedTensorWrapper,
)
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    swap_nn_with_horizonnn,
)
from horizon_plugin_pytorch.utils.check_model import check_qat_model
from horizon_plugin_pytorch.utils.model_helper import ModelState, _as_tuple
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .fx import Fuser, Quantizer
from .fx.graph_module import (
    GraphModuleWithAttr,
    ObservedGraphModule,
    QuantizedGraphModule,
)
from .fx.utils import replace_function_with_module
from .qconfig_template import (
    ModuleNameQconfigSetter,
    QconfigSetterBase,
    TemplateQconfigSetter,
)
from .quantization_mappings import get_qat_module_mappings
from .quantize import propagate_attr_

logger = logging.getLogger(__name__)


def _check_is_graph_module(model: torch.nn.Module) -> None:
    if not isinstance(model, (GraphModule, JitGraphModule)):
        raise ValueError(
            "input model must be a GraphModule, "
            + "Got type:"
            + str(type(model))
            + " Please make "
            + "sure to follow the tutorials."
        )


def _fuse_fx(
    graph_module: Union[GraphModule, JitGraphModule],
    fuse_custom_config_dict: Optional[Dict[str, Any]] = None,
    allow_recompile: bool = True,
) -> Union[GraphModule, JitGraphModule]:
    r"""Fuse fx.

    Internal helper function to fuse modules inpreparation for quantization

    Args:
        graph_module:
            GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(graph_module)

    fuser = Fuser()
    if allow_recompile:
        return fuser.fuse(graph_module, fuse_custom_config_dict)
    else:
        return fuser.fuse_with_ori_graph(graph_module, fuse_custom_config_dict)


class Scope(object):
    """Refine this docstring in the future.

    Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example:
    class Sub(torch.nn.Module):
        def forward(self, x):
            # This will be a call_method Node in GraphModule,
            # scope for this would be (module_path="sub", module_type=Sub)
            return x.transpose(1, 2)

    class M(torch.nn.Module):
        def __init__(self):
            self.sub = Sub()

        def forward(self, x):
            # This will be a call_method Node as well,
            # scope for this would be (module_path="", None)
            x = x.transpose(1, 2)
            x = self.sub(x)
            return x

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


class ScopeContextManager(object):
    """Refine this docstring in the future.

    A context manager to track the Scope of Node during symbolic
    tracing.
    When entering a forward function of a Module, we'll update the scope
    information of the current module, and when we exit, we'll restore
    the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_module: torch.nn.Module,
        current_module_path: str,
    ):
        super().__init__()
        self.prev_module_type = scope.module_type
        self.prev_module_path = scope.module_path
        self.scope = scope
        self.scope.module_path = current_module_path
        self.scope.module_type = type(current_module)

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


class QuantizationTracer(CustomTracer):
    def __init__(
        self,
        skipped_module_names: List[str],
        skipped_module_classes: List[Callable],
    ):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type
        # of top level module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.node_name_to_scope: Dict[str, str] = {}

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        return (
            (
                (
                    m.__module__.startswith("torch.nn")
                    or m.__module__.startswith("horizon_plugin_pytorch.nn")
                )
                and not isinstance(m, torch.nn.Sequential)
                and not isinstance(m, torch.nn.ModuleList)
                and not isinstance(m, torch.nn.ModuleDict)
            )
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
            or super().is_leaf_module(m, module_qualified_name)
        )

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            return super().call_module(m, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = self.scope.module_path
        return node


def _prepare_fx(
    model: torch.nn.Module,
    qconfig_dict: Dict[str, Any],
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    check_qconfig: Optional[Callable] = None,
    optimize_graph: bool = False,
    hybrid: bool = False,
    hybrid_dict: Optional[Dict[str, List]] = None,
    opset_version: str = "hbdk3",
    example_inputs: Any = None,
    example_kw_inputs: Any = None,
    qconfig_setter: Optional[
        Union[Tuple[QconfigSetterBase, ...], QconfigSetterBase]
    ] = None,
    trace_method: str = "symbolic",
) -> ObservedGraphModule:
    r"""Prepare FX.

    Internal helper function for prepare_fx
    Args:
        `model`, `qconfig_dict`, `prepare_custom_config_dict`:
            see docs for :func:`~torch.quantization.prepare_fx`
        `hybrid`:
            see :func:`~horizon_plugin_pytorch.quantization.prepare_qat_fx`
    """
    assert isinstance(model, torch.nn.Module), "model should be nn.Module."
    if qconfig_dict is None:
        qconfig_dict = {}
    else:
        assert type(qconfig_dict) is dict, "qconfig_dict should be dict."
        for k in qconfig_dict.keys():
            assert k in {
                "",
                "module_type",
                "module_name",
            }, "key of qconfig_dict must in {'', 'module_type', 'module_name'}."  # noqa: E501
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    else:
        assert (
            type(prepare_custom_config_dict) is dict
        ), "prepare_custom_config_dict should be dict."
        for k in prepare_custom_config_dict.keys():
            assert k in {
                "preserved_attributes",
            }, "key of prepare_custom_config_dict must in {'preserved_attributes'}."  # noqa: E501
    if hybrid_dict is None:
        hybrid_dict = {}
    else:
        assert type(hybrid_dict) is dict, "hybrid_dict should be dict."
        for k in hybrid_dict.keys():
            assert k in {
                "module_type",
                "module_name",
            }, "key of hybrid_dict must in {'module_type', 'module_name'}."

    assert type(optimize_graph) is bool, "optimize_graph should be boolean."
    assert type(hybrid) is bool, "hybrid should be boolean."
    assert opset_version in [
        "hbdk3",
        "hbdk4",
    ], '`opset_version` only supports "hbdk3" and "hbdk4".'
    if qconfig_setter is not None:
        assert example_inputs is not None, (
            "When using qconfig template, example_inputs and qconfig_setter "
            "should be set at same time!"
        )
        if isinstance(qconfig_setter, QconfigSetterBase):
            qconfig_setter = (qconfig_setter,)
        custom_op_setter = [
            x for x in qconfig_setter if isinstance(x, ModuleNameQconfigSetter)
        ]
        template_setter = [
            x for x in qconfig_setter if isinstance(x, TemplateQconfigSetter)
        ]
        qconfig_setter = tuple(custom_op_setter + template_setter)
    assert trace_method in ("symbolic", "jit", "jit-strip")
    if trace_method in ("jit", "jit-strip"):
        assert (
            example_inputs is not None
        ), "example_inputs must be provided to use trace method in ('jit', 'jit-strip')"  # noqa: E501

    model_state = ModelState.record(model)

    use_tensor_wrapper = False

    if trace_method == "symbolic":
        preserved_attributes = ["qconfig"]
        preserved_attributes += prepare_custom_config_dict.get(
            "preserved_attributes", []
        )

        swap_nn_with_horizonnn(model)

        tracer = QuantizationTracer([], list(get_qat_module_mappings().keys()))
        graph = tracer.trace(model)
        model.node_name_to_scope = tracer.node_name_to_scope
        graph_module = GraphModuleWithAttr(model, graph, preserved_attributes)

        replace_function_with_module(graph_module, tracer.node_name_to_scope)

        graph_module = _fuse_fx(graph_module, prepare_custom_config_dict, True)

        # get _last_called_method_name for fx
        if example_inputs is not None:
            example_inputs = _as_tuple(example_inputs)
            model(*example_inputs)
    else:
        if isinstance(model, JitGraphModule):
            graph_module = model
        else:
            swap_nn_with_horizonnn(model)
            graph_module = Tracer().trace(
                model, example_inputs, example_kw_inputs
            )

        if "strip" in trace_method:
            graph_module.strip()

        FunctionReplacer.replace_function_with_module(graph_module)
        graph_module = _fuse_fx(
            graph_module, prepare_custom_config_dict, False
        )
        use_tensor_wrapper = True

    model_state.apply(graph_module)

    quantizer = Quantizer()
    prepared = quantizer.prepare(
        graph_module,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        check_qconfig=check_qconfig,
        optimize_graph=optimize_graph,
        hybrid=hybrid,
        hybrid_dict=hybrid_dict,
        opset_version=opset_version,
        example_inputs=example_inputs,
        example_kw_inputs=example_kw_inputs,
        qconfig_setter=qconfig_setter,
        only_swap_used_mods=(trace_method == "jit-strip"),
    )

    prepared._hybrid = hybrid
    prepared._opset_version = opset_version

    if use_tensor_wrapper:
        DispatchedTensorWrapper.decorate_with_tensor_wrapper(
            prepared, prepared.graph
        )

    model_state.apply(prepared)

    return prepared


def fuse_fx(
    model: torch.nn.Module,
    fuse_custom_config_dict: Optional[Dict[str, Any]] = None,  # noqa: E501
    trace_method: str = "symbolic",
    example_inputs: Any = None,
    example_kw_inputs: Any = None,
) -> Union[GraphModule, JitGraphModule]:
    r"""Fuse modules like conv+add+bn+relu etc.

    Fusion rules are defined in
    horizon_plugin_pytorch.quantization.fx.fusion_pattern.py

    Args:
        `model`: a torch.nn.Module model
        `fuse_custom_config_dict`:
            Dictionary for custom configurations for fuse_fx, e.g.

            .. code-block:: python

                fuse_custom_config_dict = {
                    # We automativally preserve all attributes, this option is
                    # just in case and not likely to be used.
                    "preserved_attributes": ["preserved_attr"],
                }

        `trace_method`: method used to get fx graph, availiable options are:
            'symbolic': Use symbolic trace.
            'jit': Use jit trace.
            'jit-strip': Use jit trace and strip the graph outside QuantStub and Dequantstub.  # noqa: E501
        `example_inputs`: model inputs. It is used to trace model when using
            jit trace method.
        `example_kw_inputs`: model keyword inputs. It is used to trace model
            when using jit trace method.
    Example:
    fuse_fx example:

    .. code-block:: python

        from torch.quantization import fuse_fx
        m = fuse_fx(m)

    """

    torch._C._log_api_usage_once("quantization_api.quantize_fx.fuse_fx")

    assert trace_method in ("symbolic", "jit", "jit-strip")
    if trace_method in ("jit", "jit-strip"):
        assert (
            example_inputs is not None
        ), "example_inputs must be provided to use trace method in ('jit', 'jit-strip')"  # noqa: E501

    preserved_attributes = ["qconfig"]

    if fuse_custom_config_dict is not None:
        msg = (
            "fuse_custom_config_dict must be a dict and "
            "only could have 'preserved_attributes'"
        )
        assert isinstance(fuse_custom_config_dict, dict), msg
        assert len(fuse_custom_config_dict) <= 1, msg
        if len(fuse_custom_config_dict) > 0:
            assert "preserved_attributes" in fuse_custom_config_dict, msg

        preserved_attributes += fuse_custom_config_dict.get(
            "preserved_attributes", []
        )

    if trace_method == "symbolic":
        swap_nn_with_horizonnn(model)

        tracer = QuantizationTracer(
            [], skipped_module_classes=list(get_qat_module_mappings().keys())
        )
        graph = tracer.trace(model)
        model.node_name_to_scope = tracer.node_name_to_scope
        graph_module = GraphModuleWithAttr(model, graph, preserved_attributes)

        replace_function_with_module(graph_module, tracer.node_name_to_scope)

        return _fuse_fx(graph_module, fuse_custom_config_dict)
    else:
        if isinstance(model, JitGraphModule):
            graph_module = model
        else:
            swap_nn_with_horizonnn(model)

            graph_module = Tracer().trace(
                model, example_inputs, example_kw_inputs
            )

        if "strip" in trace_method:
            graph_module.strip()

        FunctionReplacer.replace_function_with_module(graph_module)

        return _fuse_fx(graph_module, fuse_custom_config_dict, False)


@typechecked
def prepare_qat_fx(
    model: Union[torch.nn.Module, GraphModule, JitGraphModule],
    qconfig_dict: Optional[Dict[str, Any]] = None,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
    optimize_graph: bool = False,
    hybrid: bool = False,
    hybrid_dict: Optional[Dict[str, List]] = None,
    example_inputs: Any = None,
    example_kw_inputs: Any = None,
    qconfig_setter: Optional[
        Union[Tuple[QconfigSetterBase, ...], QconfigSetterBase]
    ] = None,
    trace_method: str = "symbolic",
    verbose: int = 0,
) -> Union[ObservedGraphModule, JitGraphModule]:
    r"""Prepare a model for quantization aware training.

    Args:
        `model`: torch.nn.Module model or GraphModule model (maybe from
            fuse_fx)
        `qconfig_dict`: qconfig_dict is a dictionary with the following
            configurations:

            .. code-block:: python

                qconfig_dict = {
                    # optional, global config
                    "": qconfig,

                    # optional, used for module types
                    "module_type": [
                        (torch.nn.Conv2d, qconfig),
                        ...,
                    ],

                    # optional, used for module names
                    "module_name": [
                        ("foo.bar", qconfig)
                        ...,
                    ],
                    # priority (in increasing order):
                    #   global, module_type, module_name, module.qconfig
                    # qconfig == None means quantization should be
                    # skipped for anything matching the rule.
                    # The qconfig of function or method is the same as the
                    # qconfig of its parent module, if it needs to be set
                    # separately, please wrap this function as a module.
                }

        `prepare_custom_config_dict`: customization configuration dictionary
            for quantization tool:

            .. code-block:: python

                prepare_custom_config_dict = {
                    # We automativally preserve all attributes, this option is
                    # just in case and not likely to be used.
                    "preserved_attributes": ["preserved_attr"],
                }

        `optimize_graph`: whether to do some process on origin model for
            special purpose. Currently only support using torch.fx to fix
            cat input scale(only used on Bernoulli)
        `hybrid`:
            Whether prepare model in hybrid mode. Default value is False and
            model runs on BPU completely. It should be True if the model
            is quantized by model convert or contains some CPU ops. In hybrid
            mode, ops which aren't supported by BPU and ops which are specified
            by the user will run on CPU.
            How to set qconfig: Qconfig in hybrid mode is the same as qconfig
            in non-hybrid mode. For BPU op, we should ensure the input of this
            op is quantized, the activation qconfig of its previous
            non-quantstub op should not be None even if its previous
            non-quantstub op is a CPU op.
            How to specify CPU op: Define CPU module_name or module_type in
            hybrid_dict.
        `hybrid_dict`:
            hybrid_dict is a dictionary to define user-specified CPU op:

            .. code-block:: python

                hybrid_dict = {
                    # optional, used for module types
                    "module_type": [torch.nn.Conv2d, ...],

                    # optional, used for module names
                    "module_name": ["foo.bar", ...],
                }
                # priority (in increasing order): module_type, module_name
                # To set a function or method as CPU op, wrap it as a module.
        `example_inputs`: model inputs. It is used to trace model or check
            model structure.
        `example_kw_inputs`: model keyword inputs. It is used to trace model
            when using jit trace method.
        `qconfig_setter`: Qconfig setter. Only needed when using qconfig
            template.
        `trace_method`: method used to get fx graph, availiable options are:
            'symbolic': Use symbolic trace.
            'jit': Use jit trace.
            'jit-strip': Use jit trace and strip the graph outside QuantStub
            and Dequantstub.
        `verbose`: whether check model structure. It has three levels:
            0: do nothing
            1: check qat model structure.
                a. if model has shared ops
                b. if model has unfused operations
                c. model quantization config

    Return:
      A GraphModule with fake quant modules (configured by qconfig_dict),
      ready for quantization aware training

    Example:
    prepare_qat_fx example:

    .. code-block:: python

        import torch
        from horizon_plugin_pytorch.quantization import get_default_qat_qconfig
        from horizon_plugin_pytorch.quantization import prepare_qat_fx

        qconfig = get_default_qat_qconfig()
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        qconfig_dict = {"": qconfig}
        prepared_model = prepare_qat_fx(float_model, qconfig_dict)
        # Run QAT training
        train_loop(prepared_model, train_loop)

    """

    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_qat_fx")
    assert (
        verbose == 0 or verbose == 1
    ), f"Only support verbose = 0 or 1 but get {verbose}"

    model = _prepare_fx(
        model,
        qconfig_dict,
        prepare_custom_config_dict,
        None,
        optimize_graph,
        hybrid,
        hybrid_dict,
        "hbdk3",
        example_inputs,
        example_kw_inputs,
        qconfig_setter,
        trace_method,
    )

    if verbose > 0:
        if example_inputs is not None:
            check_qat_model(model, example_inputs, example_kw_inputs)
        else:
            logger.warning(
                "example_inputs must be given to run check_qat_model, "
                "but got None. Skip check."
            )
    return model


def _convert_fx(
    graph_module: GraphModule,
    inplace: bool = False,
    convert_custom_config_dict: Optional[Dict[str, Any]] = None,
    _remove_qconfig: bool = True,
    hybrid: bool = False,
    fast_mode: bool = False,
) -> QuantizedGraphModule:
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}
    else:
        assert (
            type(convert_custom_config_dict) is dict
        ), "convert_custom_config_dict should be dict."
        for k in convert_custom_config_dict.keys():
            assert k in {
                "preserved_attributes",
            }, "key of convert_custom_config_dict must in {'preserved_attributes'}."  # noqa: E501

    assert type(hybrid) is bool, "hybrid should be boolean."

    model_state = ModelState.record(graph_module)

    quantizer = Quantizer()
    quantized = quantizer.convert(
        graph_module,
        inplace,
        convert_custom_config_dict,
        _remove_qconfig=_remove_qconfig,
        hybrid=hybrid,
    )
    propagate_attr_(quantized, "fast_mode", fast_mode)

    model_state.apply(quantized)

    return quantized


@typechecked
def convert_fx(
    graph_module: Union[ObservedGraphModule, JitGraphModule],
    inplace: bool = False,
    convert_custom_config_dict: Optional[Dict[str, Any]] = None,
    _remove_qconfig: bool = True,
    fast_mode: bool = False,
) -> Union[QuantizedGraphModule, JitGraphModule]:
    r"""Convert a calibrated or trained model to a quantized model.

    Args:
        `graph_module`: A prepared and calibrated/trained model (GraphModule)
        `inplace`:
            Carry out model transformations in-place, the original module is
            mutated.
        `convert_custom_config_dict`:
            dictionary for custom configurations for convert function:

            .. code-block:: python

                convert_custom_config_dict = {
                    # We automativally preserve all attributes, this option is
                    # just in case and not likely to be used.
                    "preserved_attributes": ["preserved_attr"],
                }
        `_remove_qconfig`:
            Option to remove the qconfig attributes in the model after convert.
            for internal use only.
        `fast_mode`:
            whether to accelerate quantized model forward. If set True,
            quantized model cannot be compiled.

    Return:
        A quantized model (GraphModule)

    Example:
    convert fx example:

    .. code-block:: python

        # prepared_model: the model after prepare_fx/prepare_qat_fx and
        # calibration/training
        quantized_model = convert_fx(prepared_model)

    """

    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")

    hybrid = getattr(graph_module, "_hybrid", False)
    return _convert_fx(
        graph_module,
        inplace,
        convert_custom_config_dict,
        _remove_qconfig=_remove_qconfig,
        hybrid=hybrid,
        fast_mode=fast_mode,
    )
