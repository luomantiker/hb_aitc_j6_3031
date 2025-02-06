import copy
import logging
import traceback
from enum import Enum
from typing import Any, Optional, Tuple, Union

import torch

from horizon_plugin_pytorch.fx.jit_scheme import FunctionReplacer
from horizon_plugin_pytorch.fx.jit_scheme import GraphModule as JitGraphModule
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from horizon_plugin_pytorch.quantization.qconfig_template import (
    QconfigSetterBase,
    TemplateQconfigSetter,
)
from horizon_plugin_pytorch.quantization.quantization_mappings import (
    get_qat_module_mappings,
)
from horizon_plugin_pytorch.quantization.quantize import (
    convert,
    propagate_attr_,
    propagate_qconfig_,
)
from horizon_plugin_pytorch.quantization.quantize_fx import (
    GraphModuleWithAttr,
    QuantizationTracer,
    Quantizer,
    _fuse_fx,
    replace_function_with_module,
)
from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
    DispatchedTensorWrapper,
)
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    swap_nn_with_horizonnn,
)
from horizon_plugin_pytorch.utils.check_model import check_qat_model
from horizon_plugin_pytorch.utils.misc import check_march
from horizon_plugin_pytorch.utils.model_helper import ModelState, _as_tuple
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)


class PrepareMethod(Enum):
    EAGER = "eager"
    SYMBOLIC = "symbolic"
    JIT = "jit"
    JIT_STRIP = "jit-strip"


@typechecked
def prepare(
    model: torch.nn.Module,
    example_inputs: Any = None,
    qconfig_setter: Optional[
        Union[Tuple[QconfigSetterBase, ...], QconfigSetterBase]
    ] = None,
    method: PrepareMethod = PrepareMethod.JIT_STRIP,
    example_kw_inputs: Any = None,
) -> torch.nn.Module:
    r"""Prepare model.

    Prepare and check a copy of the model for QAT.

    Args:
        model: Model to be prepared.
        example_inputs: Model inputs. Used to trace and check model.
        qconfig_setter: Qconfig setter. Used to set qconfig.
        method: Method used to trace model, availiable options are:

            * 'eager': Don't trace.
            * 'symbolic': Use symbolic trace.
            * 'jit': Use jit trace.
            * 'jit-strip': Use jit trace and strip the graph outside QuantStub and Dequantstub.
        example_kw_inputs: Model keyword inputs. Used to trace and check model.
    """  # noqa: E501
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.quantize.prepare"
    )

    check_march("you must set march before invoking prepare_qat")
    if example_kw_inputs is not None and not isinstance(
        example_kw_inputs, dict
    ):
        raise ValueError("example_kw_inputs must be a dict")

    check_example_inputs = copy.deepcopy(example_inputs)
    check_example_kw_inputs = copy.deepcopy(example_kw_inputs)

    if qconfig_setter is not None:
        if isinstance(qconfig_setter, QconfigSetterBase):
            qconfig_setter = (qconfig_setter,)

        has_template_qconfig_setter = any(
            isinstance(i, TemplateQconfigSetter) for i in qconfig_setter
        )

        if has_template_qconfig_setter:
            if example_inputs is None and example_kw_inputs is None:
                raise ValueError(
                    "example_inputs or example_kw_inputs must be provided "
                    "when using TemplateQconfigSetter."
                )

    model_state = ModelState.record(model)
    prepared_model = copy.deepcopy(model)

    if method == PrepareMethod.EAGER:
        swap_nn_with_horizonnn(prepared_model)
        propagate_qconfig_(prepared_model)
        propagate_attr_(prepared_model, "quantized_aligned_qat", False)

        if qconfig_setter is not None:
            for setter in qconfig_setter:
                setter.set_qconfig(
                    prepared_model, example_inputs, example_kw_inputs
                )

        convert(
            prepared_model,
            mapping=get_qat_module_mappings(),
            inplace=True,
            remove_qconfig=False,
        )

    elif method in (
        PrepareMethod.SYMBOLIC,
        PrepareMethod.JIT,
        PrepareMethod.JIT_STRIP,
    ):
        use_tensor_wrapper = False
        if method == PrepareMethod.SYMBOLIC:
            swap_nn_with_horizonnn(prepared_model)

            tracer = QuantizationTracer(
                [], list(get_qat_module_mappings().keys())
            )
            graph = tracer.trace(prepared_model)
            prepared_model.node_name_to_scope = tracer.node_name_to_scope
            graph_module = GraphModuleWithAttr(
                prepared_model, graph, ["qconfig"]
            )

            replace_function_with_module(
                graph_module, tracer.node_name_to_scope
            )

            graph_module = _fuse_fx(graph_module, allow_recompile=True)

            # get _last_called_method_name for fx
            if example_inputs is not None or example_kw_inputs is not None:
                if example_inputs is None:
                    example_inputs = ()
                if example_kw_inputs is None:
                    example_kw_inputs = {}
                example_inputs = _as_tuple(example_inputs)
                prepared_model(*example_inputs, **example_kw_inputs)

        elif method in (PrepareMethod.JIT_STRIP, PrepareMethod.JIT):
            if example_inputs is None and example_kw_inputs is None:
                raise ValueError(
                    "example_inputs or example_kw_inputs must be provided "
                    "when method is jit/jit-strip."
                )
            if isinstance(prepared_model, JitGraphModule):
                graph_module = prepared_model
            else:
                swap_nn_with_horizonnn(prepared_model)
                graph_module = Tracer().trace(
                    prepared_model, example_inputs, example_kw_inputs
                )

            if method == PrepareMethod.JIT_STRIP:
                graph_module.strip()

            FunctionReplacer.replace_function_with_module(graph_module)
            graph_module = _fuse_fx(graph_module, allow_recompile=False)
            use_tensor_wrapper = True

        model_state.apply(graph_module)

        quantizer = Quantizer()
        prepared_model = quantizer.prepare(
            graph_module,
            example_inputs=example_inputs,
            example_kw_inputs=example_kw_inputs,
            qconfig_setter=qconfig_setter,
            only_swap_used_mods=(method == PrepareMethod.JIT_STRIP),
        )

        if use_tensor_wrapper:
            DispatchedTensorWrapper.decorate_with_tensor_wrapper(
                prepared_model, prepared_model.graph
            )

    model_state.apply(prepared_model)

    if check_example_inputs is not None or check_example_kw_inputs is not None:
        try:
            check_qat_model(
                prepared_model,
                check_example_inputs,
                check_example_kw_inputs,
                save_results=True,
            )
        except Exception:
            logger.warning(f"check model failed.\n{traceback.format_exc()}")
    else:
        logger.warning(
            "example_inputs must be given to run check_qat_model, "
            "but got None. Skip check..."
        )

    prepared_model._prepare_method = method

    return prepared_model
