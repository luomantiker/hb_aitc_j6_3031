"""Prepare and convert."""

import copy
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import fx

from horizon_plugin_pytorch.fx.tracer import CustomTracer
from horizon_plugin_pytorch.nn.qat import MultiScaleRoIAlign
from horizon_plugin_pytorch.nn.qat.stubs import QuantStub
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    swap_nn_with_horizonnn,
)
from horizon_plugin_pytorch.utils.check_model import check_qat_model
from horizon_plugin_pytorch.utils.misc import check_march
from horizon_plugin_pytorch.utils.model_helper import ModelState
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .observer import CalibObserver, FixedScaleObserver
from .qconfig_template import (
    ModuleNameQconfigSetter,
    QconfigSetterBase,
    TemplateQconfigSetter,
)
from .quantization_mappings import (
    get_qat_module_mappings,
    get_quantized_operator_mappings,
    wrap_qat_modules_for_fx,
)

logger = logging.getLogger(__name__)


def _set_qat_activation_post_process_state(m):
    from ..nn.qat import Conv2d, ConvTranspose2d

    fake_quant_enabled = m.activation_post_process.fake_quant_enabled.item()
    observer_enabled = m.activation_post_process.observer_enabled.item()
    if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
        m.activation_post_process = m.qconfig.activation(
            channel_len=m.out_channels
        )
    else:
        m.activation_post_process = m.qconfig.activation()
    if fake_quant_enabled != 1:
        m.activation_post_process.disable_fake_quant()
    if observer_enabled != 1:
        m.activation_post_process.disable_observer()


def _calculate_statistical_qparams(m):
    amax = m.activation_post_process.activation_post_process.compute_amax()
    _set_qat_activation_post_process_state(m)
    observer = m.activation_post_process.activation_post_process
    if isinstance(observer, FixedScaleObserver):
        logger.warning(
            "use FixedScaleObserver in qat but not in calibration",
            extra={"call_times_context": ("message")},
        )
    if amax is not None and not isinstance(observer, FixedScaleObserver):
        observer.min_val.resize_(amax.shape)
        observer.min_val.copy_(-amax)
        observer.max_val.resize_(amax.shape)
        observer.max_val.copy_(amax)

    (
        scale,
        zero_point,
    ) = observer.calculate_qparams()
    return scale, zero_point


def _get_fixed_qparams(m):
    (
        scale,
        zero_point,
    ) = m.activation_post_process.activation_post_process.calculate_qparams()
    return scale, zero_point


def _calculate_activation_qparams(m):
    setted_scale = m.activation_post_process.scale
    if isinstance(
        m.activation_post_process.activation_post_process, CalibObserver
    ):
        scale, zero_point = _calculate_statistical_qparams(m)
    else:
        assert isinstance(
            m.activation_post_process.activation_post_process,
            FixedScaleObserver,
        )
        scale, zero_point = _get_fixed_qparams(m)
        fixed_scale_observer = (
            m.activation_post_process.activation_post_process
        )
        _set_qat_activation_post_process_state(m)
        if isinstance(
            m.activation_post_process.activation_post_process,
            FixedScaleObserver,
        ):
            qat_fixed_scale = (
                m.activation_post_process.activation_post_process.scale
            )
            assert qat_fixed_scale.item() == scale.item(), (
                f"calibration fixed scale must be equal to qat scale, "
                f"but get {scale.item()} in calibration and "
                f"{qat_fixed_scale.item()} in qat"
            )
        else:
            m.activation_post_process.activation_post_process = (
                fixed_scale_observer
            )
        if m.activation_post_process.observer_enabled.item() == 0:
            assert scale.item() == setted_scale.item()
    m.activation_post_process.set_qparams(scale, zero_point)
    if m.activation_post_process.observer_enabled.item() == 0:
        with torch.no_grad():
            m.activation_post_process.scale.copy_(setted_scale)


def _calculate_weight_qparams(m):
    if hasattr(m, "weight_fake_quant"):
        fake_quant_enabled = m.weight_fake_quant.fake_quant_enabled.item()
        observer_enabled = m.weight_fake_quant.observer_enabled.item()
        m.weight_fake_quant = m.qconfig.weight(
            channel_len=m.weight_fake_quant.channel_len
        ).to(m.weight.device)
        if hasattr(m, "_get_weight_for_fake_quant"):
            weight_for_fake_quant = m._get_weight_for_fake_quant()
        else:
            weight_for_fake_quant = m.weight
        m.weight_fake_quant.activation_post_process(weight_for_fake_quant)
        if fake_quant_enabled != 1:
            m.weight_fake_quant.disable_fake_quant()
        if observer_enabled != 1:
            m.weight_fake_quant.disable_observer()
        (
            scale,
            zero_point,
        ) = m.weight_fake_quant.calculate_qparams()
        m.weight_fake_quant.set_qparams(scale, zero_point)


def replace_fake_quantize(model, hybrid=False, node_name_to_qconfig=None):
    from .fake_quantize import CalibFakeQuantize

    for m in model.modules():
        if hasattr(m, "activation_post_process"):
            if isinstance(m.activation_post_process, CalibFakeQuantize):
                if m.qconfig.activation is not None:
                    _calculate_activation_qparams(m)
                else:
                    m.activation_post_process = None
                if m.qconfig.weight is not None:
                    _calculate_weight_qparams(m)
                else:
                    m.weight_fake_quant = None

    if hybrid and node_name_to_qconfig is not None:
        # replace auto-inserted calibration fake quant in hybrid mode
        # wrap autu-inserted calibration fake quant to reuse the logic
        # of _calculate_activation_qparams
        class _WrappedCalibFakeQuantize(torch.nn.Module):
            def __init__(self, activation_post_process, qconfig):
                super().__init__()
                self.activation_post_process = activation_post_process
                self.qconfig = qconfig
                self.enable = qconfig is not None

            def forward(self, x):
                # qconfig will be removed after convert, judge by var "enable"
                if self.enable:
                    # replace calib fake quant with normal fake quant
                    return self.activation_post_process(x)
                else:
                    # remove calib fake quant
                    return x

        # autu-inserted fake quants are named_children of the model
        named_children = dict(model.named_children())
        for name, module in named_children.items():
            if isinstance(module, CalibFakeQuantize) and name.endswith(
                "_activation_post_process"
            ):
                provider_node_name = name.split("_activation_post_process")[0]
                provider_qconfig = node_name_to_qconfig.get(
                    provider_node_name, None
                )
                wrapped_fq = _WrappedCalibFakeQuantize(
                    module, provider_qconfig
                )
                if provider_qconfig is not None:
                    _calculate_activation_qparams(wrapped_fq)
                setattr(model, name, wrapped_fq)

    return model


class NotSetQConfig:
    pass


def _propagate_qconfig_helper(
    module,
    qconfig_dict,
    qconfig_parent=NotSetQConfig,
    prefix="",
    unused_qconfig_key=None,
    white_list: Optional[List[str]] = None,
):
    r"""Propagate qconfig.

    This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to
            quantization configuration
        qconfig_parent: quantization config of parent module, we will fallback
            to this config when there is no specified config for current module
        prefix: corresponding prefix of the current module, used as key in
            qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    if (
        white_list is not None
        and all(not x.startswith(prefix) for x in white_list)
        and not prefix.startswith(tuple(white_list))
    ):
        return

    if unused_qconfig_key is None:
        unused_qconfig_key = set(qconfig_dict.keys())
    module_qconfig = qconfig_parent

    if type(module) in qconfig_dict:
        module_qconfig = qconfig_dict[type(module)]
        unused_qconfig_key.discard(type(module))
    if prefix in qconfig_dict:
        module_qconfig = qconfig_dict[prefix]
        unused_qconfig_key.discard(prefix)

    module_qconfig = getattr(module, "qconfig", module_qconfig)

    if module_qconfig is not NotSetQConfig:
        # module can implement this method to modify qconfig of its children
        if hasattr(module, "propagate_qconfig"):
            module.propagate_qconfig(module_qconfig)
        module.qconfig = module_qconfig

    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        _propagate_qconfig_helper(
            child,
            qconfig_dict,
            module_qconfig,
            module_prefix,
            unused_qconfig_key,
            white_list,
        )

    if prefix == "" and len(unused_qconfig_key) > 0:
        logger.warning(
            f"The model doesn't have {unused_qconfig_key}. "
            f"Please make sure qconfig_dict is set correctly."
        )


def propagate_qconfig_(
    module: torch.nn.Module, qconfig_dict=None, white_list=None
):
    r"""Propagate qconfig.

    Propagate qconfigthrough the module hierarchy and
    assign `qconfig` attribute on each leaf module.

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        white_list: A list of mod names to propagate qconfig on. If not
            given, include all modules by default.

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}

    # If user specify qconfig by mod name, discard the white_list
    for k in tuple(qconfig_dict.keys()):
        if isinstance(k, str):
            try:
                mod = module.get_submodule(k)
            except AttributeError:
                pass
            else:
                if hasattr(mod, "qconfig"):
                    logger.warning(
                        "Mod {} already has a qconfig, the qconfig in "
                        "qconfig_dict['module_name'] will be ignored.".format(
                            k
                        )
                    )
                else:
                    mod.qconfig = qconfig_dict.pop(k)

    _propagate_qconfig_helper(module, qconfig_dict, white_list=white_list)


def propagate_attr_(module, attr_name, value):
    """
    Propagate attr.

    A general function to propagate attribute from module to its children.

    Args:
        module (torch.nn.Module): Root module.
        attr_name (str): Attribute name.
        value (Any): Default value to use.
    """
    if hasattr(module, attr_name):
        value_for_child = getattr(module, attr_name)
    else:
        setattr(module, attr_name, value)
        value_for_child = value

    for _, child in module.named_children():
        propagate_attr_(child, attr_name, value_for_child)


def _is_in_out_same_scale_node(node, modules):
    # call_function and not floatfunctional call_method node, return true
    if node.op == "call_function":
        return True
    elif node.op == "call_method":
        from horizon_plugin_pytorch.nn.qat import FloatFunctional

        return not isinstance(modules[node.args[0].target], FloatFunctional)
    elif node.op == "call_module":
        m = modules[node.target]
        if isinstance(m, QuantStub):
            return False
        if (
            hasattr(m, "activation_post_process")
            and m.activation_post_process is not None
        ):
            return m.activation_post_process.observer_enabled.item() == 0
        else:
            return True
    else:
        raise AssertionError(
            "Should not encounter 'placeholder', 'get_attr' and 'output' node"
        )


def _construct_input_list(node):
    # construct tensor input nodes to process
    # 1) call_method node inputs start from args[1]
    # 2) Usually, multi inputs of forward functions are in format
    #   a) A list at the first index. Eg. cat([x, y], dim=1)
    #   b) multi-tensor input. Eg. add(x, y)
    #   c) multi-list inputs only process first list input now. Eg.
    #       MultiScaleRoIAlign([f1, f2,...], [box1, box2,...])
    input_node_list = []
    node_type = type(node)
    for i, item in enumerate(node.args + tuple(node.kwargs.values())):
        if node.op == "call_method" and i == 0:
            # first item of call_method node is prefix name of class
            continue
        if isinstance(item, Sequence):
            input_node_list += item
            break
        elif type(item) == node_type:
            input_node_list.append(item)
    return input_node_list


def _share_inputs_activation(node_inputs_list, modules):
    shared_post_process = None
    dtype = None
    for m in node_inputs_list:
        # Process input args.
        # 1) call_function or not 'FloatFunction' call_method node or input
        #    scale = output scale Module, process their inputs recursively
        # 2) otherwise, directly replace current node scale
        shared_activation_node = m
        if _is_in_out_same_scale_node(m, modules):
            # check and find the first input_scale != output_scale node
            m_inputs_list = _construct_input_list(m)
            _share_inputs_activation(m_inputs_list, modules)
        else:
            if m.op == "call_method":
                from horizon_plugin_pytorch.nn.qat import FloatFunctional

                # Must be FloatFunctional method here now
                shared_activation_node = m.args[0]
                assert isinstance(
                    modules[shared_activation_node.target], FloatFunctional
                ), "Must be FloatFunctional here!"

            name = shared_activation_node.target
            mod = modules[name]
            if (
                isinstance(mod, QuantStub) and mod.scale is not None
            ) or isinstance(mod.activation_post_process, FixedScaleObserver):
                raise RuntimeError(
                    f"The activation_post_process of {name}"
                    + " will be shared with other nodes by torch.fx. "
                    + "Please check your fixed scale!"
                )

            if shared_post_process is None:
                shared_post_process = mod.activation_post_process
                dtype = mod.activation_post_process.dtype
            else:
                if mod.activation_post_process.dtype != dtype:
                    raise RuntimeError(
                        f"activation_post_process of {name}"
                        + " can not shared with others because dtype differs: "
                        + f"{mod.activation_post_process.dtype} != {dtype}"
                    )
                mod.activation_post_process = shared_post_process

            logger.warning(
                f"activation_post_process of {name} will"
                f" be shared with other modules",
                extra={"call_times_context": ("message")},
            )


def _unify_inputs_scale(gm, prefixes, types, functions, methods):
    """Replace cat input scale with cat result scale."""
    modules = dict(gm.named_modules())
    for node in reversed(gm.graph.nodes):
        if (
            (
                node.op == "call_module"
                and (
                    any([node.target.startswith(p) for p in prefixes])
                    or type(gm.get_submodule(node.target)) in types
                )
            )
            or (node.op == "call_function" and node.target in functions)
            or (node.op == "call_method" and node.target in methods)
        ):
            input_node_list = _construct_input_list(node)
            _share_inputs_activation(input_node_list, modules)
    gm.recompile()
    return gm


def model_preprocess(model, optimize_kwargs):
    """Preprocess model for some special purpose.

    Current only support unify cat input and output scale on Bernoulli by fx

    Args:
        model: the model to be preprocess

    Return:
        The GraphModel after preprocess

    """
    wrap_qat_modules_for_fx()
    _func_map = {
        "unify_inputs_scale": _unify_inputs_scale,
    }
    tracer = CustomTracer()
    g = tracer.trace(model)
    gm = fx.GraphModule(model, g)

    if optimize_kwargs is None:
        optimize_kwargs = {
            "opt_types": ("unify_inputs_scale",),
            "module_prefixes": (),
            "module_types": (MultiScaleRoIAlign,),
            "functions": (),
            "methods": ("cat",),
        }

    opt_types = optimize_kwargs.get("opt_types", ("unify_inputs_scale",))
    prefixes = optimize_kwargs.get("module_prefixes", ())
    types = optimize_kwargs.get("module_types", ())
    functions = optimize_kwargs.get("functions", ())
    methods = optimize_kwargs.get("methods", ())
    if not prefixes and not types and not functions and not methods:
        types = (MultiScaleRoIAlign,)
        methods = ("cat",)
    for t in opt_types:
        if t not in _func_map.keys():
            raise ValueError(f"Do not support {t} optimization")
        gm = _func_map[t](gm, prefixes, types, functions, methods)
    return gm


@typechecked
def prepare_qat(
    model: torch.nn.Module,
    mapping: Optional[
        Dict[Type[torch.nn.Module], Type[torch.nn.Module]]
    ] = None,
    inplace: bool = False,
    optimize_graph: bool = False,
    hybrid: bool = False,
    optimize_kwargs: Optional[Dict[str, Tuple]] = None,
    example_inputs: Any = None,
    qconfig_setter: Optional[
        Union[Tuple[QconfigSetterBase, ...], QconfigSetterBase]
    ] = None,
    verbose: int = 0,
):
    r"""Prepare qat.

    Prepare a copy of the model for quantization-aware training and
    converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
        optimize_graph: whether to do some process on origin model for special
                        purpose. Currently only support using torch.fx to fix
                        cat input scale(only used on Bernoulli)
        hybrid: whether to generate a hybrid model that some intermediate
                operation is computed in float. There are some constraints for
                this functionality now:
                1. The hybrid model cannot pass check_model and cannot be
                compiled.
                2. Some quantized operation cannot directly accept input from
                float operation, user need to manually insert QuantStub.
        optimize_kwargs: a dict for optimize graph with the following format:

            .. code-block:: python

                optimize_kwargs = {
                    # optional, specify which type of optimization to do. Only
                    # support "unify_inputs_scale" now
                    "opt_types": ("unify_inputs_scale",),

                    # optional, modules start with qualified name to optimize
                    "module_prefixes": ("backbone.conv",),

                    # optional, modules in these types will be optimize
                    "module_types": (horizon.nn.qat.conv2d,),

                    # optional, functions to optimize
                    "functions": (torch.clamp,),

                    # optional, methods to optimize. Only support
                    # FloatFunctional methods now
                    "methods": ("add",),
                }
        example_inputs: model inputs. It is used to trace model or check
                        model structure.
        qconfig_setter: Qconfig setter. Only needed when using qconfig
            template.
        verbose: whether check model structure. it has two levels:
                 0: do nothing
                 1: check model structure
                    a. if model has shared ops
                    b. if model has unfused operations
                    c. model quantization config
    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.quantize.prepare_qat"
    )

    check_march("you must set march before invoking prepare_qat")

    assert isinstance(inplace, bool), "param 'inplace' must be bool type"
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
        qconfig_setter = custom_op_setter + template_setter
    assert (
        verbose == 0 or verbose == 1
    ), f"Only support verbose = 0 or 1 but get {verbose}"

    model_state = ModelState.record(model)

    swap_nn_with_horizonnn(model)

    QTensor.allow_float_operation(hybrid)

    if mapping is None:
        mapping = get_qat_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig_dict=None)
    propagate_attr_(model, "quantized_aligned_qat", False)

    if example_inputs is not None and qconfig_setter is not None:
        for setter in qconfig_setter:
            setter.set_qconfig(model, example_inputs)

    convert(
        model,
        mapping=mapping,
        inplace=True,
        remove_qconfig=False,
        fast_mode=False,
    )
    if optimize_graph:
        model_preprocess(model, optimize_kwargs)

    if verbose > 0:
        if example_inputs is not None:
            check_qat_model(model, example_inputs)
        else:
            logger.warning(
                "example_inputs must be given to run check_qat_model, "
                "but got None. Skip check..."
            )

    model_state.apply(model)

    return model


def _remove_qconfig(module):
    r"""Clean up the qconfig.

    Clean up the qconfig left in the module so that new qconfig can be
    propagated.

    Args:
        module: module to be cleaned up
    """
    for child in module.children():
        _remove_qconfig(child)

    if hasattr(module, "qconfig"):
        del module.qconfig


@typechecked
def convert(
    module: torch.nn.Module,
    mapping: Optional[
        Dict[Type[torch.nn.Module], Type[torch.nn.Module]]
    ] = None,
    inplace: bool = False,
    remove_qconfig: bool = True,
    fast_mode: bool = False,
    swapable_names=None,
):
    r"""Convert modules.

    Convert submodules in input module to a different module according
    to `mapping` by calling `from_float` method on the target module class.
    And remove qconfig at the end if remove_qconfig is set to True.

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated
        fast_mode: whether to accelerate quantized model forward. If set True,
                   quantized model cannot be compiled

    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.quantize.convert"
    )

    check_march("you must set march before invoking convert")

    assert isinstance(inplace, bool), "argument 'inplace' must be of bool type"
    assert isinstance(
        remove_qconfig, bool
    ), "argument 'remove_qconfig' must be of bool type"

    model_state = ModelState.record(module)

    if not inplace:
        module = copy.deepcopy(module)

    propagate_attr_(module, "preserve_qat_mode", False)

    swapped_modules: Dict[torch.nn.Module, torch.nn.Module] = {}
    with torch.no_grad():
        # disable autograd for all buffer copies
        _convert(
            module,
            mapping,
            inplace=True,
            swapped_modules=swapped_modules,
            swapable_names=swapable_names,
        )
    if remove_qconfig:
        _remove_qconfig(module)
    if fast_mode:
        propagate_attr_(module, "fast_mode", fast_mode)

    model_state.apply(module)

    return module


def _is_swappable_module(module, mapping):
    # find leaf module
    # TODO
    pass


def _convert_qtensor_to_fake_quantized(input):
    if isinstance(input, QTensor):
        return input.to_fake_quantized()
    if isinstance(input, (list, tuple)):
        return type(input)(
            _convert_qtensor_to_fake_quantized(x) for x in input
        )
    return input


def _convert_qtensor_to_quantized(input):
    if isinstance(input, QTensor):
        return input.to_quantized()
    if isinstance(input, (list, tuple)):
        return type(input)(_convert_qtensor_to_quantized(x) for x in input)
    return input


def _convert_input_to_fake_quantized_hook(mod, input):
    return _convert_qtensor_to_fake_quantized(input)


def _convert_output_to_quantized_hook(mod, input, output):
    return _convert_qtensor_to_quantized(output)


def swap_module(mod, mapping, swapped_modules):
    r"""Swap the modules.

    Swap the modules if it has a quantized
    counterpart and it has an `observer` attached.

    copy from torch.quantization.quantize.swap_module, but judge

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module
        swapped_modules: a dictionary that maps from source module to swapped
                         module

    Return:
        The corresponding quantized module of `mod`
    """
    torch._C._log_api_usage_once(
        "horizon_plugin_pytorch.quantization.quantize.swap_module"
    )
    from torch.quantization import DeQuantStub

    new_mod = mod
    if mod in swapped_modules:
        new_mod = swapped_modules[mod]
    elif (
        hasattr(mod, "qconfig")
        and mod.qconfig is not None
        or isinstance(mod, DeQuantStub)
    ):
        # check unobserved ops when converting to quantized
        if (
            mapping is get_quantized_operator_mappings()
            and getattr(mod, "activation_post_process", None) is not None
            and (mod.activation_post_process.scale == 1.0).all()
            and hasattr(
                mod.activation_post_process.activation_post_process, "max_val"
            )
            and mod.activation_post_process.activation_post_process.max_val.numel()  # noqa: E501
            == 0
        ):
            logger.warning(
                f"{mod._get_name()} has not collected any statistics "
                f"of activations and its scale is 1, "
                f"please check whether this is intended!",
                extra={"call_times_context": ("message")},
            )

        swapped = False

        if type(mod) in mapping:
            if (
                mapping is get_quantized_operator_mappings()
                and hasattr(mod, "preserve_qat_mode")
                and mod.preserve_qat_mode
            ):
                mod.register_forward_pre_hook(
                    _convert_input_to_fake_quantized_hook
                )
                mod.register_forward_hook(_convert_output_to_quantized_hook)
            else:
                with torch.no_grad():
                    new_mod = mapping[type(mod)].from_float(mod)
                    if not hasattr(new_mod, "preserve_qat_mode") and hasattr(
                        mod, "preserve_qat_mode"
                    ):
                        new_mod.preserve_qat_mode = mod.preserve_qat_mode

                    if hasattr(mod, "quantized_aligned_qat"):
                        new_mod.quantized_aligned_qat = (
                            mod.quantized_aligned_qat
                        )

                    swapped_modules[mod] = new_mod
                    swapped = True

        # Only copy hook from qat to quantized
        if mapping is get_quantized_operator_mappings() and swapped:
            # Preserve module's hooks
            new_mod._forward_pre_hooks.update(mod._forward_pre_hooks)
            new_mod._forward_hooks.update(mod._forward_hooks)
            if hasattr(mod, "_forward_hooks_with_kwargs"):
                new_mod._forward_pre_hooks_with_kwargs.update(
                    mod._forward_pre_hooks_with_kwargs
                )
                new_mod._forward_hooks_with_kwargs.update(
                    mod._forward_hooks_with_kwargs
                )

    return new_mod


def _convert(
    module,
    mapping=None,
    inplace=False,
    swapped_modules={},  # noqa: B006
    swapable_names=None,
    prefix="",
):
    r"""Convert submodules.

    Convert submodules in input module to a different module
    according to `mapping` by calling `from_float` method on
    the target module class.

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated
        swapped_modules: a dictionary that maps from source module to swapped
                         module
        swapped_modules: a list of module names that can be swapped
        prefix: the qualified name of current input module

    """
    if mapping is None:
        mapping = get_quantized_operator_mappings()
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    SWAPPABLE_MODULES = set(  # noqa: N806
        get_qat_module_mappings().keys()
    ) | set(  # noqa: N806
        get_quantized_operator_mappings().keys()
    )

    if swapable_names is not None:
        swapable_names = tuple(swapable_names)

    for name, mod in module.named_children():
        # both swappable modules and observed custom modules are
        # swapped as one unit
        try:
            current_name = (
                name if prefix == "" else "{}.{}".format(prefix, name)
            )
            if type(mod) not in SWAPPABLE_MODULES:
                _convert(
                    mod,
                    mapping,
                    inplace=True,
                    swapped_modules=swapped_modules,
                    swapable_names=swapable_names,
                    prefix=current_name,
                )
            if swapable_names is None or current_name.startswith(
                swapable_names
            ):
                # TODO: judge swappable
                reassign[name] = swap_module(mod, mapping, swapped_modules)
        except Exception as e:
            e.args += ("When swapping mod {}".format(current_name),)
            raise e

    for key, value in reassign.items():
        setattr(module, key, value)

    return module
