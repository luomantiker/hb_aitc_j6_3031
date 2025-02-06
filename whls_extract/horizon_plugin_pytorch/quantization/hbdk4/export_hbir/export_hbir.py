import builtins
import copy
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import wraps
from typing import Any, Dict, Optional

import torch
from hbdk4.compiler import Module, ir
from hbdk4.compiler.dialects.func import FuncOp
from hbdk4.compiler.dialects.hbir import TrackAttr
from hbdk4.compiler.ops import hbir
from torch import Tensor, nn
from torch.quantization import QuantStub as TorchQuantStub
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.nn.qat import ConvBN2d
from horizon_plugin_pytorch.nn.qat.qat_meta import is_float
from horizon_plugin_pytorch.qat_mode import QATMode, get_qat_mode
from horizon_plugin_pytorch.quantization import (
    FakeQuantState,
    QuantStub,
    set_fake_quantize,
)
from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
    DispatchedTensorWrapper,
)
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    swap_nn_with_horizonnn,
)
from horizon_plugin_pytorch.utils.location_info import (
    LocationManager,
    PicklableStrDict,
)
from horizon_plugin_pytorch.utils.misc import (
    pytree_convert,
    tensor_struct_repr,
)
from horizon_plugin_pytorch.utils.model_helper import (
    _as_tuple,
    register_forward_hook,
)
from horizon_plugin_pytorch.utils.typeguard import (
    suppress_type_checks,
    typechecked,
)
from .checker import CheckInfoItem, DynamicInfo, ModelChecker
from .utils import (
    TreeSpec,
    get_hbdk4_version,
    get_hbir_dtype,
    get_hbir_tensor_type,
    pickle_treespec,
    to_numpy,
    unpickle_treespec,
)

__all__ = [
    "check",
    "export",
    "is_exporting",
    "get_hbir_input_flattener",
    "get_hbir_output_unflattener",
    "get_export_hbir_plugin_version",
]

logger = logging.getLogger(__name__)


def _fuse_conv_bn(model):
    for _, mod in model.named_modules():
        if isinstance(mod, ConvBN2d):
            qat_mode = get_qat_mode()
            assert qat_mode in [QATMode.WithBN, QATMode.WithBNReverseFold], (
                "Only support {} and {} ".format(
                    QATMode.WithBN, QATMode.WithBNReverseFold
                )
                + "mode when exporting unfused conv-bn qat model."
            )
            if qat_mode == QATMode.WithBN:
                bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
                fused_weight = mod.weight * (
                    mod.bn.weight * bn_var_rsqrt
                ).reshape([-1] + [1] * (len(mod.weight.shape) - 1))
                mod.weight.copy_(fused_weight)
                wscale = (
                    mod.weight_fake_quant.scale
                    * torch.abs(mod.bn.weight)
                    * bn_var_rsqrt
                )
                mod.weight_fake_quant.set_qparams(wscale)
                if mod.bias is not None:
                    fused_bias = (
                        mod.bias - mod.bn.running_mean
                    ) * mod.bn.weight * bn_var_rsqrt + mod.bn.bias
                    mod.bias.copy_(fused_bias)
                else:
                    fused_bias = (
                        (-1)
                        * mod.bn.running_mean
                        * mod.bn.weight
                        * bn_var_rsqrt
                        + mod.bn.bias
                    )
                    mod.bias = torch.nn.Parameter(fused_bias)
            else:
                fused_weight, fused_bias = torch.nn.utils.fuse_conv_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
                mod.weight.copy_(fused_weight)
                if mod.bias is not None:
                    mod.bias.copy_(fused_bias)
                else:
                    mod.bias = torch.nn.Parameter(fused_bias)

            mod.bn = torch.nn.Identity()


@typechecked
def check(
    model: nn.Module,
    example_inputs: Any,
    *,
    save_dir: Optional[str] = None,
    only_record_errors: bool = True,
) -> bool:
    """Check problems in export of a nn.Module.

    Export errors will not interrupt the check.

    Args:
        model: Input model.
        example_inputs: Example input for tracing.
        save_dir: If specified, save check results to this dir as csv file.
        only_record_errors: Whether only include problematic ops in report.

    Returns:
        bool: Whether the check is passed.
    """
    checker = Exporter.export(
        copy.deepcopy(model),
        example_inputs,
        True,
        only_record_errors=only_record_errors,
    )
    if checker.passed():
        logger.info(checker.summary())
    else:
        logger.warning(checker.summary())

    if save_dir is not None:
        checker.save_to(save_dir)

    return checker.passed()


@typechecked
def export(
    model: nn.Module,
    example_inputs: Any,
    *,
    name: str = "forward",
    input_names: Optional[Any] = None,
    output_names: Optional[Any] = None,
    input_descs: Optional[Any] = None,
    output_descs: Optional[Any] = None,
    native_pytree: bool = True,
) -> Module:
    """Export nn.Module to hbir model.

    Args:
        model: Input model.
        example_inputs: Example input for tracing.
        name: The name of func in exported module.
            Users can get the func by getattr(hbir_module, name).
        input_names: Set hbir inputs with given names, should have the same
            structure with example_inputs.
        output_names: Set hbir outputs with given names, should have the same
            structure with model output.
        input_descs:
            Set hbir inputs with given descriptions, should have the same
            structure with example_inputs.
        output_descs:
            Set hbir outputs with given descriptions, should have the same
            structure with model output.
        native_pytree:
            Whether use native pytree support provided by hbdk4.
            Assume the model input is passed to torch model as::

                torch_rets = torch_model(*model_args)

            When using hbdk4 pytree, the hbir model can be called as::

                hbir_rets = hbir_model.functions[0](model_args)
                assert torch_rets == hbir_rets

            When using plugin pytree, the hbir model can be called as::

                flat_inputs = get_hbir_input_flattener(hbir_model)(model_args)
                flat_rets = hbir_model.functions[0](*flat_inputs)
                hbir_rets = get_hbir_output_unflattener(hbir_model)(flat_rets)
                assert torch_rets == hbir_rets


    Returns:
        Hbir model wrapped with Module.
    """
    if not native_pytree:
        logger.warning(
            "The pytree support of plugin is deprecated and will be removed "
            "soon, please use native pytree to get better experience.",
            extra={"call_times_context": ("message")},
        )
    for n, m in model.named_modules():
        if isinstance(m, (QuantStub, TorchQuantStub)):
            logger.warning(
                "QuantStub '{}' of input model is in float state, indicating "
                "that you may be trying to export a float model.".format(n)
            )

    model = copy.deepcopy(model)
    with torch.no_grad():
        _fuse_conv_bn(model)
    ret = Exporter.export(
        model,
        example_inputs,
        False,
        name=name,
        input_names=input_names,
        output_names=output_names,
        input_descs=input_descs,
        output_descs=output_descs,
        native_pytree=native_pytree,
    )
    return ret


def is_exporting():
    return Exporter._is_exporting


class HbirModuleInfo:
    _version = 0

    def __init__(self, output_tree_spec: TreeSpec, plugin_version: str):
        self.output_tree_spec = output_tree_spec
        self.plugin_version = plugin_version

    def to_dict(self):
        return PicklableStrDict(
            {
                "InfoVersion": str(self._version),
                "OutSpec": pickle_treespec(self.output_tree_spec),
                "PluginVersion": self.plugin_version,
            }
        )

    def pickle(self):
        return self.to_dict().pickle()

    @classmethod
    def from_dict(cls, str_dict):
        if int(str_dict["InfoVersion"]) == 0:
            return cls(
                unpickle_treespec(str_dict["OutSpec"]),
                str_dict["PluginVersion"],
            )
        else:
            raise ValueError("Unknown version of HbirModuleInfo")

    @classmethod
    def unpickle(cls, string: str):
        try:
            str_dict = PicklableStrDict.unpickle(string)
            return cls.from_dict(str_dict)
        except Exception:
            return cls(unpickle_treespec(string), None)

    def write_to(self, func):
        func.internal_desc = self.pickle()

    @classmethod
    def read_from(cls, func):
        # In older version, we write info to func.desc.
        # So when reading from internal_desc failed, try to read from desc.
        # But when it is also failed reading from desc, raise the previous
        # exception.
        try:
            info = HbirModuleInfo.unpickle(func.internal_desc)
        except Exception as e:
            try:
                info = HbirModuleInfo.unpickle(func.desc)
            except Exception:
                raise e
        return info


def get_hbir_input_flattener(model: Module) -> callable:
    """Get an callable func to flatten model input into a flat tuple of Tensor.

    Args:
        model: Hbir model.
    """
    logger.warning(
        "The pytree support of plugin is deprecated and will be removed "
        "soon, please use native pytree to get better experience.",
        extra={"call_times_context": ("message")},
    )

    def flattener(input):
        return tree_flatten(input)[0]

    return flattener


def get_hbir_output_unflattener(model: Module) -> callable:
    """Get an callable func to unflatten model output into origin format.

    Args:
        model: Hbir model.
    """
    logger.warning(
        "The pytree support of plugin is deprecated and will be removed "
        "soon, please use native pytree to get better experience.",
        extra={"call_times_context": ("message")},
    )

    info = HbirModuleInfo.read_from(model.functions[0])

    def unflattener(input):
        return tree_unflatten(input, info.output_tree_spec)

    return unflattener


def get_export_hbir_plugin_version(model: Module) -> str:
    """Get the version of plugin used to export input hbir model.

    Args:
        model: Hbir model.
    """
    info = HbirModuleInfo.read_from(model.functions[0])

    return info.plugin_version


class ExporterBase:
    _converters = {}

    @classmethod
    def register_converter(cls, *targets):
        def registor(converter):
            for target in targets:
                if target in cls._converters:
                    msg = (
                        "Cannot register multi converter "
                        "for op type {}".format(target)
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                cls._converters[target] = converter

            return converter

        return registor


class FuncConverterBase:
    with_output_type = False

    @classmethod
    def convert_with_hbir(cls, *args, **kwargs):
        # All tensor inputs are replaced with hbir constant
        # through JitTensor.gather_hbir
        msg = "convert_with_hbir is not implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    @classmethod
    def convert(cls, output: Tensor, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        if cls.with_output_type:
            hbir_output = cls.convert_with_hbir(
                get_hbir_tensor_type(output.as_subclass(Tensor).dtype),
                *hbir_args,
                **hbir_kwargs,
            )
        else:
            hbir_output = cls.convert_with_hbir(*hbir_args, **hbir_kwargs)
        if hbir_output is None:
            return output
        else:
            return JitTensor.attach_hbir_to_tensor(output, hbir_output)


class ModuleConverterBase:
    @classmethod
    def convert_with_hbir(cls, mod, *args, **kwargs):
        # All tensor inputs are replaced with hbir constant
        # through JitTensor.gather_hbir
        msg = "convert_with_hbir is not implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    @classmethod
    def convert(cls, mod, output, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        hbir_output = cls.convert_with_hbir(mod, *hbir_args, **hbir_kwargs)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)

    @classmethod
    def convert_with_constant_folding(cls, mod, output, *args, **kwargs):
        args = DispatchedTensorWrapper.unwrap(args)
        kwargs = DispatchedTensorWrapper.unwrap(kwargs)
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, JitTensor):
                return cls.convert(mod, output, *args, **kwargs)
        # constant derivation
        return output


class RecordSpecTensor(Tensor):
    _msg = (
        "Converting a tensor to a Python {} might cause the trace to be "
        "incorrect. We can't record the data flow of Python values, so this "
        "value will be treated as a constant. This means that the exported "
        "hbir might not generalize to other inputs!"
    )
    _info = "Convert tensor to {}"

    def __int__(self):
        logger.warning(self._msg.format("integer"))
        if ModelChecker.enabled():
            location = LocationManager.get(op="None", update_user_stack=True)
            ModelChecker.add_dynamic_info(
                DynamicInfo(location, self._info.format("integer"))
            )
        return Tensor.__int__(self)

    def __bool__(self):
        logger.warning(self._msg.format("boolean"))
        if ModelChecker.enabled():
            location = LocationManager.get(op="None", update_user_stack=True)
            ModelChecker.add_dynamic_info(
                DynamicInfo(location, self._info.format("boolean"))
            )
        return Tensor.__bool__(self)

    def __float__(self):
        logger.warning(self._msg.format("float"))
        if ModelChecker.enabled():
            location = LocationManager.get(op="None", update_user_stack=True)
            ModelChecker.add_dynamic_info(
                DynamicInfo(location, self._info.format("float"))
            )
        return Tensor.__float__(self)

    @classmethod
    def wrap(cls, input):
        def wrap_record_spec_tensor(x):
            return torch.tensor(x).as_subclass(RecordSpecTensor)

        return pytree_convert(
            input,
            (int, float, bool),
            wrap_record_spec_tensor,
            skip_unsupported=True,
        )

    @classmethod
    def unwrap(cls, input):
        def unwrap_record_spec_tensor(x: RecordSpecTensor):
            return x.item()

        return pytree_convert(
            input,
            RecordSpecTensor,
            unwrap_record_spec_tensor,
            skip_unsupported=True,
            strict_type=True,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all((t is cls for t in types)):
            logger.warning(cls._msg.format("number"))
            if ModelChecker.enabled():
                location = LocationManager.get(op=func, update_user_stack=True)
                ModelChecker.add_dynamic_info(
                    DynamicInfo(location, "Compute with feat")
                )

            return func(*cls.unwrap(args), **cls.unwrap(kwargs))
        else:
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, Tensor):
                if ret.numel() == 1:
                    return ret
                else:
                    # When used as input to Tensor generater, torch.arange
                    # for example, output should be torch.Tensor
                    logger.warning(cls._msg.format("number"))
                    if ModelChecker.enabled():
                        location = LocationManager.get(
                            op=func, update_user_stack=True
                        )
                        ModelChecker.add_dynamic_info(
                            DynamicInfo(location, cls._info.format("number"))
                        )
                    return ret.as_subclass(Tensor)
            else:
                return ret


class JitTensor(ExporterBase, RecordSpecTensor, Tensor):
    _converters: Dict[Any, FuncConverterBase] = {}
    _sub_class_converters: Dict[Any, FuncConverterBase] = {}
    _supported_sub_classes = []
    _jit_enabled = True
    _spec_getter = (
        Tensor.shape.__get__,
        Tensor.ndim.__get__,
        Tensor.size,
        Tensor.dim,
    )
    # If enabled, unsupported ops will be converted to hbtl.call
    # to get a complete graph.
    _allow_fall_through = False

    def __new__(
        cls, base: Tensor, hbir_node, subclass_from=None, check_shape=True
    ):
        if isinstance(base, cls):
            base = base._base
        return base.as_subclass(cls)

    def __init__(
        self, base: Tensor, hbir_node, subclass_from=None, check_shape=True
    ):
        if (
            check_shape
            and hasattr(hbir_node, "type")
            and hasattr(hbir_node.type, "shape")
            and list(base.shape) != hbir_node.type.shape
        ):
            raise RuntimeError(
                "Hbir output shape is different with actual torch output "
                "({} vs {}), please report a bug to horizon_plugin_pytorch "
                "develop team".format(hbir_node.type.shape, base.shape)
            )

        self.hbir_node = hbir_node
        self.subclass_from = subclass_from
        if isinstance(base, JitTensor):
            self.__base = base._base
        else:
            self.__base = base

    @property
    def _base(self):
        return self.__base

    @classmethod
    def register_subclass_converter(cls, *targets):
        def registor(converter):
            for target in targets:
                if target in cls._sub_class_converters:
                    msg = (
                        "Cannot register multi converter "
                        "for op type {}".format(target)
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                cls._sub_class_converters[target] = converter

            return converter

        return registor

    @classmethod
    def gather_hbir(cls, tensors, make_constant=True):
        def gather_hbir_from_single_tensor(x: Tensor):
            if isinstance(x, JitTensor):
                return x.hbir_node
            else:
                x = x.as_subclass(Tensor)
                if make_constant:
                    return hbir.constant(
                        to_numpy(x),
                        output_type=get_hbir_tensor_type(x.dtype),
                    )
                else:
                    return x

        hbir_nodes = pytree_convert(
            tensors,
            Tensor,
            gather_hbir_from_single_tensor,
            skip_unsupported=True,
        )
        logger.debug(
            "Gathered hbir nodes {}".format(tensor_struct_repr(hbir_nodes))
        )

        return hbir_nodes

    @classmethod
    def get_base(cls, tensors):
        def get_base_from_single_tensor(x: Tensor):
            if isinstance(x, JitTensor):
                return x._base
            else:
                return x

        base_tensors = pytree_convert(
            tensors,
            Tensor,
            get_base_from_single_tensor,
            skip_unsupported=True,
        )
        logger.debug(
            "Gathered base tensors {}".format(tensor_struct_repr(base_tensors))
        )

        return base_tensors

    @classmethod
    def attach_hbir_to_tensor(cls, tensors, hbir_nodes, check_shape=True):
        if isinstance(tensors, DispatchedTensorWrapper) and not isinstance(
            hbir_nodes, (list, tuple, ir.OpResultList)
        ):
            tensors._t = JitTensor(
                tensors._t, hbir_nodes, check_shape=check_shape
            )
            return tensors
        if isinstance(tensors, Tensor) and not isinstance(
            hbir_nodes, (list, tuple, ir.OpResultList)
        ):
            return JitTensor(tensors, hbir_nodes, check_shape=check_shape)
        if isinstance(tensors, (list, tuple)) and isinstance(
            hbir_nodes, (list, tuple, ir.OpResultList)
        ):
            return type(tensors)(
                cls.attach_hbir_to_tensor(t, hb_op, check_shape)
                for t, hb_op in zip(tensors, hbir_nodes)
            )
        msg = (
            "Unsupported pair when constructing JitTensor: {} and {}.".format(
                type(tensors), type(hbir_nodes)
            )
        )
        logger.error(msg)
        raise ValueError(msg)

    def as_subclass(self, cls):
        if cls is DispatchedTensorWrapper:
            return self._base.as_subclass(cls)
        if self._jit_enabled:
            # Store subclass_from in the new JitTensor, if a torch operation
            # is inplaced, the converter should modify hbir_node under
            # new JitTensor and subclass_from !!!
            if self.subclass_from is not None:
                subclass_from = self.subclass_from
            else:
                subclass_from = self
            return JitTensor(
                self._base.as_subclass(cls), self.hbir_node, subclass_from
            )
        else:
            return self._base.as_subclass(cls)

    @classmethod
    def attach_inplaced_output(cls, tensor, node):
        if isinstance(tensor, cls):
            tensor.hbir_node = node
            if tensor.subclass_from is not None:
                tensor.subclass_from.hbir_node = node
        else:
            fake_base = copy.deepcopy(tensor)
            tensor.__class__ = cls
            tensor.hbir_node = node
            tensor.subclass_from = None
            tensor.__base = fake_base

    @classmethod
    def get_main_type(cls, args, kwargs):
        ret = Tensor
        for x in tree_flatten((args, kwargs))[0]:
            if isinstance(x, DispatchedTensorWrapper):
                return DispatchedTensorWrapper
            if isinstance(x, JitTensor) and type(x._base) is not Tensor:
                ret = type(x._base)
        return ret

    def to_fake_quantized(self):
        return self._base.to_fake_quantized()

    def to_quantized(self):
        return self._base.to_quantized()

    class HBDKConverter(FuncConverterBase):
        @classmethod
        def is_hbdk_kernel(cls, func):
            return hasattr(func, "__name__") and func.__name__ in (
                "cpp_custom_wrapper",
                "leap_op_wrapper",
                "triton_export_wrapper",
            )

        @classmethod
        def is_triton_kernel(cls, func):
            return (
                hasattr(func, "__name__")
                and func.__name__ == "triton_export_wrapper"
            )

        @classmethod
        def is_leap_kernel(cls, func):
            return (
                hasattr(func, "__name__")
                and func.__name__ == "leap_op_wrapper"
            )

        @classmethod
        def convert(cls, output, *args, **kwargs):
            func = kwargs.pop("hbdk4_custom_op")

            # Modify kwargs to indicate that hbdk should retuen a partial
            # function to generate a hbir node.
            kwargs["is_horizon_plugin_pytorch_export"] = True

            args_base, kwargs_base = JitTensor.get_base((args, kwargs))
            hbir_generator = func(*args_base, **kwargs_base)

            # Generate hbir.
            hbir_args = JitTensor.gather_hbir(args)
            if cls.is_leap_kernel(func):
                # For leap kernel, pass args same as tensor format
                hbir_output = hbir_generator(*hbir_args)
            else:
                # For triton and cpp, pass args as tuple
                hbir_output = hbir_generator(hbir_args)

            # Deal with list/tuple.
            if isinstance(hbir_output, (list, tuple, ir.OpResultList)):
                if isinstance(output, Tensor):
                    hbir_output = hbir_output[0]
            elif isinstance(output, (list, tuple)):
                hbir_output = [hbir_output]

            if output is None:
                return output
            else:
                return JitTensor.attach_hbir_to_tensor(output, hbir_output)

    class FallthroughConverter(FuncConverterBase):
        """Convert unsupported ops to hbtl.call to get a complete graph."""

        @classmethod
        def convert(cls, func, output, *args, **kwargs):
            from hbdk4.compiler import hbtl as hbtl_ops
            from hbdk4.compiler.ops import hbtl

            from horizon_plugin_pytorch.utils.location_info import (
                TorchLocationInfo,
            )

            args, _ = tree_flatten((args, kwargs))

            hbtl_args = []
            hbtl_call_kwargs = {}
            for arg in args:
                if hbtl_ops.is_tensor_like(arg):
                    hbtl_args.append(hbtl_ops.get_tensor_type(arg, True))
                else:
                    if "parameters" not in hbtl_call_kwargs.keys():
                        hbtl_call_kwargs["parameters"] = []
                    if arg is None:
                        arg = "None"
                    hbtl_call_kwargs["parameters"].append(arg)
                    hbtl_args.append(arg)

            hbir_args = JitTensor.gather_hbir(
                tuple(x for x in args if isinstance(x, Tensor))
            )

            if output is None:
                outputs_type = [hbir_args[0].type]
            else:
                hbir_const_output = JitTensor.gather_hbir(output)
                if not isinstance(hbir_const_output, (list, tuple)):
                    outputs_type = [hbir_const_output.type]
                else:
                    outputs_type = [x.type for x in hbir_const_output]

            hbtl_call_kwargs["outputs_type"] = outputs_type

            input_num = len(hbir_args)
            output_num = 1 if output is None else len(tree_flatten(output)[0])

            func_name = TorchLocationInfo.format_op_name(func)
            hbir_output = hbtl.call(
                hbir_args,
                "{}({}) -> ({})".format(
                    func_name.replace(".", "::"),
                    ", ".join((f"ARG{i}" for i in range(input_num))),
                    ", ".join((f"RET{i}" for i in range(output_num))),
                ),  # signature
                isCustom=True,
                **hbtl_call_kwargs,
            )

            if output is None:
                for x in args:
                    if isinstance(x, JitTensor):
                        x.hbir_node = hbir_output[0]
                        return output
                raise RuntimeError(
                    "Cannot export {}{}, please report to "
                    "horizon_plugin_pytorch develop team.".format(
                        func_name, args
                    )
                )
            else:
                if isinstance(output, Tensor):
                    hbir_output = hbir_output[0]

                return JitTensor.attach_hbir_to_tensor(output, hbir_output)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        all_types = (
            types
            + tuple((type(x) for x in tree_flatten(args)[0]))
            + tuple((type(x) for x in tree_flatten(kwargs)[0]))
        )

        if RecordSpecTensor in all_types:
            logger.warning(RecordSpecTensor._msg.format("number"))
            if ModelChecker.enabled():
                location = LocationManager.get(op=func, update_user_stack=True)
                ModelChecker.add_dynamic_info(
                    DynamicInfo(location, "Compute with feat")
                )
            args = RecordSpecTensor.unwrap(args)
            kwargs = RecordSpecTensor.unwrap(kwargs)

        main_type = cls.get_main_type(args, kwargs)

        # JitTensor should be inside DispatchedTensorWrapper, so torch usually
        # call DispatchedTensorWrapper.__torch_function__ first.
        # Except when:
        # operation(x: JitTensor, y: DispatchedTensorWrapper)
        # x can not be wrapped because its not a module output (bool input for
        # example).
        # In this case, we should jump to
        # DispatchedTensorWrapper.__torch_function__, and
        # JitTensor.__torch_function__ may be called inside it with
        # DispatchedTensorWrapper unwrapped.
        if main_type is DispatchedTensorWrapper:
            return DispatchedTensorWrapper.__torch_function__(
                func, types, args, kwargs
            )

        if cls.HBDKConverter.is_triton_kernel(func):
            # Not need to run the triton op, because it has no output
            output = None
        else:
            output = func(*cls.get_base(args), **cls.get_base(kwargs))

        if cls._jit_enabled:
            if func in cls._spec_getter:
                return RecordSpecTensor.wrap(output)
            if output is None and "out" in kwargs:
                output = kwargs.get("out")

            if cls.HBDKConverter.is_hbdk_kernel(func):
                converter = cls.HBDKConverter
                kwargs["hbdk4_custom_op"] = func
            elif main_type is Tensor:
                converter = cls._converters.get(func, None)
            else:
                converter = cls._sub_class_converters.get(func, None)

            location = LocationManager.get(op=func, update_user_stack=True)

            try:
                if converter is None:
                    if isinstance(output, Tensor) or (
                        isinstance(output, (list, tuple))
                        and isinstance(output[0], Tensor)
                    ):
                        # TODO: Add converter for op with no on-board logic
                        msg = (
                            "Function {} on type {} is not supported "
                            "in hbir exporting".format(func, main_type)
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)
                    else:
                        logger.debug("Trace func {} as constant".format(func))
                        # treat Tensor.device ... etc as constant
                        return output
                else:
                    with JitTensor.disable_jit(), ir.Location.fused(
                        [], TrackAttr.get(location.to_dict())
                    ):
                        logger.debug(
                            "Convert func {} with {}".format(func, converter)
                        )
                        output = converter.convert(
                            output,
                            *args,
                            **kwargs,
                        )
            except Exception as e:
                if ModelChecker.enabled():
                    ModelChecker.add_op_item(
                        CheckInfoItem(
                            location, " ".join((str(arg) for arg in e.args))
                        )
                    )
                    if output is not None:
                        output = JitTensor.attach_hbir_to_tensor(
                            output, JitTensor.gather_hbir(output)
                        )
                elif cls._allow_fall_through:
                    converter = cls.FallthroughConverter
                    with JitTensor.disable_jit(), ir.Location.fused(
                        [], TrackAttr.get(location.to_dict())
                    ):
                        logger.debug(
                            "Convert func {} with {}".format(func, converter)
                        )
                        output = converter.convert(
                            func,
                            output,
                            str(e),
                            *args,
                            **kwargs,
                        )
                else:
                    raise e
            else:
                if ModelChecker.enabled():
                    ModelChecker.add_op_item(CheckInfoItem(location))

        return output

    @classmethod
    def get_hbir_input_nodes(cls, tensors):
        def gen_hbir_from_tensor(x: Tensor):
            return ir.RankedTensorType.get(
                list(x.shape), get_hbir_dtype(x.dtype)
            )

        tensor_list, _ = tree_flatten(tensors)
        hbir_nodes = []
        for t in tensor_list:
            if isinstance(t, Tensor):
                hbir_nodes.append(gen_hbir_from_tensor(t))

        return hbir_nodes

    @classmethod
    @contextmanager
    def enable_jit(cls, v=True):
        try:
            old_state = cls._jit_enabled
            cls._jit_enabled = v
            if old_state != v:
                if old_state:
                    logger.debug("Trace disabled")
                else:
                    logger.debug("Trace enabled")
            yield
        finally:
            cls._jit_enabled = old_state
            if old_state != v:
                if old_state:
                    logger.debug("Trace enabled")
                else:
                    logger.debug("Trace disabled")

    @classmethod
    def disable_jit(cls):
        return cls.enable_jit(False)

    @classmethod
    def without_jit(cls, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            with JitTensor.disable_jit():
                return func(*args, **kwargs)

        return wrapped_func

    @classmethod
    @contextmanager
    def hack_isinstance(cls):
        builtin_isinstance = builtins.isinstance
        try:

            def jit_isinstance(obj, class_or_tuple):
                if builtin_isinstance(obj, class_or_tuple):
                    return True
                elif builtin_isinstance(obj, JitTensor):
                    return builtin_isinstance(obj._base, class_or_tuple)
                return False

            builtins.isinstance = jit_isinstance
            yield
        finally:
            builtins.isinstance = builtin_isinstance

    @classmethod
    def register_subclass(cls, subcls):
        if subcls not in cls._supported_sub_classes:
            cls._supported_sub_classes.append(subcls)

    @classmethod
    @contextmanager
    def patch_subclasses(cls):
        sub_news = []
        patched_news = []
        for subcls in cls._supported_sub_classes:
            sub_new = subcls.__new__
            sub_news.append(sub_new)

            def subclass_new_with_jit_tensor(cls, data, *args, **kwargs):
                if isinstance(data, JitTensor):
                    return JitTensor(
                        sub_new(cls, data._base, *args, **kwargs),
                        data.hbir_node,
                    )
                else:
                    return sub_new(cls, data, *args, **kwargs)

            patched_news.append(subclass_new_with_jit_tensor)

        try:
            for subcls, new_with_jit in zip(
                cls._supported_sub_classes, patched_news
            ):
                subcls.__new__ = new_with_jit
            yield
        finally:
            for subcls, ori_new in zip(cls._supported_sub_classes, sub_news):
                subcls.__new__ = ori_new


class Exporter(ExporterBase):
    _converters = {}
    _is_exporting = False

    @classmethod
    def gen_hbir_hook(cls, name: str, converter: ModuleConverterBase):
        # define gen_hbir_hook method to hold arg converter
        def _record_hbir_hook(mod, args, output):
            location = LocationManager.get(mod)
            with JitTensor.disable_jit(), ir.Location.fused(
                [], TrackAttr.get(location.to_dict())
            ):
                logger.debug(
                    "Gen hbir of {}, type {} with {}".format(
                        name, type(mod), converter.__name__
                    )
                )
                logger.debug("Mod inputs {}".format(tensor_struct_repr(args)))
                logger.debug(
                    "Mod outputs {}".format(tensor_struct_repr(output))
                )
                try:
                    output = converter.convert_with_constant_folding(
                        mod, output, *args
                    )
                except Exception as e:
                    if ModelChecker.enabled():
                        ModelChecker.add_op_item(
                            CheckInfoItem(
                                location,
                                " ".join((str(arg) for arg in e.args)),
                            )
                        )
                        output = JitTensor.attach_hbir_to_tensor(
                            output, JitTensor.gather_hbir(output)
                        )
                    else:
                        raise e
                else:
                    if ModelChecker.enabled():
                        ModelChecker.add_op_item(CheckInfoItem(location))
            return output

        return _record_hbir_hook

    @classmethod
    def export(
        cls,
        model: nn.Module,
        example_inputs,
        check_model: bool,
        *,
        name: str = "forward",
        input_names=None,
        output_names=None,
        input_descs=None,
        output_descs=None,
        native_pytree: bool = True,
        _get_raw_graph=False,
        **checker_kwargs,
    ):
        if isinstance(model, torch.jit.ScriptModule):
            raise ValueError(
                "Input model must be a torch.nn.Module, "
                "ScriptModule is not supported"
            )

        if LooseVersion(get_hbdk4_version()) <= LooseVersion("4.1.11"):
            raise RuntimeError(
                "Please install hbdk4_compiler>4.1.11, or the computation "
                "between plugin and hbdk4 will be inconsistent"
            )

        cls._is_exporting = True

        swap_nn_with_horizonnn(model)
        model.eval()
        set_fake_quantize(model, FakeQuantState.VALIDATION)

        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = (example_inputs,)

        handles = {}

        def has_fp16_process(model):
            if is_float(model):
                return False
            for _, mod in model.named_modules():
                if is_float(mod):
                    return True
            return False

        def attach_hook(model: nn.Module, prefix=""):
            converter = cls._converters.get(type(model), None)
            if converter is not None and not has_fp16_process(model):
                logger.debug(
                    "Hook mod {}, type {} with {}".format(
                        prefix, type(model), converter.__name__
                    )
                )
                # This hook must be inserted before DispatchedTensorWrapper's,
                # because JitTensor should be inside the wrapper.
                handles[prefix] = register_forward_hook(
                    model, cls.gen_hbir_hook(prefix, converter), prepend=True
                )

                # disable JitTensor in module forward
                model.forward = JitTensor.without_jit(model.forward)
            else:
                for n, m in model.named_children():
                    attach_hook(m, prefix + ("." if prefix else "") + n)

        attach_hook(model)

        flat_inputs, input_spec = tree_flatten(example_inputs)
        output_specs = []
        output_annotations = []

        def gather_hbir_outputs(nodes):
            ret = []
            for node in nodes:
                if isinstance(node, Tensor):
                    logger.warning(
                        "Detect output without node info, which means output"
                        " number of hbir is not same with python code."
                    )
                else:
                    ret.append(node)

            return ret

        def gather_annotation(t):
            if hasattr(t, "annotation"):
                return t.annotation
            else:
                return None

        def forward(*hbir_inputs):
            jit_inputs = []

            hbir_idx = 0
            for t in flat_inputs:
                if isinstance(t, Tensor):
                    jit_inputs.append(
                        JitTensor.attach_hbir_to_tensor(
                            t, hbir_inputs[hbir_idx]
                        )
                    )
                    hbir_idx += 1
                else:
                    jit_inputs.append(t)

            assert hbir_idx == len(hbir_inputs)

            structured_input = tree_unflatten(jit_inputs, input_spec)

            with suppress_type_checks(), LocationManager(
                model
            ), JitTensor.hack_isinstance(), JitTensor.patch_subclasses():
                ret = model(*_as_tuple(structured_input))
                logger.info(
                    "Model ret: {}".format(
                        tensor_struct_repr(JitTensor.get_base(ret))
                    )
                )

            flat_outputs, output_spec = tree_flatten(ret)
            output_specs.append(output_spec)
            hbir_outputs = JitTensor.gather_hbir(flat_outputs)

            for x in flat_outputs:
                output_annotations.append(gather_annotation(x))

            hbir_rets = gather_hbir_outputs(hbir_outputs)

            return hbir_rets

        module = ir.Module.create()
        checker = ModelChecker(check_model, **checker_kwargs)
        with ir.InsertionPoint(module.body), checker:
            hbir_inputs = JitTensor.get_hbir_input_nodes(flat_inputs)
            FuncOp.from_py_func(*hbir_inputs, name=name)(forward)

        for h in handles.values():
            h.remove()

        if check_model:
            return checker

        logger.debug(
            "Hbir Module:\n{}".format(
                module.operation.get_asm(
                    enable_debug_info=True, pretty_debug_info=True
                )
            )
        )

        if _get_raw_graph:
            return module

        module = Module(module)

        from horizon_plugin_pytorch.utils.global_quant_round_mode import (
            QuantRoundMode,
        )

        if QuantRoundMode.get() == QuantRoundMode.BPU_ROUND:
            module._legacy_round = True

        func = module.functions[0]
        try:
            import horizon_plugin_pytorch

            HbirModuleInfo(
                output_specs[0], horizon_plugin_pytorch.__version__
            ).write_to(func)
        except Exception as e:
            logger.warning(
                "Failed to serialize output pytree info, "
                "get_hbir_output_unflattener will not work on exported "
                "hbir model!\nOrigin exception is {}".format(e.args)
            )

        if input_names is not None:
            for input, name in zip(func.inputs, tree_flatten(input_names)[0]):
                if name is not None:
                    input.name = name
        if output_names is not None:
            for output, name in zip(
                func.outputs, tree_flatten(output_names)[0]
            ):
                if name is not None:
                    output.name = name
        if input_descs is not None:
            for input, desc in zip(func.inputs, tree_flatten(input_descs)[0]):
                if desc is not None:
                    input.desc = desc
        for output, annot in zip(func.outputs, output_annotations):
            if annot is not None:
                output.desc = annot
        if output_descs is not None:
            for output, desc in zip(
                func.outputs, tree_flatten(output_descs)[0]
            ):
                if desc is not None:
                    output.desc = desc

        if native_pytree:
            if not hasattr(module.functions[0], "_in_tree_spec"):
                raise ValueError(
                    "Current hbdk4 version {} does not support native_pytree, "
                    "please update hbdk4 or use native_pytree=False".format(
                        get_hbdk4_version()
                    )
                )
            module.functions[0]._in_tree_spec = input_spec
            module.functions[0]._out_tree_spec = output_specs[0]

        cls._is_exporting = False
        return module
