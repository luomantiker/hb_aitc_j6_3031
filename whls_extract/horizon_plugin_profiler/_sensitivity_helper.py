import copy
import logging
import os
from typing import Dict, Sequence

from horizon_plugin_profiler.utils.bc_helper import get_single_bc_error
from horizon_plugin_profiler.utils.location_info import (
    LocationManager,
    TorchLocationInfo,
)
from horizon_plugin_profiler.utils.model_helper import (
    HookAndTorchFunctionHelper,
    _as_tuple,
    _set_attr,
    apply_to_collection,
    find_leaf_modules,
    get_device,
    pytree_convert,
)

import torch
from tabulate import tabulate
from torch import Tensor
from torch.nn import functional as F  # noqa N812
from torch.quantization.fake_quantize import FakeQuantizeBase
from torch.utils._pytree import tree_flatten
from tqdm import tqdm

from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.march import set_march
from horizon_plugin_pytorch.nn.grid_sample import warp
from horizon_plugin_pytorch.nn.qat import GELU
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import convert, convert_fx, prepare
from horizon_plugin_pytorch.quantization.fake_quantize import (
    FakeQuantState,
    set_fake_quantize,
)
from horizon_plugin_pytorch.quantization.qconfig_template import (
    ModuleNameQconfigSetter,
)

try:
    from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
        DispatchedTensorWrapper,
    )
except ImportError:
    DispatchedTensorWrapper = None
from ._similarity import _compute_similarity, _func_map
from .set_preserve_qat_mode import set_preserve_qat_mode

logger = logging.getLogger(__name__)


class FloatOpLoader:
    """Loaders of float ops.

    Manage loaders of ops that need to use float results or keep float behavior
    when computing sensitivity.
    """

    _FLOAT_OP_LOADER_MAP = {}

    @classmethod
    def register(cls, op):
        def wrapper(func):
            if isinstance(op, Sequence):
                for each_op in op:
                    key = (
                        each_op.__class__
                        if isinstance(each_op, torch.nn.Module)
                        else each_op
                    )
                    assert key not in cls._FLOAT_OP_LOADER_MAP
                    cls._FLOAT_OP_LOADER_MAP[key] = func
            else:
                key = op.__class__ if isinstance(op, torch.nn.Module) else op
                assert key not in cls._FLOAT_OP_LOADER_MAP
                cls._FLOAT_OP_LOADER_MAP[key] = func
            return func

        return wrapper

    @classmethod
    def dispatch(cls, mod):
        if isinstance(mod, torch.nn.Module):
            mod = mod.__class__
        loader = cls._FLOAT_OP_LOADER_MAP.get(mod, None)
        if loader is None:
            name = TorchLocationInfo.format_op_name(mod)
            raise RuntimeError(f"No register loader run op {name}")
        return loader

    @classmethod
    def has_registered(cls, mod):
        if isinstance(mod, torch.nn.Module):
            return mod.__class__ in cls._FLOAT_OP_LOADER_MAP
        return mod in cls._FLOAT_OP_LOADER_MAP


@FloatOpLoader.register(
    (
        torch.topk,
        torch.sort,
        torch._C._TensorBase.topk,
        torch._C._TensorBase.sort,
    )
)
def _load_sorted_like(func, float_ret, qat_ret, args, kwargs):
    """Use float index to get results."""
    assert float_ret is not None, f"Can not find float {func} results."
    return_type_map = {
        torch.topk: torch.return_types.topk,
        torch.sort: torch.return_types.sort,
        torch._C._TensorBase.topk: torch.return_types.topk,
        torch._C._TensorBase.sort: torch.return_types.sort,
    }
    _, index = float_ret
    origin_qat_ret = qat_ret[0]

    if "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        if func in (torch.topk, torch.Tensor.topk):
            if len(args) >= 3:
                dim = args[2]
            else:
                dim = -1
        else:
            if len(args) >= 2:
                dim = args[1]
            else:
                dim = -1

    ret = args[0].as_subclass(torch.Tensor).gather(dim=dim, index=index)
    ret = return_type_map[func](
        [
            QTensor(
                ret,
                origin_qat_ret.q_scale(),
                origin_qat_ret.dtype,
                origin_qat_ret.q_per_channel_axis(),
            )
            if isinstance(origin_qat_ret, QTensor)
            else ret,
            index,
        ]
    )
    return ret


@FloatOpLoader.register((torch.argmax, torch._C._TensorBase.argmax))
def _load_argmax(func, float_ret, qat_ret, args, kwargs):
    # just return float model index here. It must be a int index tensor
    assert float_ret is not None, "Can not find float argmax results."
    return float_ret


@FloatOpLoader.register(
    (torch.max, torch.Tensor.max, torch.min, torch.Tensor.min)
)
def _load_max(func, float_ret, qat_ret, args, kwargs):
    assert float_ret is not None, f"Can not find float {func} results."
    # torch.min / max directory return tensor result
    if isinstance(float_ret, Tensor):
        return qat_ret

    index = float_ret[1]
    origin_qat_ret = qat_ret[0]

    if "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        if len(args) >= 2:
            dim = args[1]

    keepdim = False
    if "keepdim" in kwargs:
        keepdim = kwargs["keepdim"]
    else:
        if len(args) >= 3:
            keepdim = args[2]

    ret = args[0].as_subclass(torch.Tensor)
    if keepdim:
        ret = ret.gather(dim=dim, index=index)
    else:
        ret = ret.gather(dim=dim, index=index.unsqueeze(dim)).squeeze(dim)

    return_types = (
        torch.return_types.max
        if func in (torch.max, torch.Tensor.max)
        else torch.return_types.min
    )
    ret = return_types(
        [
            QTensor(
                ret,
                origin_qat_ret.q_scale(),
                origin_qat_ret.dtype,
                origin_qat_ret.q_per_channel_axis(),
            )
            if isinstance(origin_qat_ret, QTensor)
            else ret,
            index,
        ]
    )
    return ret


@FloatOpLoader.register(
    (
        F.grid_sample,
        F.interpolate,
        warp,
        torch.clamp,
        torch._C._TensorBase.clamp,
        torch.clip,
        torch._C._TensorBase.clip,
    )
)
def _load_func(func, float_ret, qat_ret, args, kwargs):
    """Use float grid sample with qat args."""
    ret = func(
        *pytree_convert(args, QTensor, lambda x: x.as_subclass(Tensor)),
        **pytree_convert(kwargs, QTensor, lambda x: x.as_subclass(Tensor)),
    )
    return (
        QTensor(
            ret, qat_ret.q_scale(), qat_ret.dtype, qat_ret.q_per_channel_axis()
        )
        if isinstance(qat_ret, QTensor)
        else ret
    )


class FloatOpsRecorder:
    """Manage float ops results."""

    def __init__(self):
        self.records = {}

    @staticmethod
    def format_key(loc):
        return loc.mod_name + (
            f".{loc.op_name}" if loc.op_type == "call_function" else ""
        )

    def add(self, key, idx, value):
        self.records[key + f"_{idx}"] = value

    def get(self, key, idx):
        key = key + f"_{idx}"
        if key not in self.records:
            return None
        return self.records[key]


class SensitiveHelper(HookAndTorchFunctionHelper):
    """Getting graph with module hook and __torch_function__ of tensor."""

    _current = None
    _model_type = "float"
    _enable_float_loader = True

    class TracerTensor(HookAndTorchFunctionHelper.TracerTensor):
        """Patch Tensor for running sensitivity."""

        # SensitiveHelper.TraceTensor postprocess should only run once here
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            ori_args, ori_kwargs = args, kwargs

            func, types, args, kwargs = cls._torch_function_preprocess(
                func, types, args, kwargs
            )

            if DispatchedTensorWrapper is not None and any(
                [
                    isinstance(x, DispatchedTensorWrapper)
                    for x in tree_flatten((args, kwargs))[0]
                ]
            ):
                rewrapped_args = pytree_convert(
                    args,
                    DispatchedTensorWrapper,
                    cls.insert_jit_tensor_into_wrapper,
                )
                rewrapped_kwargs = pytree_convert(
                    kwargs,
                    DispatchedTensorWrapper,
                    cls.insert_jit_tensor_into_wrapper,
                )
                func_ret = DispatchedTensorWrapper.__torch_function__(
                    func, types, rewrapped_args, rewrapped_kwargs
                )
                func_ret = pytree_convert(
                    func_ret,
                    DispatchedTensorWrapper,
                    cls.del_jit_tensor_in_wrapper,
                )
                func_ret = cls.wrap(func_ret)
            else:
                func_ret = func(*args, **kwargs)
                func_ret = cls._torch_function_postprocess(
                    func, types, ori_args, ori_kwargs, func_ret
                )

            return func_ret

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__.

            Save registered torch func results of float model and load float
            result when running qat model.
            """
            if not SensitiveHelper.is_loader_enabled():
                return func_ret
            if SensitiveHelper.is_tracing() and FloatOpLoader.has_registered(
                func
            ):
                loc = LocationManager.get(func)
                key = FloatOpsRecorder.format_key(loc)
                idx = SensitiveHelper._current.op_appear_times.get(key, 0)
                SensitiveHelper._current.op_appear_times[key] = idx + 1

                if SensitiveHelper.is_float_model():
                    # save float model ops results
                    SensitiveHelper._current.info_manager.add(
                        key,
                        idx,
                        cls.unwrap(func_ret),
                    )
                else:
                    # load float model ops results
                    float_ret = SensitiveHelper._current.info_manager.get(
                        key, idx
                    )
                    func_ret = FloatOpLoader.dispatch(func)(
                        func,
                        float_ret,
                        func_ret,
                        cls.unwrap(args),
                        cls.unwrap(kwargs),
                    )
            return cls.wrap(func_ret)

    def __init__(self, model: torch.nn.Module, with_stack_info=False) -> None:
        self.model = model
        self.op_appear_times = {}
        self.info_manager: FloatOpsRecorder = None
        super(SensitiveHelper, self).__init__(with_stack_info)

    @classmethod
    def current(cls, *args, **kwargs):
        return cls._current

    @classmethod
    def is_tracing(cls):
        return cls._current is not None

    @classmethod
    def is_loader_enabled(cls):
        return cls._enable_float_loader

    @classmethod
    def is_float_model(cls):
        return cls._model_type == "float"

    @classmethod
    def set_model_type(cls, model_type):
        cls._model_type = model_type

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook.

        Save float module and results of float modules and load float results
        when running qat model.
        """
        if not self.is_loader_enabled():
            return output
        if FloatOpLoader.has_registered(mod):
            loc = LocationManager.get(mod)
            key = FloatOpsRecorder.format_key(loc)
            idx = self.op_appear_times.get(key, 0)
            self.op_appear_times[key] = idx + 1
            if self.is_float_model():
                self.info_manager.add(key, idx, (mod, output))
            else:
                ori_output = output
                if DispatchedTensorWrapper is not None:
                    output = DispatchedTensorWrapper.unwrap(output)
                float_mod, float_ret = self.info_manager.get(key, idx)
                output = FloatOpLoader.dispatch(mod)(
                    float_mod,
                    mod,
                    float_ret,
                    output,
                    self.TracerTensor.unwrap_to_origin(args),
                    self.TracerTensor.unwrap_to_origin(kwargs),
                )
                if DispatchedTensorWrapper is not None and isinstance(
                    ori_output, DispatchedTensorWrapper
                ):
                    output = DispatchedTensorWrapper.wrap(output)
        return self.TracerTensor.wrap(output)

    def __enter__(self, *args, **kwargs):
        self.old_obj = SensitiveHelper._current
        SensitiveHelper._current = self

        self.info_manager = FloatOpsRecorder()
        super().__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__()
        SensitiveHelper._current = self.old_obj

    def get_info_manager(self):
        return self.info_manager

    def set_info_manager(self, info_manager):
        self.info_manager = info_manager

    def reset_op_appear_times(self):
        self.op_appear_times = {}


def _sensitive_worker(
    device_id,
    module_info,
    ops,
    data,
    out_index,
    enable_log,
    queue,
    march,
):
    """Run sensitive func on each process. Only used in multi gpu sensitivity.

    This function converts float model to qat model, construct and run
    sensitivity on device i. The sensitive table will be put on multiprocessing
    queue to be gather in main process.

    Args:
        i: device id
        module_info: infos to convert model and construct sensitive model
        ops: leaf ops to be computed
        data: model input
        out_index: output index
        enable_log: whether enable sensitive running log
        queue: multiprocessing queue to gather sensitive table
        march: BPU march
    """
    set_march(march)
    device = get_device(device_id)
    (
        baseline_model,
        qconfig_dict,
        state_dict,
        baseline_model_cached_attrs,
        analysis_model_cached_attrs,
        analysis_model_type,
        metric,
        op_flops_mapping,
        prepare_method,
    ) = module_info
    qat_model = copy.deepcopy(baseline_model)

    baseline_model_cached_attrs = apply_to_collection(
        baseline_model_cached_attrs, Tensor, lambda x: x.to(device)
    )
    analysis_model_cached_attrs = apply_to_collection(
        analysis_model_cached_attrs, Tensor, lambda x: x.to(device)
    )
    example_input = copy.deepcopy(data)
    example_input = apply_to_collection(
        example_input, Tensor, lambda x: x.to(device)
    )
    for k, v in baseline_model_cached_attrs.items():
        _set_attr(qat_model, k, v)

    qat_model = prepare(
        qat_model.eval().to(device),
        example_inputs=example_input,
        qconfig_setter=ModuleNameQconfigSetter(qconfig_dict),
        method=prepare_method,
    )
    qat_model.load_state_dict(state_dict, strict=False)
    module = Sensitivity(
        baseline_model,
        qat_model,
        baseline_model_cached_attrs,
        analysis_model_cached_attrs,
        analysis_model_type,
        metric,
        device,
        op_flops_mapping,
    )
    module._enable_log = enable_log
    module = module.to(device)
    data = apply_to_collection(data, Tensor, lambda x: x.to(device))
    module(data, ops, out_index)
    queue.put(module.sensitivity_table)


class SensitivityTable:
    """Manage sensitivity results.

    Manage sensitivity results in a table. Each item is a list of 5 elements:
    1. op_name
    2. sensitive_type(activation or weight)
    3. quant dtype
    4. op type
    5. sensitive value in current metric

    Args:
        metric: Sensitivity compute metric
    """

    def __init__(self, metric: str):
        self.result = []
        self.metric = metric

    def compute(self, baseline_ret, analysis_ret):
        """Compute each output sensitivity."""
        baseline_ret = tree_flatten(baseline_ret)[0]
        analysis_ret = tree_flatten(analysis_ret)[0]
        simi = _compute_similarity(
            baseline_ret, analysis_ret, _func_map[self.metric]
        )
        simi = [
            x.item() if isinstance(x, Tensor) else x
            for x in simi
            if x is not None
        ]
        return simi

    def add(
        self,
        baseline_ret,
        analysis_ret,
        index,
        op,
        attr,
        op_type,
        qdtype,
        flops,
    ):
        """Compute and add sensitivity result."""
        simi = self.compute(baseline_ret, analysis_ret)
        assert index < len(simi)
        self.result.append([op, attr, op_type, simi[index], qdtype, flops])

    def sort(self, reverse):
        """Sort sensitivity result."""
        descending = False if self.metric in ("Cosine", "SQNR") else True
        descending = not descending if reverse else descending
        self.result = sorted(
            self.result, key=lambda x: x[3], reverse=descending
        )

    def get(self):
        return self.result

    def dump(self, path, out_name):
        """Dump sensitivity results into txt and pt file."""
        txtname = os.path.join(
            path, f"output_{out_name}_{self.metric}_sensitive_ops.txt"
        )
        with open(txtname, "w") as f:
            f.write(
                tabulate(
                    self.result,
                    headers=[
                        "op_name",
                        "sensitive_type",
                        "op_type",
                        self.metric,
                        "quant_dtype",
                        "flops",
                    ],
                )
            )
        ptname = os.path.join(
            path, f"output_{out_name}_{self.metric}_sensitive_ops.pt"
        )
        torch.save(self.result, ptname)

    @classmethod
    def merge(cls, tables):
        """Merge several tables and return a new SensitivityTable."""
        assert (
            len({t.metric for t in tables}) == 1
        ), "All tables must have the same metric"
        merge_table = cls(metric=tables[0].metric)
        for t in tables:
            merge_table.result += t.result
        return merge_table


class Sensitivity(torch.nn.Module):
    """Model for computing op sensitivity.

    Args:
        baseline_model: the model with good precision
        analysis_model: the model with low precision
        baseline_model_cached_attr: baseline model cached attr
        analysis_model_cached_attr: analysis model cached attr
        analysis_model_type: the low precision type
        metric: compute and sort metric
        device: run model on which device. Only used in quantized sensitivity
    """

    def __init__(
        self,
        baseline_model: torch.nn.Module,
        analysis_model: torch.nn.Module,
        baseline_model_cached_attrs: Dict,
        analysis_model_cached_attrs: Dict,
        analysis_model_type: str,
        metric: str,
        device: torch.device,
        op_flops_mapping: Dict = None,
    ):
        super(Sensitivity, self).__init__()
        self.baseline_model = baseline_model
        self.analysis_model = analysis_model
        self.analysis_model_type = analysis_model_type
        self.baseline_model_cached_attrs = baseline_model_cached_attrs
        self.analysis_model_cached_attrs = analysis_model_cached_attrs
        self.device = device
        self.baseline_info_manager = None
        self.baseline_ret = None
        self.sensitivity_table = SensitivityTable(metric)
        self.total_flops = sum(list(op_flops_mapping.values()))
        self.op_flops_mapping = {
            k: f"{v}({v/self.total_flops:.2%})"
            for k, v in op_flops_mapping.items()
        }
        self._enable_log = True

    def set_cached_attrs(self, model, cached_attrs):
        for k, v in cached_attrs.items():
            _set_attr(model, k, v)

    def run_baseline_model(self, example_input):
        self.set_cached_attrs(
            self.baseline_model, self.baseline_model_cached_attrs
        )
        if self.analysis_model_type == "fake_quant":
            with torch.no_grad(), SensitiveHelper(
                self.baseline_model
            ) as baseline_profiler:
                SensitiveHelper.set_model_type("float")
                self.baseline_ret = self.baseline_model(*example_input)
            self.baseline_info_manager = baseline_profiler.get_info_manager()
        else:
            with torch.no_grad():
                self.baseline_ret = self.baseline_model(*example_input)
        return self.baseline_ret

    def run_analysis_model(self, example_input):
        self.set_cached_attrs(
            self.analysis_model, self.analysis_model_cached_attrs
        )
        with torch.no_grad():
            analysis_ret = self.analysis_model(*example_input)
        return analysis_ret

    def forward(self, example_input, ops, out_index):
        """Run two model and compute sensitivity.

        Args:
            example_input: model input
            ops: leaf ops to be computed
            out_index: output index
        """
        example_input = _as_tuple(example_input)

        # run baseline model
        if self.baseline_ret is None:
            self.run_baseline_model(copy.deepcopy(example_input))

        if self.analysis_model_type == "fake_quant":
            self.fake_quant_sensitivity(example_input, ops, out_index)
        elif self.analysis_model_type == "quantized":
            self.quantized_sensitivity(example_input, ops, out_index)
        elif self.analysis_model_type == "hbir":
            self.qat_hbir_sensitivity(example_input, ops, out_index)

    def fake_quant_sensitivity(self, example_input, ops, out_index):
        for _, mod in self.analysis_model.named_modules():
            if isinstance(mod, FakeQuantizeBase):
                mod.enable_fake_quant(False)

        total_ops = len(ops)

        with SensitiveHelper(self.analysis_model) as analysis_profiler:
            SensitiveHelper.set_model_type("qat")
            analysis_profiler.set_info_manager(self.baseline_info_manager)

            for i, op in enumerate(ops):
                module = self.analysis_model.get_submodule(op)

                if (
                    hasattr(module, "activation_post_process")
                    and module.activation_post_process is not None
                ):
                    module.activation_post_process.enable_fake_quant(True)
                    analysis_profiler.reset_op_appear_times()
                    analysis_ret = self.run_analysis_model(example_input)
                    self.sensitivity_table.add(
                        self.baseline_ret,
                        analysis_ret,
                        out_index,
                        op,
                        "activation",
                        type(module),
                        module.activation_post_process.dtype,
                        self.op_flops_mapping.get(op, "0(0%)"),
                    )
                    module.activation_post_process.enable_fake_quant(False)

                if (
                    isinstance(module, GELU)
                    and hasattr(module, "lut")
                    and hasattr(module.lut, "activation_post_process")
                    and module.lut.activation_post_process is not None
                ):
                    module.lut.activation_post_process.enable_fake_quant(True)
                    analysis_profiler.reset_op_appear_times()
                    analysis_ret = self.run_analysis_model(example_input)
                    self.sensitivity_table.add(
                        self.baseline_ret,
                        analysis_ret,
                        out_index,
                        op,
                        "activation",
                        type(module),
                        module.lut.activation_post_process.dtype,
                        self.op_flops_mapping.get(op, "0(0%)"),
                    )
                    module.lut.activation_post_process.enable_fake_quant(False)

                if (
                    hasattr(module, "weight_fake_quant")
                    and module.weight_fake_quant is not None
                ):
                    module.weight_fake_quant.enable_fake_quant(True)
                    analysis_profiler.reset_op_appear_times()
                    analysis_ret = self.run_analysis_model(example_input)
                    self.sensitivity_table.add(
                        self.baseline_ret,
                        analysis_ret,
                        out_index,
                        op,
                        "weight",
                        type(module),
                        module.weight_fake_quant.dtype,
                        self.op_flops_mapping.get(op, "0(0%)"),
                    )
                    module.weight_fake_quant.enable_fake_quant(False)

                if self._enable_log and i % 50 == 0:
                    logger.info(f"{i}/{total_ops} ops done...")

    def quantized_sensitivity(self, example_input, ops, out_index):
        convert_func = (
            convert_fx
            if isinstance(self.baseline_model, torch.fx.GraphModule)
            else convert
        )

        all_preserve_qat_model = copy.deepcopy(self.baseline_model)
        all_ops = find_leaf_modules(all_preserve_qat_model)
        set_preserve_qat_mode(all_preserve_qat_model, prefixes=tuple(all_ops))

        total_ops = len(ops)
        for i, op in enumerate(ops):
            module = self.analysis_model.get_submodule(op)
            tmp_model = copy.deepcopy(all_preserve_qat_model)
            set_preserve_qat_mode(tmp_model, prefixes=(op,), value=False)
            single_op_int_model = (
                convert_func(tmp_model).eval().to(self.device)
            )
            self.set_cached_attrs(
                single_op_int_model, self.analysis_model_cached_attrs
            )
            with torch.no_grad():
                int_ret = single_op_int_model(*example_input)
            self.sensitivity_table.add(
                self.baseline_ret,
                int_ret,
                out_index,
                op,
                "activation",
                type(module),
                None,
                self.op_flops_mapping.get(op, "0(0%)"),
            )
            if self._enable_log and i % 50 == 0:
                logger.info(f"{i}/{total_ops} ops done...")

    def qat_hbir_sensitivity(self, example_input, ops, out_index):
        total_ops = len(ops)
        for i, op in enumerate(ops):
            module = self.baseline_model.get_submodule(op)

            # export one module to hbir
            def _forward_hook(mod, args, kwargs, output):
                bc_out = get_single_bc_error(mod, args, kwargs, output)
                origin_device = [
                    x for x in tree_flatten(args)[0] if isinstance(x, Tensor)
                ][0].device
                hbir_ret = bc_out["qatbc"]
                hbir_ret = apply_to_collection(
                    hbir_ret, Tensor, lambda x: x.to(origin_device)
                )
                if isinstance(output, QTensor):
                    return QTensor(
                        hbir_ret[0],
                        output.q_scale(),
                        output.dtype,
                        output.q_per_channel_axis(),
                    )
                elif isinstance(output, DispatchedTensorWrapper):
                    if isinstance(output._t, QTensor):
                        output._t = QTensor(
                            hbir_ret[0],
                            output.q_scale(),
                            output.dtype,
                            output.q_per_channel_axis(),
                        )
                    elif isinstance(output._t, Tensor):
                        output._t = hbir_ret[0]
                    return output
                elif isinstance(output, Tensor):
                    return hbir_ret[0]

            handle = module.register_forward_hook(
                _forward_hook, with_kwargs=True, prepend=True
            )

            with torch.no_grad():
                self.set_cached_attrs(
                    self.baseline_model, self.baseline_model_cached_attrs
                )
                hbir_ret = self.baseline_model(*copy.deepcopy(example_input))

            handle.remove()

            self.sensitivity_table.add(
                self.baseline_ret,
                hbir_ret,
                out_index,
                op,
                "activation",
                type(module),
                None,
                self.op_flops_mapping.get(op, "0(0%)"),
            )
            if i % 50 == 0:
                logger.info(f"{i}/{total_ops} ops done...")

    # beta function
    def auto_find_topk(self, example_input, out_name, index, out_dir):
        example_input = _as_tuple(example_input)
        # run baseline model
        if self.baseline_ret is None:
            self.run_baseline_model(example_input)

        table = self.sensitivity_table.get()

        topk_error = []
        set_fake_quantize(self.analysis_model, FakeQuantState.VALIDATION)
        # origin error
        analysis_ret = self.run_analysis_model(example_input)
        simi = self.sensitivity_table.compute(self.baseline_ret, analysis_ret)
        assert index < len(simi)
        topk_error.append((0, simi[index]))

        logger.info("Compute topk sensitive ops error...")
        for i in tqdm(range(len(table))):
            item = table[i]
            if item[4] == "qint8":
                submod = self.analysis_model.get_submodule(item[0])
                if item[1] == "activation":
                    submod.activation_post_process.reset_dtype(qint16)
                elif item[1] == "weight":
                    submod.weight_fake_quant.reset_dtype(qint16)
                analysis_ret = self.run_analysis_model(example_input)
                simi = self.sensitivity_table.compute(
                    self.baseline_ret, analysis_ret
                )
                assert index < len(simi)
                topk_error.append((i + 1, simi[index]))

        metric = self.sensitivity_table.metric
        txtname = os.path.join(
            out_dir, f"output_{out_name}_{metric}_topk_error.txt"
        )
        with open(txtname, "w") as f:
            f.write(tabulate(topk_error, headers=["topk", metric]))
