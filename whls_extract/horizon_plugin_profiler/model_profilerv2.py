import copy
import logging
import math
import multiprocessing as mp
import os
from multiprocessing import Process, Queue
from numbers import Real
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import horizon_plugin_profiler as hpp
from horizon_plugin_profiler.utils.cal_ops import cal_flops
from horizon_plugin_profiler.utils.hbdk4_optional import HbirModule
from horizon_plugin_profiler.utils.location_info import TorchLocationInfo
from horizon_plugin_profiler.utils.model_helper import (
    HbirModuleWrapper,
    ModelStage,
    _as_tuple,
    apply_to_collection,
    deepcopy,
    find_leaf_modules,
    get_device,
    get_model_stage,
)
from horizon_plugin_profiler.utils.op_running_info import OpRunningInfoManager
from horizon_plugin_profiler.utils.profiler_tracer import (
    OpInfoRecorder,
    OpInfoRecorderWithSinglebc,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import numpy as np
import torch
from jinja2 import Environment, PackageLoader
from tabulate import tabulate
from torch import Tensor
from torch.utils._pytree import tree_flatten
from tqdm import tqdm

from horizon_plugin_pytorch import QTensor
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.fx.jit_scheme import GraphModule
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn.qat import DeQuantStub, QuantStub
from horizon_plugin_pytorch.nn.quantized import DeQuantize, Quantize
from horizon_plugin_pytorch.quantization import PrepareMethod, prepare
from horizon_plugin_pytorch.quantization.fuse_modules import (
    get_op_list_to_fuser_mapping,
)
from horizon_plugin_pytorch.quantization.fx.graph_module import (
    GraphModuleWithAttr,
)
from horizon_plugin_pytorch.quantization.qconfig_template import (
    ModuleNameQconfigSetter,
)
from ._sensitivity_helper import (
    Sensitivity,
    SensitivityTable,
    _sensitive_worker,
)
from ._similarity import _compute_similarity, _func_map
from .find_bad_case import _set_attr, find_bad_case
from .get_module_called_count import get_module_called_count
from .hbir_model_profiler import HbirModelProfiler

logger = logging.getLogger(__name__)


def _get_y_bound(min1, max1, min2, max2):
    """Get ymin ymax for echarts figure.

    Args:
        min1: min data shown on yaxis0
        max1: max data shown on yaxis0
        min2: min data shown on yaxis1
        max2: max data shown on yaxis1

    Returns:
        y1min, y1max, y2min, y2max
    """
    if min2 == max2:
        if min1 == max1:
            max1 = max(0, min1)
            min1 = min(0, min1)
        y1max = max1
        y2max = y1max
        y1min = min1
        y2min = y1min
    else:
        ratio = (max1 - min1) / (max2 - min2)
        if max1 < max2 * ratio:
            y1max = max2 * ratio
            y2max = max2
        else:
            y1max = max1
            y2max = max1 / ratio

        if min1 < min2 * ratio:
            y1min = min1
            y2min = min1 / ratio
        else:
            y1min = min2 * ratio
            y2min = min2

    y1min, y1max = y1min * 1.1, y1max * 1.1
    y2min, y2max = y2min * 1.1, y2max * 1.1
    # JS only support list, not tuple
    return [y1min, y1max, y2min, y2max]


class ModelProfiler(OpInfoRecorder):
    """Model featuremaps and parameters profiler context manager.

    Run model and dump each layer inputs/outputs/weights/bias to `out_dir`.

    Args:
        model: the model to profiler
        out_dir: path to save each layer info

    Examples:
    .. code-block:: python

        # 1. use ModelProfiler as a decorator and run model
        with ModelProfiler(net, "./your_path") as p:
            net(data)

        # 2. show profiler result in a table
        p.get_info_manager().table()

        # 3. show histogram in tensorboard
        p.get_info_manager().tensorboard()

    """

    pass


class QuantAnalysis(object):
    """Analysis quantization model and help figure out low precision problem.

    Args:
        baseline_model: the model with good precision.
        analysis_model: the model with low precision.
        analysis_model_type: the low precision type. Support two types now:
            1. "fake_quant": the analysis_model can be calibration/qat model
                and the baseline_model can be float model or a int8/16 mixed
                qconfig model with good precision.
            2. "quantized": the analysis_model must be quantized model and the
                baseline_model must be calibration/qat model.
        device_ids: GPU device ids to run analysis. Default None.
        post_process: post process function which performs on model output.
        out_dir: path to save advisor and comparsion result. If None, all
            results will be saved in ./horizon_quant_analysis.

    Examples:
    .. code-block:: python

        # 1. init a analysis
        qa = QuantAnalysis(float_net, calibration_model, "fake_quant", path)

        # 2. find bad case input
        qa.auto_find_bad_case(dataloader)
        # if you have a custom transform to do on data, pass a simple generator
        qat.auto_find_bad_case([transform(x) for x in dataloader])

        # 3. run models to get each model per layer info
        qa.run()

        # 4. compare per layer to find abnormal layer
        # this function gives some advisor
        qa.compare_per_layer()

        # 5. find quantization sensitive ops
        # topk sensitive ops can be set int16 to get higher precision
        qa.sensitivity()

    If two model profiler have been ran before analysis

    .. code-block:: python

        # 1. init a analysis
        qa = QuantAnalysis(float_net, calibration_model, "fake_quant", path)

        # 2. set profiler path
        qa.set_model_profiler_dir(float_profiler_path, calib_profiler_path)

        # 3. directly compare
        qa.compare_per_layer()

    """

    @typechecked
    def __init__(
        self,
        baseline_model: Union[torch.nn.Module, HbirModule],
        analysis_model: Union[torch.nn.Module, HbirModule],
        analysis_model_type: str,
        device_ids: Union[List[int], int] = None,
        post_process: Optional[Callable] = None,
        out_dir: Optional[str] = None,
    ):
        if analysis_model_type not in ("fake_quant", "quantized", "hbir"):
            raise ValueError(
                "Only support analysis_model_type be 'fake_quant' or "
                + f"'quantized', but get {analysis_model_type}."
            )

        if not isinstance(baseline_model, torch.nn.Module):
            baseline_model = HbirModuleWrapper(baseline_model)
        if not isinstance(analysis_model, torch.nn.Module):
            analysis_model = HbirModuleWrapper(analysis_model)

        self.baseline_model = baseline_model
        self.analysis_model = analysis_model
        self.analysis_model_type = analysis_model_type
        if device_ids is not None:
            if type(device_ids) == int:
                device_ids = [
                    device_ids,
                ]
            assert len(device_ids) > 0
            self.device = device_ids
        else:
            self.device = [None]

        self.post_process = post_process
        if post_process is not None:

            def post_process_hook(mod, args, output):
                return post_process(output)

            self.baseline_model.register_forward_hook(post_process_hook)
            self.analysis_model.register_forward_hook(post_process_hook)

        self.out_dir = (
            out_dir if out_dir is not None else "./horizon_quant_analysis"
        )
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.input = None
        self.baseline_model_cached_attr = {}
        self.analysis_model_cached_attr = {}
        # index -> (input_data, model1_cached_attr, model2_cached_attr)
        self.bad_input_dict = {}

        # metric -> bad input index
        self.worst_index_dict = {}

        # [
        #     [output0: cosine_bad_index, mse_bad_index, L1_bad_index, ...],
        #     [output1: cosine_bad_index, mse_bad_index, L1_bad_index, ...],
        #     ...
        # ]
        self.bad_index_table = []

        self.baseline_model_profiler_path = os.path.join(
            self.out_dir, "baseline_model"
        )
        self.analysis_model_profiler_path = os.path.join(
            self.out_dir, "analysis_model"
        )

    @typechecked
    def auto_find_bad_case(
        self,
        data_generator: Iterable,
        num_steps: Optional[int] = None,
        metric: str = "L1",
        device: Optional[Union[torch.device, str, int, List[int]]] = None,
        custom_metric_func: Optional[Callable] = None,
        custom_metric_order_seq: Optional[str] = None,
        cached_attrs: Optional[Tuple[str, ...]] = None,
    ):
        """
        Automatically run over given data generator to find the baddest case.

        Args:
            data_generator: input dataloader or a custom iterable object. It
                must generate a data each iteration. Example:
                1. torch dataloader:
                    data_generator = torch.utils.data.DataLoader()
                2. a custom generator:
                    data_generator = [x for x in [1, 2, 3]]
            num_steps: num of steps to find bad case.
            metric: support Cosine/MSE/L1/KL/SQNR or "custom". If
                metric == "custom", `custom_metric_func` and
                `custom_metric_order_seq` must be set. Default L1.
            device(Deprecated): run model on which device. Default: None
            custom_metric_func: user-defined callable metric func to compute
                model output difference
            custom_metric_order_seq: user-defined metric order sequence from
                bad to good. For example, cosine metric order is ascending, the
                smallest is the baddest. While L1 metric order is descending,
                the biggest is the baddest. Only support
                "ascending"/"descending".
            cached_attrs: cached attrs to use as input. Usually used in
                sequence model. Default None.
        """
        if metric == "custom":
            assert (
                custom_metric_func is not None
                and custom_metric_order_seq is not None
            ), (
                "custom metric func and order sequence must be set "
                + "when use metric = 'custom'."
            )
            assert custom_metric_order_seq in ("ascending", "descending")
        if len(self.device) > 1:
            logger.info(
                "auto_find_bad_case currently only support run on single "
                + f"device, will use GPU {self.device[0]}"
            )
        (
            self.bad_index_table,
            self.worst_index_dict,
            self.bad_input_dict,
        ) = find_bad_case(
            self.baseline_model,
            self.analysis_model,
            data_generator,
            num_steps,
            # current support find bad case on one device
            get_device(self.device[0]),
            custom_metric_func,
            custom_metric_order_seq,
            cached_attrs,
            self.out_dir,
        )
        index = self.worst_index_dict[metric]
        logger.info(
            f"Function `run` and `compare_per_layer` defaultly use index "
            f"{index} data in dataloader, which is baddest under {metric}. "
            "If you want to run and compare using other index, specify "
            "`index` when call function `run`."
        )
        self.input, attrs = self.bad_input_dict[index]
        (
            self.baseline_model_cached_attr,
            self.analysis_model_cached_attr,
        ) = attrs
        # save default badcase into badcase.pt
        torch.save(
            (
                self.input,
                self.baseline_model_cached_attr,
                self.analysis_model_cached_attr,
            ),
            os.path.join(self.out_dir, "badcase.pt"),
        )
        return self.input

    @typechecked
    def set_bad_case(
        self,
        data: Any,
        baseline_model_cached_attr: Optional[Dict] = None,
        analysis_model_cached_attr: Optional[Dict] = None,
    ):
        """Set bad case manually.

        Args:
            data: bad case input
            baseline_model_cached_attr: baseline model cached attr
            analysis_model_cached_attr: analysis model cached attr
        """
        self.input = data
        self.baseline_model_cached_attr = (
            {}
            if baseline_model_cached_attr is None
            else baseline_model_cached_attr
        )
        self.analysis_model_cached_attr = (
            {}
            if analysis_model_cached_attr is None
            else analysis_model_cached_attr
        )

        def _check_attr(model, cached_attrs):
            for attr in cached_attrs:
                prefixes = attr.split(".")
                target = model
                for name in prefixes:
                    if hasattr(target, name):
                        target = getattr(target, name)
                    else:
                        raise AttributeError(
                            f"Can not find {attr} in model. Please check key "
                            + "in cached_attr dict."
                        )

        _check_attr(self.baseline_model, self.baseline_model_cached_attr)
        _check_attr(self.analysis_model, self.analysis_model_cached_attr)

    @typechecked
    def load_bad_case(self, filename: Optional[str] = None):
        """Use `torch.load` load bad case from a given file.

        Args:
            filename: the bad case file name.
        """
        if filename is None:
            filename = os.path.join(self.out_dir, "badcase.pt")
        if os.path.exists(filename):
            (
                self.input,
                self.baseline_model_cached_attr,
                self.analysis_model_cached_attr,
            ) = torch.load(filename)
        else:
            raise RuntimeError(
                f"Data file path {filename} does not exist. "
                + "Please give a valid path."
            )

        path = os.path.join(self.out_dir, "all_badcase_info.pt")
        if os.path.exists(path):
            (
                self.bad_index_table,
                self.worst_index_dict,
                self.bad_input_dict,
            ) = torch.load(path)
            return
        else:
            logger.warning(
                f"No all_badcase_info.pt in {self.out_dir}. Skip load."
            )

        # keep for compatibility
        # bad input dict
        path = os.path.join(self.out_dir, "all_badcase_inputs.pt")
        if os.path.exists(path):
            self.bad_input_dict = torch.load(path)
        else:
            logger.warning(
                f"No all_badcase_inputs.pt in {self.out_dir}. Skip load."
            )

        # badcase index table
        path = os.path.join(self.out_dir, "badcase_index_table.pt")
        if os.path.exists(path):
            self.bad_index_table = torch.load(path)
        else:
            logger.warning(
                f"No badcase_index_table.pt in {self.out_dir}. Skip load."
            )

    def save_bad_case(self):
        """Use `torch.save` save the bad case in '{self.out_dir}/badcase.txt'.

        Args:
            filename: the bad case file name
        """
        if self.input is not None:
            torch.save(
                (
                    self.input,
                    self.baseline_model_cached_attr,
                    self.analysis_model_cached_attr,
                ),
                os.path.join(self.out_dir, "badcase.pt"),
            )
        else:
            raise RuntimeError(
                "Can not save bad case 'None'. Please use `auto_find_bad_case`"
                + " or `set_bad_case` to set a valid bad case."
            )

    @typechecked
    def set_model_profiler_dir(
        self,
        baseline_model_profiler_path: str,
        analysis_model_profiler_path: str,
    ):
        """Set model profiler path.

        In some cases, two model profiler runs before QuantAnalysis init. This
        function directly sets two model profiler out_dir, which supports
        running `compare_per_layer` without running `run`.
        """
        for path in (
            baseline_model_profiler_path,
            analysis_model_profiler_path,
        ):
            if not os.path.exists(path):
                raise RuntimeError(f"The path {path} does not exist.")

        self.baseline_model_profiler_path = baseline_model_profiler_path
        self.analysis_model_profiler_path = analysis_model_profiler_path

    def check_model_params(self, example_input):
        base_stage = get_model_stage(self.baseline_model)
        ana_stage = get_model_stage(self.analysis_model)

        # only check float vs calib
        if not (
            self.analysis_model_type == "fake_quant"
            and base_stage == ModelStage.FLOAT
            and ana_stage == ModelStage.QAT
        ):
            return

        # fuse baseline float model
        device = get_device(self.device[0])
        logger.info("Begin model parameters check...")
        float_model = copy.deepcopy(self.baseline_model).to(device).eval()
        example_input = copy.deepcopy(example_input)
        example_input = apply_to_collection(
            example_input, Tensor, lambda x: x.to(device)
        )

        # get prepare method
        if isinstance(self.analysis_model, GraphModuleWithAttr):
            prepare_method = PrepareMethod.SYMBOLIC
        elif isinstance(self.analysis_model, GraphModule):
            prepare_method = PrepareMethod.JIT_STRIP
            if hasattr(self.analysis_model, "_prepare_method"):
                prepare_method = self.analysis_model._prepare_method
        else:
            prepare_method = PrepareMethod.EAGER
            logger.info(
                "Fuse method is uncertainty in eager model. "
                "Skip model param check."
            )
            return

        qconfig_dict = {
            name: mod.qconfig
            for name, mod in self.analysis_model.named_modules()
            if hasattr(mod, "qconfig")
        }

        # only fuse cannot process `withbn` module key. Directly prepare here
        fused_model = (
            prepare(
                float_model,
                example_inputs=example_input,
                qconfig_setter=ModuleNameQconfigSetter(qconfig_dict),
                method=prepare_method,
            )
            .eval()
            .to(device)
        )
        module_count = get_module_called_count(
            fused_model, example_input, print_tabulate=False
        )
        unused_mod = tuple(
            [name for name, count in module_count.items() if count == 0]
        )

        fused_state_dict = fused_model.state_dict()
        ana_state_dict = self.analysis_model.state_dict()

        for key, value in fused_state_dict.items():
            # Skip fake quant params and bn num_batches_tracked
            # Old version plugin from_float does not copy num_batches_tracked
            # while new plugin does! So num_batches_tracked may diff from
            # current prepared model and old checkpoint.
            if any(
                x in key
                for x in (
                    "activation_post_process",
                    "weight_fake_quant",
                    "num_batches_tracked",
                )
            ):
                continue
            if key not in ana_state_dict:
                raise KeyError(
                    f"Cannot find `{key}` in analysis_model. Please check "
                    "if analysis_model converted from baseline_model."
                )
            diff = torch.max(torch.abs(value - ana_state_dict[key].to(device)))
            if not key.startswith(unused_mod) and diff > 1e-5:
                raise ValueError(
                    f"`{key}` in analysis_model differs from that in model "
                    f"converted from baseline_model. The max diff is {diff}. "
                    "Please check analysis_model converting pipeline."
                )

        logger.info("End model parameters check")

    @typechecked
    def run(
        self,
        device: Optional[Union[torch.device, str, int, List[int]]] = None,
        index: Optional[int] = None,
    ):
        """Run models and save data.

        Args:
            device(Deprecated): run model on which device. Default: None
            index: use which index input as example input. Default: None, use
                self.input as example input.
        """
        for path in (
            self.baseline_model_profiler_path,
            self.analysis_model_profiler_path,
        ):
            if not os.path.exists(path):
                os.mkdir(path)

        assert self.input is not None or index is not None
        if index is not None:
            if len(self.bad_input_dict) < 1:
                raise RuntimeError(
                    "No bad case found. Please run auto_find_bad_case first."
                )
            if index not in self.bad_input_dict:
                raise RuntimeError(
                    f"{index} is not a bad input index. Please choose a valid "
                    + "index in badcase.txt."
                )
            example_input, attrs = self.bad_input_dict[index]
            (
                self.baseline_model_cached_attr,
                self.analysis_model_cached_attr,
            ) = attrs
        else:
            example_input = self.input

        self.check_model_params(example_input)

        device = get_device(self.device[0])
        data = copy.deepcopy(example_input)
        data = _as_tuple(
            apply_to_collection(data, Tensor, lambda x: x.to(device))
        )
        self.baseline_model_cached_attr = apply_to_collection(
            self.baseline_model_cached_attr, Tensor, lambda x: x.to(device)
        )
        self.analysis_model_cached_attr = apply_to_collection(
            self.analysis_model_cached_attr, Tensor, lambda x: x.to(device)
        )

        baseline_model = deepcopy(self.baseline_model)
        if isinstance(baseline_model, HbirModuleWrapper):
            base_model_profiler = HbirModelProfiler(
                baseline_model, self.baseline_model_profiler_path
            )
        elif self.analysis_model_type == "hbir":
            # only run hbdk4 single op error when type == hbir
            base_model_profiler = OpInfoRecorderWithSinglebc(
                baseline_model, self.baseline_model_profiler_path
            )
        else:
            base_model_profiler = ModelProfiler(
                baseline_model, self.baseline_model_profiler_path
            )

        with base_model_profiler:
            baseline_model.to(device).eval()
            for k, v in self.baseline_model_cached_attr.items():
                _set_attr(baseline_model, k, v)
            with torch.no_grad():
                baseline_model(*copy.deepcopy(data))
            for k in self.baseline_model_cached_attr:
                _set_attr(baseline_model, k, None)

        base_model_profiler.get_info_manager().table()

        analysis_model = deepcopy(self.analysis_model)
        if isinstance(analysis_model, HbirModuleWrapper):
            analysis_model_profiler = HbirModelProfiler(
                analysis_model, self.analysis_model_profiler_path
            )
        else:
            analysis_model_profiler = ModelProfiler(
                analysis_model, self.analysis_model_profiler_path
            )
        with analysis_model_profiler:
            analysis_model.to(device).eval()
            for k, v in self.analysis_model_cached_attr.items():
                _set_attr(analysis_model, k, v)
            with torch.no_grad():
                analysis_model(*copy.deepcopy(data))
            for k in self.analysis_model_cached_attr:
                _set_attr(analysis_model, k, None)

        analysis_model_profiler.get_info_manager().table()

    def _load_models_meta(self):
        """Load model forward sequence and module path map."""
        for path in (
            self.baseline_model_profiler_path,
            self.analysis_model_profiler_path,
        ):
            if not os.path.exists(path):
                raise RuntimeError(f"{path} does not exist.")

        return OpRunningInfoManager.load_meta(
            self.baseline_model_profiler_path
        ), OpRunningInfoManager.load_meta(self.analysis_model_profiler_path)

    def _check_input_output(self, m: OpRunningInfoManager):
        abnormal_layers = []
        for loc in m.call_sequence:
            if loc.op_name in (
                TorchLocationInfo.format_op_name(QuantStub),
                TorchLocationInfo.format_op_name(Quantize),
            ):
                op_info = m.get_info_by_loc(loc)
                if isinstance(op_info.input, tuple):
                    quant_input = op_info.input[0]
                else:
                    quant_input = op_info.input

                if isinstance(op_info.output, tuple):
                    quant_output = op_info.output[0]
                else:
                    quant_output = op_info.output

                if isinstance(quant_input, QTensor):
                    quant_input = quant_input.dequantize()
                input_min = quant_input.min()
                input_max = quant_input.max()
                input_range = input_max - input_min
                msg = ""
                advisor = ""
                if input_min >= 0 or input_max <= 0:
                    msg += "Asymmetric about 0. "
                    advisor += "Do symmetric normalization about 0. "
                if input_range >= 128 and quant_output.dtype == "qint8":
                    msg += "Large input range."
                    advisor += "Maybe use int16 quantization."

                if msg != "" and advisor != "":
                    abnormal_layers.append(
                        (loc.mod_name, loc.op_name, msg, advisor)
                    )

            msg = "Model outputs are not in high precision. "
            advisor = (
                "If they come from conv/linear, it is recommended to"
                " config `activation=None` to allow high precision out."
            )
            if loc.op_name in (
                TorchLocationInfo.format_op_name(DeQuantStub),
                TorchLocationInfo.format_op_name(DeQuantize),
            ):
                op_info = m.get_info_by_loc(loc)
                if isinstance(op_info.input, tuple):
                    dequant_input = op_info.input[0]
                else:
                    dequant_input = op_info.input

                if (
                    isinstance(dequant_input, QTensor)
                    and dequant_input.q_scale() is not None
                ):

                    abnormal_layers.append(
                        (loc.mod_name, loc.op_name, msg, advisor)
                    )

        return abnormal_layers

    def _find_abnormal_layers(
        self, m: OpRunningInfoManager, analysis_table: List[Dict[str, Any]]
    ):
        headers = ["mod_name", "op_type", "abnormal_info", "advice"]

        abnormal_table = self._check_input_output(m)

        for t in analysis_table:
            if t["Cosine"] is None or t["base_model_min"] is None:
                continue
            mod_name = t["mod_name"]
            op_type = t["base_op_type"]

            dmax = max(
                [
                    abs(t["base_model_min"]),
                    abs(t["base_model_max"]),
                    abs(t["analy_model_min"]),
                    abs(t["analy_model_max"]),
                ]
            )
            # check large data range
            if dmax > 128 and t["quant_dtype"] == "qint8":
                abnormal_table.append(
                    (
                        mod_name,
                        op_type,
                        (
                            f"Data range {2 * dmax} maybe too large "
                            + "for int8 quantization."
                        ),
                        "Please try qint16 quantization.",
                    )
                )
            if dmax > 32768:
                abnormal_table.append(
                    (
                        mod_name,
                        op_type,
                        f"Total data range {2 * dmax} maybe too "
                        + "large for quantization.",
                        "Please change model structure or "
                        + "limit this output range",
                    )
                )
            # check analysis_min >= quant_min & analysis_max <= quant_max
            if t["qscale"] is not None:
                if t["quant_dtype"] not in ("qint32", torch.float32) and (
                    (
                        qinfo(t["quant_dtype"]).min * t["qscale"]
                        - t["analy_model_min"]
                    )
                    > t["qscale"]
                    or (
                        t["analy_model_max"]
                        - qinfo(t["quant_dtype"]).max * t["qscale"]
                    )
                    > t["qscale"]
                ):
                    abnormal_table.append(
                        (
                            mod_name,
                            op_type,
                            "Current scale does not cover the data range",
                            "Please check whether fake quant enabled.",
                        )
                    )
            # check inf/NaN
            if any(
                [
                    math.isinf(x) or math.isnan(x)
                    for x in (
                        t["base_model_min"],
                        t["base_model_max"],
                        t["analy_model_min"],
                        t["analy_model_max"],
                    )
                ]
            ):
                abnormal_table.append(
                    (mod_name, op_type, "inf/nan output, check if expected.")
                )

        if len(abnormal_table) != 0:
            abnormal_path = os.path.join(
                self.out_dir, "abnormal_layer_advisor.txt"
            )
            with open(abnormal_path, "w") as f:
                f.write(
                    tabulate(
                        abnormal_table,
                        headers=headers,
                        tablefmt="psql",
                    )
                )

    def _reloc_fuse_module(self, table):
        fusion_patterns = get_op_list_to_fuser_mapping()
        fusion_patterns_name = [
            [TorchLocationInfo.format_op_type(mod) for mod in key]
            for key in fusion_patterns.keys()
        ]
        begin_mods = {x[0] for x in fusion_patterns_name}
        i = 0
        while i < len(table):
            if table[i]["base_op_type"] in begin_mods:
                # at most 4 fuse ops (conv bn add relu)
                for j in range(3, 0, -1):
                    base_pattern = [
                        x["base_op_type"] for x in table[i : i + j + 1]
                    ]
                    # process add
                    base_pattern = [
                        ".".join(x.split(".")[:-1])
                        if x.startswith(
                            "horizon_plugin_pytorch.nn.quantized.functional_modules.FloatFunctional"  # noqa E501
                        )
                        else x
                        for x in base_pattern
                    ]
                    ana_pattern = [
                        x["analy_op_type"] for x in table[i : i + j + 1]
                    ]
                    if (
                        base_pattern in fusion_patterns_name
                        and len(
                            [
                                x
                                for x in ana_pattern
                                if x != "torch.nn.modules.linear.Identity"
                            ]
                        )
                        == 1
                    ):
                        # clear previous compare metric
                        for index in range(i, i + j):
                            table[index]["shape"] = None
                            table[index]["Cosine"] = None
                            table[index]["L1"] = None
                            table[index]["Atol"] = None
                            table[index]["max_qscale_diff"] = None
                        i += j
                        break
            i += 1
        return table

    @typechecked
    def compare_per_layer(
        self, prefixes: Tuple[str, ...] = None, types: Tuple[Type, ...] = None
    ):
        """Compare each layer output in two models.

        This function compares each layer output in two models and produces
        these files:
            1. abnormal_layer_advisor.txt: show abnormal layers and advisor.
            2. profiler.html: show compare results in html
            3. compare_per_layer_out.txt: each layer details in txt format
            4. compare_per_layer_out.csv: each layer details in csv format

        Args:
            prefixes: get features info by the prefix of qualified name
                Default: None.
            types: get features info by module type. Default: None.
        """
        base_info_manager, analysis_info_manager = self._load_models_meta()

        def search_same_location(
            loc: TorchLocationInfo, m: OpRunningInfoManager
        ):
            if loc.type() == "call_function":
                if loc.mod_name in m.name_to_locs:
                    for other_loc in reversed(m.name_to_locs[loc.mod_name]):
                        if (
                            other_loc.op_name == loc.op_name
                            and other_loc.idx == loc.idx
                        ):
                            return other_loc
            else:
                for other_loc in reversed(m.call_sequence):
                    if (
                        other_loc.idx == loc.idx
                        and other_loc.mod_name.startswith(loc.mod_name)
                    ):
                        # skip different mod with same prefix name like
                        # self.quant vs self.quant_mask
                        if (
                            len(other_loc.mod_name) > len(loc.mod_name)
                            and other_loc.mod_name[len(loc.mod_name)] != "."
                        ):
                            continue
                        return other_loc

            return None

        if types is not None:
            types = tuple(TorchLocationInfo.format_op_type(t) for t in types)
        analysis_table = []
        logger.info("Compare per layer...")
        for base_loc in tqdm(base_info_manager.call_sequence.keys()):
            if not (
                (prefixes is None and types is None)
                or (types is not None and base_loc.op_name.startswith(types))
                or (
                    prefixes is not None
                    and base_loc.mod_name.startswith(prefixes)
                )
            ):
                continue
            analysis_loc = search_same_location(
                base_loc, analysis_info_manager
            )
            if analysis_loc is not None:
                base_info = base_info_manager.get_info_by_loc(base_loc)
                analysis_info = analysis_info_manager.get_info_by_loc(
                    analysis_loc
                )
                if (
                    base_info.output is not None
                    and analysis_info.output is not None
                ):
                    # check input same
                    if (
                        ".QuantStub" in base_loc.op_name
                        and base_info.input is not None
                        and analysis_info.input is not None
                        and not isinstance(base_info.input[0], QTensor)
                        and not isinstance(analysis_info.input[0], QTensor)
                    ):
                        base_in = base_info.input[0].as_subclass(torch.Tensor)
                        analy_in = (
                            analysis_info.input[0]
                            .as_subclass(torch.Tensor)
                            .to(base_in.device)
                        )
                        if not torch.equal(base_in, analy_in):
                            max_diff = torch.max(
                                torch.abs(base_in - analy_in)
                            ).item()
                            logger.warning(
                                f"`{base_loc.mod_name}` input differs between "
                                f"two models. The max diff is {max_diff}. If "
                                "this input is model attr, please use "
                                "`cached_attrs` param in `auto_find_bad_case` "
                                "and rerun QuantAnalysis pipeline."
                            )

                    if isinstance(base_info.output, tuple):
                        base_output = base_info.output[0]
                    else:
                        base_output = base_info.output
                    if isinstance(analysis_info.output, tuple):
                        analysis_output = analysis_info.output[0]
                    else:
                        analysis_output = analysis_info.output

                    if base_output.shape != analysis_output.shape:
                        analysis_item = {
                            "mod_name": base_loc.mod_name,
                            "base_op_type": base_loc.op_name,
                            "analy_op_type": analysis_loc.op_name
                            + (
                                ""
                                if not hasattr(analysis_info, "hbir_op_type")
                                else f"[{analysis_info.hbir_op_type}]"
                            ),
                            "shape": "mismatch shape",
                            "quant_dtype": analysis_output.dtype,
                            "qscale": analysis_output.q_scale().item()
                            if isinstance(analysis_output, QTensor)
                            and analysis_output.q_scale().numel() == 1
                            else None,
                            "Cosine": None,
                            # "MSE": None,
                            "L1": None,
                            # "KL": None,
                            # "SQNR": None,
                            "Atol": None,
                            "max_qscale_diff": None,
                            # "Rtol": None,
                            "base_model_min": None,
                            "analy_model_min": None,
                            "base_model_max": None,
                            "analy_model_max": None,
                            "base_model_mean": None,
                            "analy_model_mean": None,
                        }
                        analysis_table.append(analysis_item)

                        # logger.warning(
                        #     "Out shape mismatched in:"
                        #     "\n{}<{}>({}):\n{} vs {}".format(
                        #         base_loc.mod_name,
                        #         base_loc.op_name,
                        #         base_loc.idx,
                        #         base_output.shape,
                        #         analysis_output.shape,
                        #     )
                        # )

                        continue

                    analysis_item = {
                        "mod_name": base_loc.mod_name,
                        "base_op_type": base_loc.op_name,
                        "analy_op_type": analysis_loc.op_name
                        + (
                            ""
                            if not hasattr(analysis_info, "hbir_op_type")
                            else f"[{analysis_info.hbir_op_type}]"
                        ),
                        "shape": base_output.shape,
                        "quant_dtype": analysis_output.dtype,
                        "qscale": analysis_output.q_scale().item()
                        if isinstance(analysis_output, QTensor)
                        and analysis_output.q_scale().numel() == 1
                        else None,
                    }
                    if analysis_item["qscale"] is None:
                        analysis_item["qscale"] = (
                            base_output.q_scale().item()
                            if isinstance(base_output, QTensor)
                            and base_output.q_scale().numel() == 1
                            else None
                        )

                    if isinstance(base_output, QTensor):
                        base_output = base_output.dequantize()
                    if isinstance(analysis_output, QTensor):
                        analysis_output = analysis_output.dequantize()

                    for name in ("Cosine", "L1", "Atol"):
                        simi = _compute_similarity(
                            base_output,
                            analysis_output,
                            _func_map[name],
                        )[0]
                        simi = (
                            simi.item() if isinstance(simi, Tensor) else simi
                        )
                        analysis_item[name] = simi
                    if (
                        base_output.numel() == 0
                        and analysis_output.numel() == 0
                    ):
                        analysis_item["max_qscale_diff"] = None
                        analysis_item["base_model_min"] = None
                        analysis_item["analy_model_min"] = None
                        analysis_item["base_model_max"] = None
                        analysis_item["analy_model_max"] = None
                        analysis_item["base_model_mean"] = None
                        analysis_item["analy_model_mean"] = None
                        analysis_item["same_in_qatbc_qdiff"] = None
                        analysis_item["same_in_intbc_qdiff"] = None
                    else:
                        analysis_item["max_qscale_diff"] = (
                            None
                            if analysis_item["qscale"] is None
                            else analysis_item["Atol"]
                            / analysis_item["qscale"]
                        )
                        analysis_item["base_model_min"] = (
                            base_output.float().min().item()
                        )
                        analysis_item["analy_model_min"] = (
                            analysis_output.float().min().item()
                        )
                        analysis_item["base_model_max"] = (
                            base_output.float().max().item()
                        )
                        analysis_item["analy_model_max"] = (
                            analysis_output.float().max().item()
                        )
                        analysis_item["base_model_mean"] = (
                            base_output.float().mean().item()
                        )
                        analysis_item["analy_model_mean"] = (
                            analysis_output.float().mean().item()
                        )
                        if (
                            hasattr(base_info, "same_in_qatbc_out")
                            and base_info.same_in_qatbc_out is not None
                        ):
                            qat_bc_diff = (
                                torch.abs(
                                    base_output.float()
                                    - base_info.same_in_qatbc_out[0]
                                    .float()
                                    .to(base_output.device)
                                )
                                .max()
                                .item()
                            )
                            analysis_item["same_in_qatbc_qdiff"] = (
                                None
                                if analysis_item["qscale"] is None
                                else qat_bc_diff / analysis_item["qscale"]
                            )

                        if (
                            hasattr(base_info, "same_in_intbc_out")
                            and base_info.same_in_intbc_out is not None
                        ):
                            int_bc_diff = (
                                torch.abs(
                                    base_output.float()
                                    - base_info.same_in_intbc_out[0]
                                    .float()
                                    .to(base_output.device)
                                )
                                .max()
                                .item()
                            )
                            analysis_item["same_in_intbc_qdiff"] = (
                                None
                                if analysis_item["qscale"] is None
                                else int_bc_diff / analysis_item["qscale"]
                            )

                    analysis_table.append(analysis_item)

        # only keep result of last op in fuse pattern
        analysis_table = self._reloc_fuse_module(analysis_table)

        with open(
            os.path.join(self.out_dir, "compare_per_layer_out.txt"), "w"
        ) as f:
            f.write(
                tabulate(
                    analysis_table,
                    headers="keys",
                    tablefmt="psql",
                    floatfmt=".7f",
                    showindex=True,
                )
            )

        with open(
            os.path.join(self.out_dir, "compare_per_layer_out.csv"), "w"
        ) as f:
            if len(analysis_table) == 0:
                logger.warning(
                    "Do not find any matched mods. Please check models or "
                    + "prefixes and types parameters."
                )
                return
            csv_headers = ", ".join(analysis_table[0].keys()) + "\n"
            f.write(csv_headers)
            for analysis_item in analysis_table:
                f.write(
                    ", ".join(
                        str(v).replace(",", "*") if k == "shape" else str(v)
                        for k, v in analysis_item.items()
                    )
                    + "\n"
                )

        self._find_abnormal_layers(analysis_info_manager, analysis_table)
        self._show_html(analysis_table)

    def _show_html(self, table):
        env = Environment(
            loader=PackageLoader(
                "horizon_plugin_profiler", "profiler_templates"
            )
        )
        template = env.get_template("profiler_templatev2")

        # create a soft link of css and echarts.js
        srcdir = os.path.join(hpp.__path__[0], "profiler_templates")
        css_src_dir = os.path.join(srcdir, "style.css")
        echarts_src_dir = os.path.join(srcdir, "echarts.min.js")
        css_dst_dir = os.path.join(self.out_dir, "style.css")
        echarts_dst_dir = os.path.join(self.out_dir, "echarts.js")
        for src, dst in zip(
            (css_src_dir, echarts_src_dir), (css_dst_dir, echarts_dst_dir)
        ):
            if not os.path.exists(dst):
                # if origin file does not exists while the soft link exists,
                # delete the useless soft link
                if os.path.islink(dst):
                    os.remove(dst)
                # check the origin file must exists
                assert os.path.exists(src), f"Can not find file {src}!"
                os.symlink(src, dst)

        def _dump(array):
            ret = []
            for v in array:
                if v is None or v == "skip":
                    ret.append("null")
                elif np.isnan(v):
                    ret.append("NaN")
                elif np.isinf(v):
                    ret.append("Inf")
                else:
                    ret.append(v)
            return ret

        def _compute_y_bounds(left, right):
            left = [x for x in left if isinstance(x, Real)]
            right = [x for x in right if isinstance(x, Real)]
            # left or right maybe all Inf/NaN
            if len(left) == 0 and len(right) == 0:
                left = [
                    0,
                ]
                right = [
                    0,
                ]
            elif len(left) == 0:
                left = right
            elif len(right) == 0:
                right = left
            return _get_y_bound(min(left), max(left), min(right), max(right))

        # similarity
        name = [
            "{}<{}>".format(t["mod_name"], t["base_op_type"]) for t in table
        ]
        cosine = _dump([t["Cosine"] for t in table])
        l1 = _dump([t["L1"] for t in table])
        atol = _dump([t["Atol"] for t in table])
        similarity_dict = {
            "name": name,
            "cosine": cosine,
            "l1": l1,
            "atol": atol,
        }

        # compute ybounds for show
        # cosine/mse/l1 in one figure
        similarity_dict["cml_ybounds"] = _compute_y_bounds(
            cosine
            + [
                0,
            ],
            l1,
        )

        # featuremap statistic
        qrange_diff = _dump(
            [
                abs(
                    max(abs(t["base_model_min"]), abs(t["base_model_max"]))
                    - max(abs(t["analy_model_min"]), abs(t["analy_model_max"]))
                )
                for t in table
                if t["base_model_min"] is not None
            ]
        )
        mean_diff = _dump(
            [
                abs(t["base_model_mean"] - t["analy_model_mean"])
                for t in table
                if t["base_model_mean"] is not None
            ]
        )

        statistic_dict = {}
        statistic_dict["diff"] = {
            "name": name,
            "qrange": qrange_diff,
            "mean": mean_diff,
        }

        statistic_dict["model0"] = {
            "name": name,
            "min": _dump(t["base_model_min"] for t in table),
            "max": _dump(t["base_model_max"] for t in table),
            "mean": _dump(t["base_model_mean"] for t in table),
        }
        statistic_dict["model0"]["ybounds"] = _compute_y_bounds(
            statistic_dict["model0"]["min"] + statistic_dict["model0"]["max"],
            statistic_dict["model0"]["mean"],
        )

        statistic_dict["model1"] = {
            "name": name,
            "min": _dump(t["analy_model_min"] for t in table),
            "max": _dump(t["analy_model_max"] for t in table),
            "mean": _dump(t["analy_model_mean"] for t in table),
        }
        statistic_dict["model1"]["ybounds"] = _compute_y_bounds(
            statistic_dict["model1"]["min"] + statistic_dict["model1"]["max"],
            statistic_dict["model1"]["mean"],
        )

        out = template.render(
            similarity_dict=similarity_dict,
            statistic_dict=statistic_dict,
        )

        out_path = os.path.join(self.out_dir, "profiler.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(out)

    @typechecked
    def sensitivity(
        self,
        device: Any = None,
        metric: str = "L1",
        reverse: bool = False,
    ):
        """Find quantization sensitive ops.

        This function finds quantization sensitive ops, which can be set int16
        for better precision. This function produces two files:
        1. "{self.out_dir}/sensitive_ops.txt": which saves the whole results.
        2. "{self.out_dir}/sensitive_ops.pt": the whole sensitive results,
            which can be load in other functions.

        Args:
            device(Deprecated): run model on which device. Default: None
            metric: sorted result in which metric. Support
                Cosine/MSE/L1/KL/SQNR. Default: L1.
            reverse: sort the sensitivity in reverse order to get insensitive
                ops, which can be set int8 to get higher performance.

        Returns:
            A dict from index to sensitive op table list. Each list format is
                [op_name, sensitive_type, op_type, metric1, metric2, ...]

            Example:
                {
                    0: [
                        [op1, "activation", op1_type, L1],
                        [op2, "activation", op2_type, L1],
                        [op3, "activation", op3_type, L1],
                        [op1, "weight", op1_type, L1],
                        [op2, "both", op2_type, L1],
                        ...
                    ],
                    1: [
                        [op1, "activation", op1_type, L1],
                        [op2, "activation", op2_type, L1],
                        [op3, "activation", op3_type, L1],
                        [op1, "weight", op1_type, L1],
                        [op2, "both", op2_type, L1],
                        ...
                    ],
                }
        """
        if isinstance(self.baseline_model, HbirModuleWrapper) and isinstance(
            self.analysis_model, HbirModuleWrapper
        ):
            raise ValueError("hbir sensitivity needs one torch model")
        if metric not in ("Cosine", "MSE", "L1", "KL", "SQNR"):
            raise ValueError(
                "Only support metric be one or some of Cosine/MSE/L1/KL/SQNR"
                + f" but get {metric}."
            )

        cpu = torch.device("cpu")
        baseline_model = deepcopy(self.baseline_model).eval().to(cpu)
        analysis_model = deepcopy(self.analysis_model).eval().to(cpu)

        assert self.analysis_model_type in ("fake_quant", "quantized", "hbir")
        if len(self.device) > 1:
            if self.analysis_model_type == "fake_quant":
                logger.info(f"Sensitivity will run on device {self.device}")
            else:
                logger.info(
                    "Quantized sensitivity do not support multigpu, "
                    + "use first device as default."
                )
                self.device = self.device[:1]

        # get analysis leaf ops to run sensitivity
        if self.analysis_model_type == "hbir":
            ops = find_leaf_modules(baseline_model)
        else:
            ops = find_leaf_modules(analysis_model)

        # prepare input data
        index_list = set()
        assert self.input is not None or len(self.bad_index_table) > 0
        if len(self.bad_index_table) == 0:
            logger.info("Use manually set input as example input.")
            index_list.add(-1)
            self.bad_input_dict[-1] = (
                self.input,
                (
                    self.baseline_model_cached_attr,
                    self.analysis_model_cached_attr,
                ),
            )
            # manual set input should apply to all outputs
            device = get_device(self.device[0])
            example_data = _as_tuple(copy.deepcopy(self.input))
            example_data = apply_to_collection(
                example_data, Tensor, lambda x: x.to(device)
            )
            model = deepcopy(self.baseline_model).eval().to(device)
            example_output = model(*example_data)
            example_output = tree_flatten(example_output)[0]
            for i in range(len(example_output)):
                self.bad_index_table.append(
                    [f"manual_set_input_{i}", -1, -1, -1, -1, -1]
                )

        # get flops
        flops_device = get_device(self.device[0])
        model = (
            copy.deepcopy(
                self.baseline_model
                if self.analysis_model_type in ("quantized", "hbir")
                else self.analysis_model
            )
            .eval()
            .to(flops_device)
        )
        example_input, attrs = list(self.bad_input_dict.values())[0]
        data = _as_tuple(copy.deepcopy(example_input))
        data = apply_to_collection(data, Tensor, lambda x: x.to(flops_device))
        _, op_flops_mapping = cal_flops(model, copy.deepcopy(data))

        self.check_model_params(data)

        ret = {}
        support_metrics = ("Cosine", "MSE", "L1", "KL", "SQNR")
        for out_index, index_list in enumerate(self.bad_index_table):
            out_name = index_list[0]
            data_index = index_list[support_metrics.index(metric) + 1]

            logger.info(
                f"Run output {out_name} sensitivity with index {data_index}..."
            )
            example_input, attrs = self.bad_input_dict[data_index]
            baseline_model_cached_attrs, analysis_model_cached_attrs = attrs

            data = _as_tuple(copy.deepcopy(example_input))
            data = apply_to_collection(data, Tensor, lambda x: x.cpu())
            baseline_model_cached_attrs = apply_to_collection(
                baseline_model_cached_attrs, Tensor, lambda x: x.cpu()
            )
            analysis_model_cached_attrs = apply_to_collection(
                analysis_model_cached_attrs, Tensor, lambda x: x.cpu()
            )

            # preapre multi-gpus model
            # Torch DDP and DP both have incompatible errors, directly use
            # python multiprocessing here

            # local function cannot be spawn on multi devices, fallback
            if len(self.device) > 1 and self.post_process is not None:
                logger.warning(
                    "Local postprocess function cannot be spawn on multi "
                    "devices, fallback to single device. To enable multi "
                    "devices, move post process function into model."
                )
                self.device = self.device[:1]
            nprocess = len(self.device)
            if nprocess > 1:
                # CUDA do not support in fork
                mp.set_start_method("spawn", force=True)

                # global value march cannot spawn on each process
                march = get_march()

                # GraphModuleImpl is local variable which cannot spawn
                # convert float module to qat in each process
                qconfig_dict = {
                    name: mod.qconfig
                    for name, mod in analysis_model.named_modules()
                    if hasattr(mod, "qconfig")
                }

                # get prepare method
                if isinstance(analysis_model, GraphModuleWithAttr):
                    prepare_method = PrepareMethod.SYMBOLIC
                elif isinstance(analysis_model, GraphModule):
                    prepare_method = PrepareMethod.JIT_STRIP
                else:
                    prepare_method = PrepareMethod.EAGER

                work_model_info = (
                    copy.deepcopy(baseline_model),
                    qconfig_dict,
                    analysis_model.state_dict(),
                    copy.deepcopy(baseline_model_cached_attrs),
                    copy.deepcopy(analysis_model_cached_attrs),
                    self.analysis_model_type,
                    metric,
                    op_flops_mapping,
                    prepare_method,
                )

                # prepare each process ops
                num_ops = len(ops) // nprocess
                batch_ops = [
                    ops[num_ops * i : num_ops * (i + 1)]
                    for i in range(nprocess)
                ]
                for j, op in enumerate(ops[num_ops * nprocess :]):
                    batch_ops[j].append(op)

                # use queue to gather sensitive table on each process
                queue = Queue()

                threads = [
                    Process(
                        target=_sensitive_worker,
                        args=(
                            self.device[i],
                            work_model_info,
                            batch_ops[i],
                            data,
                            out_index,
                            True if i == 0 else False,
                            queue,
                            march,
                        ),
                    )
                    for i in range(nprocess)
                ]
                try:
                    tables = []
                    for thread in threads:
                        thread.start()
                    for _, thread in enumerate(threads):
                        # get table result of each thread to avoid join here
                        tables.append(queue.get())
                        thread.join()
                    while not queue.empty():
                        tables.append(queue.get())
                    table = SensitivityTable.merge(tables)
                except Exception as e:
                    raise (e)
            else:
                device = get_device(self.device[0])
                data = apply_to_collection(
                    data, Tensor, lambda x: x.to(device)
                )
                baseline_model_cached_attrs = apply_to_collection(
                    baseline_model_cached_attrs, Tensor, lambda x: x.to(device)
                )
                analysis_model_cached_attrs = apply_to_collection(
                    analysis_model_cached_attrs, Tensor, lambda x: x.to(device)
                )
                sensitive_model = Sensitivity(
                    baseline_model,
                    analysis_model,
                    baseline_model_cached_attrs,
                    analysis_model_cached_attrs,
                    self.analysis_model_type,
                    metric,
                    device,
                    op_flops_mapping,
                )
                sensitive_model.to(device)
                sensitive_model(data, ops, out_index)
                table = sensitive_model.sensitivity_table

            table.sort(reverse)
            table.dump(self.out_dir, out_name)
            ret[f"{out_name}_{table.metric}"] = table.get()
            # sensitive_model.auto_find_topk(
            #     data, out_name, out_index, self.out_dir
            # )
        return ret

    def clean(self):
        """Clean all pt files in tmp dir."""
        m1, m2 = self._load_models_meta()
        m1.clean()
        m2.clean()
