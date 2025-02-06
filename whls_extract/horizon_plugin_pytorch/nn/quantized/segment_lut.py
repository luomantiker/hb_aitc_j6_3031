import logging
import warnings

import torch
from torch import Tensor
from torch.nn import Module

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.dtype import qinfo, qint8, qint16
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn import qat
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.script_quantized_fn import (
    script_quantized_fn,
)
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import lut, quantize, segment_lut

logger = logging.getLogger(__name__)


def _arange(start, stop, step, device=None, output_length=None):
    if isinstance(start, Tensor):
        device = start.device
        start = start.item()
    if isinstance(stop, Tensor):
        device = stop.device
        stop = stop.item()
    if isinstance(step, Tensor):
        device = step.device
        step = step.item()
    if step == 0:
        return torch.full((output_length,), start, device=device)
    else:
        return torch.arange(start, stop, step, device=device)


def _generate_single_table(func, input_scale, output_scale):
    x = _arange(-128, 128, 1, input_scale.device)
    y = func(x * input_scale)

    return quantize(
        y,
        output_scale,
        torch.zeros_like(output_scale, dtype=torch.long),
        -1,
        qint8,
    )


def _get_linear_kb_by_points(x1, x2, y1, y2):
    diffx = x1 - x2
    diffy = y1 - y2
    if diffx == 0:
        k = torch.zeros_like(x1)
    else:
        k = diffy / diffx
    b = y1 - x1 * k
    return k, b


def _get_linear_kb(func, x1, x2):
    # If func has constant float arguments, like eps=1e-5, this eps will be
    # traced as torch.tensor(1e-5, dtype=torch.float64) in scriptmodule. If
    # func is x + eps and the saved scriptmodule pt file is loaded and run by
    # torch, this func out is always fp32. While in hbdk parser, this func is
    # computed by torch.add with unexpected behavior
    #   tensor(x) + tensor(1e-5, dtype=fp64) = tensor(y, dtype=fp64)
    # So, change tensor(x) to tensor([x]) here, make sure
    #   tensor([x]) + tensor(1e-5, dtype=fp64) = tensor([y], dtype=fp32)
    return _get_linear_kb_by_points(
        x1, x2, func(x1.reshape(-1)), func(x2.reshape(-1))
    )


def _convert_linear_kb(k, b, input_scale, output_scale):
    # (x / input_scale * int_k + (int_b << left_shift)) >> right_shift
    #     = (x * k + b) / output_scale
    # x / input_scale * int_k + (int_b << left_shift)
    #     = (x * k / output_scale + b / output_scale) << right_shift
    # int_k / input_scale = (k / output_scale) << right_shift
    # int_b << left_shift = (b / output_scale) << right_shift
    # int_k >> right_shift = (k / output_scale * input_scale)
    # int_b << left_shift = (b / output_scale) << right_shift
    int_k, neg_right_shift = torch.ops.horizon.toqint(
        k / output_scale * input_scale,  # x
        16,  # qbits
        16,  # max_bits
        False,  # allow_left_shift
        True,  # allow_right_shift
    )
    right_shift = -neg_right_shift

    # limit int_b << left_shift to int31
    max_right_shift = max(
        30 - torch.log2((b / output_scale).abs() + 1).ceil().to(torch.int32), 0
    )
    if right_shift > max_right_shift:
        int_k = (int_k >> (right_shift - max_right_shift)).to(
            dtype=torch.int32
        )
        right_shift[:] = max_right_shift

    int_b, left_shift = torch.ops.horizon.toqint(
        b / output_scale * (1 << right_shift.item()),  # x
        16,  # qbits
        31,  # max_bits
        True,  # allow_left_shift
        False,  # allow_right_shift
    )
    return int_k, int_b, left_shift, right_shift


def _generate_table(
    func, xmin, xmax, output_scale, output_dtype=qint16, table_length=64
):
    step = (xmax - xmin) / (table_length - 1)
    x = _arange(xmin, xmax + step * 0.5, step=step, output_length=table_length)
    y = func(x)

    return quantize(
        y,
        output_scale,
        torch.zeros_like(output_scale, dtype=torch.long),
        -1,
        output_dtype,
    )


def _get_optimized_dividing_points(
    func, input_float_min, input_float_max, strategy, accu_divide_num=256
):
    if strategy == "curvature":
        # use curvature to decide dividing points
        step = (input_float_max - input_float_min) / accu_divide_num
        x = _arange(
            input_float_min,
            input_float_max + step * 0.5,
            step,
            output_length=accu_divide_num + 1,
        )
        y = func(x)
        dy = (y[1:] - y[:-1]) / step
        ddy = (dy[1:] - dy[:-1]) / step
        if ddy.isnan().sum() > 0:
            raise ValueError("input_max = input_min, please check the model")
        ddy = ddy.abs()
        ddy = torch.cat([ddy, ddy[-1:]])
        accumulate = torch.cumsum(ddy, dim=0)
        segment_idx = accumulate.div(accumulate[-1] / 6, rounding_mode="floor")

        dividing_points = input_float_min.reshape(1)
        for i, p in zip(segment_idx + 1, x[1:]):
            if i > 6:
                break
            if i > dividing_points.numel():
                # constraint segment range not smaller than step
                dividing_points = torch.cat([dividing_points, p.reshape(1)])

        if len(dividing_points) < 6:
            warnings.warn(
                "The curvature of the function that segment_lut is simulating"
                " too vertical, which may raise unexpected precision gap"
                " between qat and quantized models. Please adjust your input"
                " range or try other dividing_points generation strategy."
            )
        tail_list = [input_float_max.reshape(1)] * (7 - len(dividing_points))

        dividing_points = torch.cat([dividing_points, x[-1:]] + tail_list)

        return dividing_points

    elif strategy == "evenly":
        step = (input_float_max - input_float_min) / 6
        return _arange(
            input_float_min,
            input_float_max + step * 1.5,
            step,
            output_length=8,
        )
    else:
        raise ValueError("Unsupported strategy")


def _get_exp_input_range(input_scale, input_dtype, output_scale, output_dtype):
    input_info = qinfo(input_dtype)
    output_info = qinfo(output_dtype)

    input_min = input_scale * input_info.min
    input_max = torch.min(
        input_scale * input_info.max, torch.log(output_scale * output_info.max)
    )

    return input_min, input_max


_input_range_mapping = {torch.exp: _get_exp_input_range}


class SegmentLUT(Module):
    out_scale: Tensor
    out_zero_point: Tensor

    _version = 2
    _QAT_MODULE = qat.SegmentLUT

    def __init__(
        self,
        simulated_func,
        is_centrosymmetric=False,
        dividing_points=None,
        input_range=None,
        auto_divide_strategy="evenly",
        inverse_func=None,
        gradients=None,
        output_scale=None,
        output_dtype=qint16,
        device=None,
        num_entries_per_table=64,
    ):
        super(SegmentLUT, self).__init__()

        assert output_dtype in (qint8, qint16)
        self.num_entries_per_table = num_entries_per_table
        if device is None:
            device = torch.device("cpu")
        output_scale = output_scale.clone().detach().to(device=device)

        self.simulated_func = simulated_func
        self.is_centrosymmetric = is_centrosymmetric
        self.dividing_points = dividing_points
        self.input_range = input_range
        self.auto_divide_strategy = auto_divide_strategy
        # only used for monotonically decreasing function to
        # generate input range
        self.inverse_func = inverse_func
        self.gradients = gradients
        self.output_dtype = output_dtype
        self.idx_bits = 8

        self.register_buffer(
            "out_scale",
            torch.tensor(output_scale.clone().detach(), dtype=torch.float32),
        )
        self.register_buffer(
            "out_zero_point",
            torch.zeros(1, dtype=torch.long, device=output_scale.device),
        )

        self.handle_load_state_dict()

    def handle_load_state_dict(self):
        def _allow_miss_key_hook(
            state_dict: dict,
            prefix: str,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            ignore_keys = [
                "single_table",
                "table",
                "scale",
                "beta",
                "left_shift",
                "right_shift",
                "max",
            ]

            msg = (
                "LUT params are generated on the fly instead of saved in "
                + "buffers since version 0.14.7, please update the "
                + "quantized ckpt to avoid this warning"
            )

            have_unexpected_keys = False
            for ignore_key in ignore_keys:
                if prefix + ignore_key in state_dict:
                    state_dict.pop(prefix + ignore_key)
                    have_unexpected_keys = True
            if have_unexpected_keys:
                logger.warning(msg)

        self._register_load_state_dict_pre_hook(_allow_miss_key_hook)

    @script_quantized_fn
    def _init_single_table_params(
        self,
        input_scale,
    ):
        table = _generate_single_table(
            self.simulated_func, input_scale.cpu(), self.out_scale.cpu()
        )
        return table

    @script_quantized_fn
    def _init_multi_table_params(
        self,
        input_scale: Tensor,
        input_dtype,
    ):
        input_scale = input_scale.cpu()
        device = torch.device("cpu")

        table = torch.zeros(
            (6, self.num_entries_per_table), dtype=torch.int16, device=device
        )
        alpha = torch.zeros(8, dtype=torch.int16, device=device)
        beta = torch.zeros(8, dtype=torch.int16, device=device)
        left_shift = torch.zeros(8, dtype=torch.int8, device=device)
        right_shift = torch.zeros(8, dtype=torch.int8, device=device)

        info = qinfo(input_dtype)
        # get input min max
        compute_input_range = _input_range_mapping.get(
            self.simulated_func, None
        )
        if self._version >= 2 and compute_input_range is not None:
            input_float_min, input_float_max = compute_input_range(
                input_scale,
                input_dtype,
                self.out_scale.cpu(),
                self.output_dtype,
            )
        else:
            input_float_min = info.min * input_scale
            input_float_max = info.max * input_scale
            # generate monotonically function input range
            # for monotonically decreasing function f(x):
            #   input_range: [qin_min * s_in, qin_max * s_in]
            #   output_range: [f(qin_max * s_in), f(qin_min * s_in)]
            #                =[qout_min * s_out, qout_max * s_out]
            #   so: input_min = f^{-1}(qout_max * s_out)
            #       input_max = qin_max * s_in
            #
            # for monotonically increasing function f(x):
            #   input_range: [qin_min * s_in, qin_max * s_in]
            #   output_range: [f(qin_min * s_in), f(qin_max * s_in)]
            #                =[qout_min * s_out, qout_max * s_out]
            #   so: input_min = f^{-1}(qout_min * s_out)
            #       input_max = qin_max * s_in
            # Note: this range is the theoretically upper bound of
            #   input range.
            #   If want more precise result, try to specify input range by
            #   parameter input_range
            if self.inverse_func is not None:
                # use input_float_max / 2 and input_float_max / 4 to avoid
                # no definition value in endpoints
                input_float_min = (
                    self.inverse_func(
                        qinfo(self.output_dtype).max * self.out_scale.cpu()
                    )
                    if self.simulated_func(input_float_max / 4)
                    > self.simulated_func(input_float_max / 2)
                    else self.inverse_func(
                        qinfo(self.output_dtype).min * self.out_scale.cpu()
                    )
                )

        if self.input_range is not None:
            setted_input_min = self.input_range[0]
            setted_input_max = self.input_range[1]
            if setted_input_min is not None:
                setted_input_min = torch.tensor(
                    setted_input_min, device=device, dtype=torch.float32
                )
                input_float_min = torch.maximum(
                    input_float_min, setted_input_min
                )
            if setted_input_max is not None:
                setted_input_max = torch.tensor(
                    setted_input_max, device=device, dtype=torch.float32
                )
                input_float_max = torch.minimum(
                    input_float_max, setted_input_max
                )
        if hasattr(self, "_input_observer"):
            input_name, observer = self._input_observer
            if observer.min_val.numel() != 0:
                input_float_min = torch.maximum(
                    input_float_min, observer.min_val
                )
            if observer.max_val.numel() != 0:
                input_float_max = torch.minimum(
                    input_float_max, observer.max_val
                )
            logger.info(
                f"Input float min max updated by input {input_name} observer: "
                f"[{input_float_min}, {input_float_max}]"
            )

        if self.is_centrosymmetric and input_float_min < 0:
            input_float_min[:] = 0

        # generate dividing_points
        if self.dividing_points is None:
            dividing_points = _get_optimized_dividing_points(
                self.simulated_func,
                input_float_min,
                input_float_max,
                self.auto_divide_strategy,
            )
        else:
            dividing_points = torch.tensor(self.dividing_points, device=device)

        # generate int params
        segment_max = (
            (dividing_points / input_scale)
            .round()
            .clamp(info.min, info.max)
            .to(torch.int16)
        )
        segment_max[-1] = info.max

        # must recompute dividing points according to max !
        dividing_points = segment_max * input_scale

        if self.gradients is None:
            left_k, right_k = None, None
        else:
            left_k, right_k = self.gradients
        # params for left linear fit
        if left_k is not None:
            k = torch.tensor(left_k, device=dividing_points.device)
            b = self.simulated_func(dividing_points[0])
        elif input_float_min == dividing_points[0]:
            k, b = _get_linear_kb(
                self.simulated_func,
                input_float_min,
                (dividing_points[1] - dividing_points[0])
                / (self.num_entries_per_table - 1)
                + dividing_points[0],
            )
        else:
            k, b = _get_linear_kb(
                self.simulated_func, input_float_min, dividing_points[0]
            )
        int_k, int_b, lshift, rshift = _convert_linear_kb(
            k, b, input_scale, self.out_scale.cpu()
        )
        alpha[0] = int_k
        beta[0] = int_b
        left_shift[0] = lshift
        right_shift[0] = rshift

        # params for right linear fit
        if right_k is not None:
            k = torch.tensor(right_k, device=dividing_points.device)
            b = self.simulated_func(dividing_points[-2])
        elif input_float_max == dividing_points[-1]:
            k, b = _get_linear_kb(
                self.simulated_func,
                (dividing_points[-2] - dividing_points[-1])
                / (self.num_entries_per_table - 1)
                + dividing_points[-1],
                input_float_max,
            )
        else:
            k, b = _get_linear_kb(
                self.simulated_func,
                dividing_points[-1],
                input_float_max,
            )
        int_k, int_b, lshift, rshift = _convert_linear_kb(
            k, b, input_scale, self.out_scale.cpu()
        )
        alpha[-1] = int_k
        beta[-1] = int_b
        left_shift[-1] = lshift
        right_shift[-1] = rshift

        # params for segment lut
        table_list = []
        for i in range(1, 7):
            xmin = dividing_points[i - 1]
            xmax = dividing_points[i]
            # WAR(yushu.gao): hbdk can't parse aten::select correctly,
            # use aten::stack instead. Revert me when hbdk fix select error.
            # table[i - 1].copy_(
            #     _generate_table(
            #         self.simulated_func,
            #         xmin,
            #         xmax,
            #         self.out_scale,
            #         self.output_dtype,
            #     )
            # )
            table_i = _generate_table(
                self.simulated_func,
                xmin,
                xmax,
                self.out_scale.cpu(),
                self.output_dtype,
                self.num_entries_per_table,
            )
            table_list.append(table_i)
            k, b = _get_linear_kb_by_points(
                xmin, xmax, 0, (self.num_entries_per_table - 1)
            )
            int_k, int_b, lshift, rshift = _convert_linear_kb(
                k, b, input_scale, 1.0 / (1 << self.idx_bits)
            )
            alpha[i] = int_k
            beta[i] = int_b
            left_shift[i] = lshift
            right_shift[i] = rshift
        table = (torch.stack(table_list)).to(torch.int16)
        return (
            table,
            alpha,
            beta,
            left_shift,
            right_shift,
            segment_max,
        )

    @typechecked
    def forward(self, input: QTensor) -> QTensor:
        if input.dtype == qint8 and self.output_dtype == qint8:
            ret = lut(
                input.as_subclass(Tensor),
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                self._init_single_table_params(input.q_scale()),
                self.out_scale,
                self.out_zero_point,
                self.output_dtype,
            )
        else:
            (
                table,
                alpha,
                beta,
                left_shift,
                right_shift,
                segment_max,
            ) = self._init_multi_table_params(input.q_scale(), input.dtype)
            if self.num_entries_per_table != 64:
                # call impl directly to make torchsript consistent with hbdk
                ret = hz.nn.quantized.functional_impl._segment_lut(
                    input.as_subclass(Tensor),
                    table,
                    alpha,
                    beta,
                    left_shift,
                    right_shift,
                    segment_max,
                    self.is_centrosymmetric,
                    input.q_scale(),
                    input.q_zero_point(),
                    input.dtype,
                    self.out_scale,
                    self.out_zero_point,
                    self.output_dtype,
                    self.num_entries_per_table,
                    get_march(),
                )
            else:
                ret = segment_lut(
                    input.as_subclass(Tensor),
                    table,
                    alpha,
                    beta,
                    left_shift,
                    right_shift,
                    segment_max,
                    self.is_centrosymmetric,
                    input.q_scale(),
                    input.q_zero_point(),
                    input.dtype,
                    self.out_scale,
                    self.out_zero_point,
                    self.output_dtype,
                )
        return QTensor(ret, self.out_scale, self.output_dtype)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        get_version(self, prefix, local_metadata)

        super(SegmentLUT, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def activated(cls):
        return cls._QAT_MODULE.activated()

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module."""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        assert (
            mod.activation_post_process
        ), "qat mod  must have activation_post_process"

        quantized_mod = cls(
            simulated_func=mod.simulated_func,
            is_centrosymmetric=mod.is_centrosymmetric,
            dividing_points=mod.dividing_points,
            input_range=mod.input_range,
            auto_divide_strategy=mod.auto_divide_strategy,
            inverse_func=mod.inverse_func,
            gradients=mod.gradients,
            output_scale=mod.activation_post_process.scale,
            output_dtype=mod.activation_post_process.dtype,
            device=mod.activation_post_process.scale.device,
            num_entries_per_table=mod.num_entries_per_table,
        )

        quantized_mod._version = mod._version

        return quantized_mod
