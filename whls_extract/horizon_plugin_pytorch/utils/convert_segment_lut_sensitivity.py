import os

import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
from torch import nn
from torch.utils._pytree import tree_flatten
from tqdm import tqdm


def cos_eval_func(origin_ret, current_ret):
    origin_ret = tree_flatten(origin_ret)[0][0]
    current_ret = tree_flatten(current_ret)[0][0]
    return nn.CosineSimilarity(dim=0)(
        origin_ret.flatten(), current_ret.flatten()
    ).item()


def l1_eval_func(origin_ret, current_ret):
    origin_ret = tree_flatten(origin_ret)[0][0]
    current_ret = tree_flatten(current_ret)[0][0]
    return (origin_ret - current_ret).abs().max().item()


def convert_segment_lut_sensitivity(
    model: nn.Module, analysis_input, eval_func=None
):
    if eval_func is None:
        eval_func = cos_eval_func

    origin_ret = model(*analysis_input)

    from horizon_plugin_pytorch.nn.qat.segment_lut import (
        QuantizedQATSegmentLUT,
    )

    QuantizedQATSegmentLUT.convert_segment_lut(model, enabled=False)

    lut_names = []
    for n, m in model.named_modules():
        if isinstance(m, QuantizedQATSegmentLUT):
            lut_names.append(n)

    headers = (
        "name",
        "diff",
        "in_dtype",
        "in_scale",
        "out_dtype",
        "out_scale",
    )
    records = []
    pbar = tqdm(total=len(lut_names))
    for n in lut_names:
        lut_mod = model.get_submodule(n)
        lut_mod.quantized_forward = True
        current_ret = model(*analysis_input)
        lut_mod.quantized_forward = False
        records.append(
            (
                n,
                eval_func(origin_ret, current_ret),
                lut_mod.input_dtype,
                lut_mod.input_scale,
                lut_mod.output_dtype,
                lut_mod.output_scale,
            )
        )
        pbar.update(1)
    pbar.close()

    records = sorted(records, key=lambda x: x[1])

    print(tabulate(records, headers))

    QuantizedQATSegmentLUT.recover_qat_segment_lut(model)


def convert_segment_lut_single_op_diff(
    model: nn.Module, analysis_input, eval_func=None, save_fig_dir=None
):
    from horizon_plugin_pytorch.nn.qat.segment_lut import (
        QuantizedQATSegmentLUT,
    )

    QuantizedQATSegmentLUT.convert_segment_lut(model, enabled=False)

    headers = (
        "name",
        "l1_in_scale",
        "diff",
        "in_dtype",
        "in_scale",
        "out_dtype",
        "out_scale",
    )
    records = []

    if save_fig_dir is not None:
        os.makedirs(save_fig_dir, exist_ok=True)

    def draw_scatter(input, qat_ret, quantized_ret, save_name, title):
        diff = quantized_ret - qat_ret
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        if isinstance(qat_ret, torch.Tensor):
            qat_ret = qat_ret.cpu().numpy()
        if isinstance(quantized_ret, torch.Tensor):
            quantized_ret = quantized_ret.cpu().numpy()
        if isinstance(diff, torch.Tensor):
            diff = diff.cpu().numpy()

        plt.figure(figsize=(18, 6))
        plt.suptitle(title)

        plt.subplot(1, 3, 1)
        plt.scatter(input, qat_ret)
        plt.title("qat")

        plt.subplot(1, 3, 2)
        plt.scatter(input, quantized_ret)
        plt.title("quantized")

        plt.subplot(1, 3, 3)
        plt.scatter(input, diff)
        plt.title("diff")

        plt.savefig(os.path.join(save_fig_dir, save_name))
        plt.close()

        print("saving {}".format(os.path.join(save_fig_dir, save_name)))

    def gen_record_hook(mod_name):
        def forward_hook(mod, args, output):
            assert not mod.quantized_forward
            input = args[0]
            input = input.to_quantized()
            ret = mod.quantized_mod(input)
            quantized_ret = ret.dequantize()

            l1_diff = l1_eval_func(
                output.dequantize(), quantized_ret.dequantize()
            )

            records.append(
                (
                    mod_name,
                    l1_diff / mod.output_scale,
                    (
                        l1_diff
                        if eval_func is None
                        else eval_func(
                            output.dequantize(), quantized_ret.dequantize()
                        )
                    ),
                    mod.input_dtype,
                    mod.input_scale,
                    mod.output_dtype,
                    mod.output_scale,
                )
            )

            if save_fig_dir is not None:
                input_min = mod.input_dtype.min * mod.input_scale
                input_max = mod.input_dtype.max * mod.input_scale
                device = output.device

                example_input = torch.arange(
                    input_min.item(),
                    input_max.item(),
                    (input_max - input_min).item() / 100,
                    device=device,
                )
                from horizon_plugin_pytorch import QTensor

                example_input = QTensor(
                    example_input, input.q_scale(), input.dtype
                )

                qat_ret = mod.qat_mod(example_input).dequantize()
                quantized_ret = mod.quantized_mod(
                    example_input.to_quantized()
                ).dequantize()

                draw_scatter(
                    example_input.dequantize(),
                    qat_ret,
                    quantized_ret,
                    "{}.jpeg".format(mod_name),
                    "{} input: {} {}, output {} {}".format(
                        mod_name,
                        mod.input_dtype,
                        mod.input_scale.item(),
                        mod.output_dtype,
                        mod.output_scale.item(),
                    ),
                )

        return forward_hook

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, QuantizedQATSegmentLUT):
            handles.append(m.register_forward_hook(gen_record_hook(n)))

    model(*analysis_input)

    for h in handles:
        h.remove()

    records = sorted(records, key=lambda x: x[1], reverse=True)

    print(tabulate(records, headers))

    QuantizedQATSegmentLUT.recover_qat_segment_lut(model)
