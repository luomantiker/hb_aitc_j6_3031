import copy
import os
from typing import Any, Callable, Optional

from horizon_plugin_profiler.utils.model_helper import (
    attach_qualified_name,
    register_hook_on_leaf,
    swap_ff_with_horizonff,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import plotly.graph_objects as go
import torch


@typechecked
def show_cuda_memory_consumption(
    model: torch.nn.Module,
    example_inputs: Any,
    device: torch.device,
    check_leaf_module=None,
    out_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    custom_backward: Optional[Callable] = None,
):
    """
    Evaluate memory consumption of a model during forward and backward.

    Result will be saved as html file.

    Known Issue: If checkpoint is used, some result of backward will named
    as 'forward', because during backward the forward hook is called,
    rather than backward hook.

    Args:
        model: The input model.
        example_inputs (Any[Tensor]): The input data feed to model.
        device: Evaluate on this device.
        check_leaf_module: A function to check if a module is leaf. Pass None
            to use pre-defined `is_leaf_module`. Defaults to None.
        out_dir: path to save the result. If None, will save in the current
            directory. Default: None
        file_name: result file name. If None, will save result with
            name 'mem_info'. Default: None
        custom_backward: Run backward by the model ret,
            must set retain_graph=False. Defaults to None.
            This param must be provided when model ret is not a single Tensor.
    """
    assert (
        device.type == "cuda"
    ), "show_cuda_memory_consumption only works on gpu"

    model = copy.deepcopy(model).to(device)
    swap_ff_with_horizonff(model)
    attach_qualified_name(model)
    torch.autograd.set_detect_anomaly(True)

    if len(list(model.parameters())) > 0:
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0001)
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer = None

    mem_info = []

    mem_params = sum(
        [
            param.nelement() * param.element_size()
            for param in model.parameters()
        ]
    )
    mem_bufs = sum(
        [buf.nelement() * buf.element_size() for buf in model.buffers()]
    )
    model_mem = mem_params + mem_bufs  # in bytes

    # contents not related to this model
    base_consumption = torch.cuda.memory_allocated(device) - model_mem

    print(
        "There is a base memory consumption of {} Bytes, "
        "and will be subtracted in the result.".format(base_consumption)
    )

    def max_memory_allocated_by_model():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - base_consumption

    def _forward_hook(mod, input, output):
        mem_info.append(
            (
                mod._qualified_name + "_forward",
                max_memory_allocated_by_model(),
            )
        )
        torch.cuda.reset_peak_memory_stats()

    def _backward_hook(mod, grad_input, grad_output):
        mem_info.append(
            (
                mod._qualified_name + "_backward",
                max_memory_allocated_by_model(),
            )
        )
        torch.cuda.reset_peak_memory_stats()

    register_hook_on_leaf(
        model,
        _forward_hook,
        None,
        _backward_hook,
        check_leaf_module=check_leaf_module,
    )

    torch.cuda.reset_peak_memory_stats()
    mem_info.append(("Start", max_memory_allocated_by_model()))
    torch.cuda.reset_peak_memory_stats()

    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)

    ret = model(*example_inputs)

    torch.cuda.reset_peak_memory_stats()
    mem_info.append(("After_Forward", max_memory_allocated_by_model()))
    torch.cuda.reset_peak_memory_stats()

    if custom_backward is None:
        if not isinstance(ret, torch.Tensor):
            raise ValueError(
                "Please provide custom_backward function when"
                " model ret is not a single Tensor"
            )
        mask = torch.rand_like(ret)
        ret.mul(mask).sum().backward(retain_graph=False)
    else:
        custom_backward(ret)

    mem_info.append(("Backward", max_memory_allocated_by_model()))
    torch.cuda.reset_peak_memory_stats()

    del ret

    torch.cuda.reset_peak_memory_stats()
    mem_info.append(("Release_Ret", max_memory_allocated_by_model()))
    torch.cuda.reset_peak_memory_stats()

    if optimizer is not None:
        optimizer.step()

    mem_info.append(("Update", max_memory_allocated_by_model()))

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    mem_info.append(("After_ZeroGrad", max_memory_allocated_by_model()))
    torch.cuda.reset_peak_memory_stats()

    torch.autograd.set_detect_anomaly(False)

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(range(len(mem_info))),
                y=[mem_info[i][1] for i in range(len(mem_info))],
                # mode="lines+markers",
                hovertext=[mem_info[i][0] for i in range(len(mem_info))],
                name="var",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Memory Usage(Byte)",
        hovermode="x unified",
    )
    out_path = os.path.join(
        out_dir or ".", (file_name or "mem_info") + ".html"
    )
    fig.write_html(out_path, auto_open=False)

    return torch.tensor([value for stage, value in mem_info]).max().item()
