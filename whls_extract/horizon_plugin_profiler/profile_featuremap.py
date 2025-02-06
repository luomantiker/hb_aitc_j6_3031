import os
from typing import Dict, List, Optional, Tuple

from horizon_plugin_profiler.utils.logger import format_msg
from horizon_plugin_profiler.utils.typeguard import typechecked
from horizon_plugin_profiler.utils.version_helper import (
    check_torch_numpy_version,
)

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tabulate import tabulate
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from horizon_plugin_pytorch.qtensor import QTensor


@typechecked
def profile_featuremap(
    featuremap: List[Dict],
    with_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None,
    print_per_channel_scale: bool = False,
    show_per_channel: bool = False,
    out_dir: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Tuple]:
    """Profile featuremap value with log or tensorboard.

    Print min/max/mean/var/scale of each feature profiled by `get_raw_features`
    by default. If `with_tensorboard` set True, the histogram of each feature
    will be shown in tensorboard, which is useful to see the data distribution.

    If you want to get more info about features, you can define your custom
    profile functions to process the results of `get_raw_features`.

    Args:
        featuremap: raw featuremaps returned by `get_raw_features`
        with_tensorboard: whether to use tensorboard. Default: False
        tensorboard_dir: tensorboard log file path. Default: None
        print_per_channel_scale: whether to print per channel scales.
            Default: False
        show_per_channel: show each featuremap in per channel ways
            in tensorboard. Default: False
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None
        file_name: result file name. If None, will save result and fig with
            name 'statistic'.(statistic.txt and statistic.html). Default: None

    Returns:
        A List of tuple. Each tuple is each layer statistic in format
        (index, module name, module type, attr, min, max, mean, var, scale)
    """
    table = []
    per_channel_table = []
    nan_inf_table = []

    index = -1
    pre_name = None
    for f in featuremap:
        if pre_name != f["module_name"]:
            index += 1
            pre_name = f["module_name"]
        fdata = f["data"].as_subclass(Tensor).float()
        if torch.any(torch.isnan(fdata)):
            nan_inf_table.append(
                (index, f["module_name"], f["module_type"], f["attr"], "NaN")
            )
        if torch.any(torch.isinf(fdata)):
            nan_inf_table.append(
                (index, f["module_name"], f["module_type"], f["attr"], "Inf")
            )
        if f["scale"] is None:
            # maybe last qat conv out or float
            table.append(
                (
                    index,
                    f["module_name"],
                    f["module_type"],
                    f["attr"],
                    fdata.min().item(),
                    fdata.max().item(),
                    fdata.mean().item(),
                    fdata.var().item(),
                    None,
                    f["dtype"],
                )
            )
        elif f["scale"].numel() == 1:
            s = f["scale"] if f["data"].is_quantized else 1.0
            table.append(
                (
                    index,
                    f["module_name"],
                    f["module_type"],
                    f["attr"],
                    (fdata.min() * s).item(),
                    (fdata.max() * s).item(),
                    (fdata.mean() * s).item(),
                    (fdata.var() * s * s).item(),
                    f["scale"].item(),
                    f["dtype"],
                )
            )
        else:
            fdata = f["data"].dequantize()
            table.append(
                (
                    index,
                    f["module_name"],
                    f["module_type"],
                    f["attr"],
                    fdata.min().item(),
                    fdata.max().item(),
                    fdata.mean().item(),
                    fdata.var().item(),
                    "per channel scale",
                    f["dtype"],
                )
            )
            per_channel_table.append(
                (
                    index,
                    f["module_name"],
                    f["module_type"],
                    f["attr"],
                    len(f["scale"]),
                    f["ch_axis"],
                    f["scale"].cpu(),
                    f["dtype"],
                )
            )
    # in symmetric quantization, scale is determined by max value
    ordered_table = sorted(
        table, key=lambda x: max(abs(x[5]), abs(x[4])), reverse=True
    )
    out_dir = "." if out_dir is None else out_dir
    file_path = (
        os.path.join(out_dir, "statistic" if file_name is None else file_name)
        + ".txt"
    )
    fig_path = (
        os.path.join(out_dir, "statistic" if file_name is None else file_name)
        + ".html"
    )
    with open(file_path, "w") as f:
        if nan_inf_table:
            f.write("This layers data may be unusual, please check...\n")
            f.write(
                tabulate(
                    nan_inf_table,
                    headers=[
                        "Module Index",
                        "Module Name",
                        "Module Type",
                        "Input/Output/Attr",
                        "NaN or Inf",
                    ],
                    tablefmt="psql",
                )
            )
        f.write(
            tabulate(
                table,
                headers=[
                    "Module Index",
                    "Module Name",
                    "Module Type",
                    "Input/Output/Attr",
                    "Min",
                    "Max",
                    "Mean",
                    "Var",
                    "Scale",
                    "Dtype",
                ],
                tablefmt="psql",
                floatfmt=".7f",
                numalign="left",
            )
        )
        f.write("\n\nStatistics with quant range in descending order...\n")
        f.write(
            tabulate(
                ordered_table,
                headers=[
                    "Module Index",
                    "Module Name",
                    "Module Type",
                    "Input/Output/Attr",
                    "Min",
                    "Max",
                    "Mean",
                    "Var",
                    "Scale",
                    "Dtype",
                ],
                tablefmt="psql",
                floatfmt=".7f",
                numalign="left",
            )
        )

        if print_per_channel_scale:
            f.write(
                tabulate(
                    per_channel_table,
                    headers=[
                        "Module Index",
                        "Module Name",
                        "Module Type",
                        "Input/Output",
                        "Channel Len",
                        "Channel Axis",
                        "Scale",
                        "Dtype",
                    ],
                    tablefmt="grid",
                    numalign="left",
                )
            )

    if nan_inf_table:
        print(
            format_msg(
                "This layers data may be unusual, please check...", "red"
            )
        )
        print(
            tabulate(
                nan_inf_table,
                headers=[
                    "Module Index",
                    "Module Name",
                    "Module Type",
                    "Input/Output/Attr",
                    "NaN or Inf",
                ],
                tablefmt="psql",
            )
        )
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(
            x=[
                table[i][0]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            y=[
                table[i][5]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            mode="lines+markers",
            hovertext=[
                "{}: {}".format(table[i][1], str(table[i][2]))
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            name="max",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[
                table[i][0]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            y=[
                table[i][6]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            mode="lines+markers",
            hovertext=[
                "{}: {}".format(table[i][1], str(table[i][2]))
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            name="mean",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[
                table[i][0]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            y=[
                table[i][4]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            mode="lines+markers",
            hovertext=[
                "{}: {}".format(table[i][1], str(table[i][2]))
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            name="min",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[
                table[i][0]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            y=[
                table[i][7]
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            # mode="lines+markers",
            hovertext=[
                "{}: {}".format(table[i][1], str(table[i][2]))
                for i in range(len(table))
                if table[i][3] == "output"
            ],
            name="var",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        xaxis_title="Module Index",
        yaxis_title="min/max/mean",
        yaxis2_title="var",
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(255,255,255,0.75)",
            "font": {"color": "black"},
        },
    )
    fig.write_html(fig_path, auto_open=False)

    if with_tensorboard:
        check_torch_numpy_version()
        writer = SummaryWriter(log_dir=tensorboard_dir)
        for f in featuremap:
            f["data"] = (
                f["data"].dequantize()
                if isinstance(f["data"], QTensor)
                else f["data"]
            )
            # shown per channel quantized weight and features
            if f["ch_axis"] != -1:
                for i in range(f["data"].shape[f["ch_axis"]]):
                    writer.add_histogram(
                        f"{f['module_name']}:{f['attr']}",
                        # ch_axis = 0 or 1
                        f["data"][i] if f["ch_axis"] == 0 else f["data"][:, i],
                        i,
                    )
            # tensorboard histogram result is confused when only one number
            elif f["data"].numel() != 1:
                # show data histogram in per channel type
                if show_per_channel and f["data"].ndim >= 2:
                    ch_axis = 1
                    for i in range(f["data"].shape[ch_axis]):
                        writer.add_histogram(
                            f"{f['module_name']}:{f['attr']}",
                            f["data"][:, i],
                            i,
                        )
                else:
                    writer.add_histogram(
                        f"{f['module_name']}:{f['attr']}", f["data"]
                    )
        writer.close()

    return table
