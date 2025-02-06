import os
from numbers import Real
from typing import Any, Callable, List, Optional, Union

from horizon_plugin_profiler.utils.logger import format_msg
from horizon_plugin_profiler.utils.typeguard import typechecked

import plotly.graph_objects as go
import torch
from tabulate import tabulate
from torch.nn.modules.utils import _pair

from ._similarity import featuremap_similarity as _fs


@typechecked
def featuremap_similarity(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    inputs: Any,
    similarity_func: Union[str, Callable] = "Cosine",
    threshold: Optional[Real] = None,
    devices: Union[torch.device, tuple, None] = None,
    out_dir: Optional[str] = None,
) -> List[List]:
    """
    Compute the similarity of feature maps.

    The input models can be float/fused/calibration/qat/quantized model.

    Args:
        model1: can be float/fused/calibration/qat/quantized model
        model2: can be float/fused/calibration/qat/quantized model
        inputs: the input data feed to model
        similarity_func: similarity computation function. Support "Cosine",
            "MSE", "L1", "KL", "SQNR", or any user-defined Callable object. If
            it is a user-defined object, it should return a scalar or tensor
            with only one number. Otherwise the result shown may be unexpected.
            Default: "Cosine"
        threshold: if similarity value exceeds or less than this threshold,
            the featuremap info will be shown on the screen. If threshold is
            none, it will be set to different values according to different
            similarity functions. Default: None
        devices: run model on which devices (cpu, gpu). If can be:
            1) None. Run model with given inputs;
            2) torch.device. Both models and given inputs will be moved on this
                specified device;
            3) tuple. A tuple of 2 torch.devices. The two models will be moved
                on specified devices seperatedly. It may be used to compare the
                CPU and GPU results difference.
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None

    Returns:
        A List of list. Each list is each layer similarity info in format
        [index, module name, module type, similarity, scale, atol,
        atol(N scale), single op error(N scale)]
    """
    if devices is not None:
        assert type(devices) == torch.device or (
            type(devices) == tuple and len(devices) == 2
        ), "devices arg should be torch.device or a tuple of 2 torch.device!"
        print(
            "\nmodel1 and model2 will be run on {} and {}.".format(
                *_pair(devices)
            )
        )
    devices = _pair(devices)

    ret, threshold_ret = _fs(
        model1,
        model2,
        inputs,
        similarity_func,
        threshold,
        devices,
    )
    # print(ret)

    out_dir = "." if out_dir is None else out_dir
    file_path = os.path.join(out_dir, "similarity.txt")
    single_op_order_path = os.path.join(
        out_dir, "ordered_op_error_similarity.txt"
    )
    fig_path = os.path.join(out_dir, "similarity.html")
    with open(file_path, "w") as f:
        f.write("{:-^{width}}\n".format("-", width=63))
        f.write("Note:\n")
        f.write("* Suffix '(I)' means this layer is Identity in one model\n")
        f.write(
            "* Suffix '(I vs I)' means this layer is Identity in both models\n"  # noqa: E501
        )
        f.write("* Suffix '(i)'(i >= 1) means this op is shared i times\n")
        f.write("{:-^{width}}\n".format("-", width=63))
        f.write(
            tabulate(
                ret,
                headers=[
                    "Index",
                    "Module Name",
                    "Module Type",
                    "Similarity",
                    "qscale",
                    "Acc Error\n(float atol)",
                    "Acc Error\n(N out_qscale)",
                    "Op Error with Same\nInput (N out_qscale)",
                ],
                tablefmt="psql",
                floatfmt=("g", "g", "g", ".7f", ".7f", ".7f", "g", "g"),
                numalign="left",
            )
        )
    ordered_ret = sorted(
        ret,
        key=lambda x: x[-1] if isinstance(x[-1], Real) else float("-inf"),
        reverse=True,
    )
    with open(single_op_order_path, "w") as f:
        f.write("{:-^{width}}\n".format("-", width=63))
        f.write("Note:\n")
        f.write("* Suffix '(I)' means this layer is Identity in one model\n")
        f.write(
            "* Suffix '(I vs I)' means this layer is Identity in both models\n"  # noqa: E501
        )
        f.write("* Suffix '(i)'(i >= 1) means this op is shared i times\n")
        f.write("{:-^{width}}\n".format("-", width=63))
        f.write(
            tabulate(
                ordered_ret,
                headers=[
                    "Index",
                    "Module Name",
                    "Module Type",
                    "Similarity",
                    "qscale",
                    "Acc Error\n(float atol)",
                    "Acc Error\n(N out_qscale)",
                    "Op Error with Same\nInput (N out_qscale)",
                ],
                tablefmt="psql",
                floatfmt=("g", "g", "g", ".7f", ".7f", ".7f", "g", "g"),
                numalign="left",
            )
        )

    if threshold_ret:
        print(format_msg("\nThis layers similarity maybe unusual...", "red"))
        print(
            tabulate(
                threshold_ret,
                headers=[
                    "Index",
                    "Module Name",
                    "Module Type",
                    "Similarity",
                    "qscale",
                    "Acc Error\n(float atol)",
                    "Acc Error\n(N out_qscale)",
                    "Op Error with Same\nInput (N out_qscale)",
                ],
                tablefmt="psql",
                floatfmt=("g", "g", "g", ".7f", ".7f", ".7f", "g", "g"),
                numalign="left",
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[ret[i][0] for i in range(len(ret))],
            y=[ret[i][3] for i in range(len(ret))],
            mode="lines+markers",
            text=[
                "{}: {}".format(ret[i][1], str(ret[i][2]))
                for i in range(len(ret))
            ],
        )
    )
    fig.update_layout(
        xaxis_title="Module Index",
        yaxis_title="Similarity",
        yaxis_range=[0, 1.1],
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(255,255,255,0.75)",
            "font": {"color": "black"},
        },
    )
    fig.write_html(fig_path, auto_open=False)

    return ret
