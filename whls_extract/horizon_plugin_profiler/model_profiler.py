import os
from numbers import Real
from typing import Any, Optional

import horizon_plugin_profiler as hpp
from horizon_plugin_profiler.utils.typeguard import typechecked

import numpy as np
import torch
from jinja2 import Environment, PackageLoader
from torch import Tensor

from .check_qconfig import check_qconfig
from .check_unfused_operations import check_unfused_operations
from .compare_weights import compare_weights
from .featuremap_similarity import featuremap_similarity
from .get_module_called_count import get_module_called_count
from .get_raw_features import get_raw_features
from .profile_featuremap import profile_featuremap


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


@typechecked
def model_profiler(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    example_inputs: Any,
    mode: str,
    out_dir: Optional[str] = None,
    kwargs_dict: Optional[dict] = None,
):
    """Profiler the models using debug tools and show result in one page.

    This function computes
    1) similarity, statistics, weights similarity and shared ops of the given
        models
    2) check unfused ops of the float model and qconfig of the qat model, which
        controlled by `mode`
    The results are shown in one html page named `profiler.html`, which stored
    in default dir or `out_dir`.

    NOTE:
        1) Only support models compared in any two adjacent stages.
            `float vs qat` or `qat vs quantized` is supported, while
            `float vs quantized` or `qat vs float` is unsupported.
        2) Visual model structures in onnx format and featuremap histogram are
            not shown in the html file. You can call `export_to_onnx and
            `profile_featuremap` with `with_tensorboard=True`. Custom args can
            also be passed by `kwargs_dict`.

    Args:
        model1: can be float/calibration/qat model
        model2: can be calibration/qat/quantized model
        example_inputs: model inputs
        mode: specific which two models to be compared. Only three modes shown
            below are supported
            "FvsQ": float vs qat. In this mode, `model2` can be either
                calibration or qat model.
            "QvsQ": qat vs quantized.
        out_dir: path to save `profiler.html` and all other result files. If
            None, results are saved in `horizon_quant_debug` dir in current dir
        kwargs_dict: kwargs of debug tools functions in dict format. E.g.
            kwargs_dict = {
                "featuremap_similarity": {
                    "similarity_func": Cosine,
                },
                "profile_featuremap": {
                    "with_tensorboard": True,
                }
                ...
            }
            Only support 7 keys, which are the names of the 7 debug functions
            that will be invoked in this function. The supported keys are:
                1) featuremap_similarity
                2) get_raw_features
                3) profile_featuremap
                4) get_module_called_count
                5) check_unfused_operations
                6) compare_weights
                7) check_qconfig
            NOTE:
                1) model and example_inputs must not be defined in kwargs
                2) `out_dir` in kwargs will be replaced with `out_dir` in
                    `model_profiler` args
    """
    assert mode in (
        "FvsQ",
        "QvsQ",
    ), f"{mode} is not supported, only support FvsQ/QvsQ"

    if kwargs_dict is None:
        kwargs_dict = {}
    else:
        for k in kwargs_dict.keys():
            assert k in (
                "featuremap_similarity",
                "get_raw_features",
                "profile_featuremap",
                "get_module_called_count",
                "check_unfused_operations",
                "compare_weights",
                "check_qconfig",
            ), f"Unexpected key {k} in kwargs_dict"

    models = (model1, model2)
    float_index = 0 if mode == "FvsQ" else None
    qat_index = 0 if mode == "QvsQ" else 1

    similarity_dict = {}
    statistic_dict = {}
    wt_dict = {}
    module_to_fuse = []
    module_count = {}
    qconfig_dict = {}

    if out_dir is None:
        out_dir = "./horizon_quant_debug"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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

    # feature similarity
    simi_kwargs = kwargs_dict.get("featuremap_similarity", {})
    simi_kwargs.update({"out_dir": out_dir})
    simi = featuremap_similarity(
        models[0],
        models[1],
        example_inputs,
        **simi_kwargs,
    )

    name = [f[1] for f in simi]
    types = [str(f[2]) for f in simi]
    simi_v = _dump(
        [f[3].item() if isinstance(f[3], Tensor) else f[3] for f in simi]
    )
    atol = _dump([f[5] for f in simi])
    acc = _dump([f[6] for f in simi])
    single_op_error = _dump([f[7] for f in simi])
    y1data = [x for x in simi_v if isinstance(x, Real)]
    y2data = [x for x in acc + single_op_error if isinstance(x, Real)]
    ybounds = _get_y_bound(min(min(y1data), 0), max(y1data), 0, max(y2data))
    similarity_dict.update(
        {
            "name": name,
            "type": types,
            "simi": simi_v,
            "atol": atol,
            "acc": acc,
            "single_op_error": single_op_error,
            "ybounds": ybounds,
        }
    )

    # statistic
    raw_kwargs = kwargs_dict.get("get_raw_features", {})
    statistic_kwargs = kwargs_dict.get("profile_featuremap", {})
    statistic_kwargs.update(
        {"out_dir": out_dir, "file_name": "model0_statistic"}
    )
    stat0 = profile_featuremap(
        get_raw_features(models[0], example_inputs, **raw_kwargs),
        **statistic_kwargs,
    )
    statistic_kwargs.update({"file_name": "model1_statistic"})
    stat1 = profile_featuremap(
        get_raw_features(models[1], example_inputs, **raw_kwargs),
        **statistic_kwargs,
    )

    for index, stat in enumerate((stat0, stat1)):
        name = [f"{f[1]}\n{f[3]}" for f in stat]
        fmin = [f[4] for f in stat]
        fmax = [f[5] for f in stat]
        fmean = [f[6] for f in stat]
        # convert python nan to js NaN
        fvar = _dump([f[7] for f in stat])

        min1, max1 = min(fmin), max(fmax)
        min2 = min([x for x in fvar if isinstance(x, Real)])
        max2 = max([x for x in fvar if isinstance(x, Real)])
        ybounds = _get_y_bound(min1, max1, min2, max2)

        statistic_dict.update(
            {
                f"model{index}": {
                    "name": name,
                    "fmin": fmin,
                    "fmax": fmax,
                    "fmean": fmean,
                    "fvar": fvar,
                    "ybounds": ybounds,
                }
            }
        )

    # weight
    weight_kwargs = kwargs_dict.get("compare_weights", {})
    weight_kwargs.update({"out_dir": out_dir})
    _, weight = compare_weights(models[0], models[1], **weight_kwargs)
    name = [f[0] for f in weight]
    simi_v = _dump(
        [f[1].item() if isinstance(f[1], Tensor) else f[1] for f in weight]
    )
    atol = _dump(
        [f[2].item() if isinstance(f[2], Tensor) else f[2] for f in weight]
    )
    y1data = [x for x in simi_v if isinstance(x, Real)]
    y2data = [x for x in atol if isinstance(x, Real)]
    ybounds = _get_y_bound(min(min(y1data), 0), max(y1data), 0, max(y2data))
    wt_dict.update(
        {"name": name, "simi": simi_v, "atol": atol, "ybounds": ybounds}
    )

    # shared_op check
    module_count_kwargs = kwargs_dict.get("get_module_called_count", {})
    module_count_kwargs.update({"print_tabulate": False})
    module0_count = get_module_called_count(
        models[0], example_inputs, **module_count_kwargs
    )
    module1_count = get_module_called_count(
        models[1], example_inputs, **module_count_kwargs
    )
    module_count.update({"model0": module0_count, "model1": module1_count})

    # fused check
    if float_index is not None:
        fuse_kwargs = kwargs_dict.get("check_unfused_operations", {})
        fuse_kwargs.update({"print_tabulate": False})
        module_to_fuse = check_unfused_operations(
            models[float_index], example_inputs, **fuse_kwargs
        )
        for f in module_to_fuse:
            for i, ff in enumerate(f):
                f[i] = list(ff)
                # <class xxx> will be treated as a label in html, so delete <>
                f[i][1] = str(f[i][1])[1:-1]
    else:
        module_to_fuse = "null"

    # qconfig check
    if qat_index is not None:
        qconfig_kwargs = kwargs_dict.get("check_qconfig", {})
        qconfig_kwargs.update({"out_dir": out_dir})
        out_info_map, weight_info_map, unusual_map = check_qconfig(
            models[qat_index], example_inputs, **qconfig_kwargs
        )
        for f in out_info_map:
            f[1] = str(f[1])[1:-1]
            f[2] = str(f[2])
            f[3] = str(f[3])
            f[4] = str(f[4])
        for f in weight_info_map:
            f[1] = str(f[1])[1:-1]
            f[2] = str(f[2])
            f[3] = str(f[3])
        for f in unusual_map:
            f[1] = str(f[1])[1:-1]
        qconfig_dict.update(
            {
                "out": out_info_map,
                "weight": weight_info_map,
                "unusual": unusual_map,
            }
        )

    env = Environment(
        loader=PackageLoader("horizon_plugin_profiler", "profiler_templates")
    )
    template = env.get_template("profiler_template")

    # create a soft link of css and echarts.js
    srcdir = os.path.join(hpp.__path__[0], "profiler_templates")
    css_src_dir = os.path.join(srcdir, "style.css")
    echarts_src_dir = os.path.join(srcdir, "echarts.min.js")
    css_dst_dir = os.path.join(out_dir, "style.css")
    echarts_dst_dir = os.path.join(out_dir, "echarts.js")
    for src, dst in zip(
        (css_src_dir, echarts_src_dir), (css_dst_dir, echarts_dst_dir)
    ):
        if not os.path.exists(dst):
            # if the origin file does not exists while the soft link exists,
            # delete the useless soft link
            if os.path.islink(dst):
                os.remove(dst)
            # check the origin file must exists
            assert os.path.exists(src), f"Can not find file {src}!"
            os.symlink(src, dst)

    out = template.render(
        similarity_dict=similarity_dict,
        statistic_dict=statistic_dict,
        weight_data=wt_dict,
        module_count=module_count,
        unfused=module_to_fuse,
        qconfig=qconfig_dict,
    )

    out_path = os.path.join(out_dir, "profiler.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out)
