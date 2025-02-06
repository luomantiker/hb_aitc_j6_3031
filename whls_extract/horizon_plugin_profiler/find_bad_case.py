import copy
import logging
import os
import sys
from collections import OrderedDict
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    _get_attr,
    _set_attr,
    apply_to_collection,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
import torch.nn.functional as F  # noqa: N812
from tabulate import tabulate
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.qtensor import QTensor

logger = logging.getLogger(__name__)


def _compute_diff(data1, data2, func):
    assert type(data1) == type(data2)
    if isinstance(data1, QTensor):
        data1 = data1.dequantize()
        data2 = data2.dequantize()
    assert isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor)
    assert data1.numel() == data2.numel()
    if isinstance(func, torch.nn.KLDivLoss):
        # model1 is considered as target result
        # use fp64 to improve computation precision
        dim = 1 if data1.ndim > 2 else -1
        ret = func(
            F.log_softmax(data2.to(torch.float64).cpu(), dim=dim),
            F.softmax(data1.to(torch.float64).cpu(), dim=dim),
        )
    elif isinstance(func, torch.nn.CosineSimilarity):
        ret = func(data1.flatten().float(), data2.flatten().float()).cpu()
    else:
        ret = func(data1.float().cpu(), data2.float().cpu())
    return ret.item()


def _regroup(origin_ret, index_ret, prefix="", ret=[]):  # noqa: B006
    if isinstance(origin_ret, torch.Tensor):
        ret.append(
            [
                prefix,
            ]
            + list(index_ret.values())
        )

    if isinstance(origin_ret, Sequence) and not isinstance(origin_ret, str):
        for i, ori_r in enumerate(origin_ret):
            _regroup(
                ori_r,
                index_ret[i],
                f"{i}" if prefix == "" else f"{prefix}-{i}",
                ret,
            )

    if isinstance(origin_ret, Mapping):
        for k, v in origin_ret.items():
            _regroup(v, index_ret[k], k, ret)


@typechecked
def find_bad_case(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    data_generator: Iterable,
    num_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    custom_metric_func: Optional[Callable] = None,
    custom_metric_order_seq: Optional[str] = None,
    cached_attrs: Optional[Tuple[str, ...]] = None,
    out_dir: Optional[str] = None,
):
    """
    Run all data to find the baddest case(the largest diff out between models).

    The input models can be float/fused/calibration/qat/quantized model. This
    function computes the Cosine/MSE/L1/KL/SQNR difference of the two models
    outputs to find the baddest input case in the dataloader. This bad case can
    be inputs of other debug tools, like featuremap_similarity, etc. The
    badcase index of the dataloader will be printed and saved into
    "badcase.txt" or `out_dir`.

    Note: It is recommended that only deploy model is given, which has no
    preprocess or postprocess. And the model outputs should only include
    tensors, without str or int/float type.

    Args:
        model1: can be float/fused/calibration/qat/quantized model
        model2: can be float/fused/calibration/qat/quantized model
        data_generator: input dataloader or a custom iterable object
        num_steps: Num of steps to find bad case.
        device: run model on which device. Default: None
        custom_metric_func: user-defined callable metric func to compute model
            output difference
        custom_metric_order_seq: user-defined metric order sequence from bad to
            good. For example, cosine metric order is ascending, the smallest
            is the baddest. While L1 metric order is descending, the biggest is
            the baddest. Only support "ascending"/"descending".
        cached_attrs: cached attrs to use as input. Usually used in sequence
            model. Default None.
        out_dir: path to save the result file

    Returns:
        A list of three elements. 1. badcase index of each input in different
        metrics; 2. worst index of each metric; 3. badcase index input data
        Example:

        Origin model output:
            [tensor(a), tensor(b), tensor(c), tensor(d)]
        find_bad_case output:
            1. badcase index of each input in different metrics, each small
                list is [output-index, ] + badcase index of
                [cosine, mse, l1, kl, sqnr]

                [
                    ['0', 1, 0, 2, 3, 1],
                    ['1', 2, 2, 1, 3, 4],
                    ['2', 1, 0, 2, 3, 1],
                    ['3', 2, 2, 1, 3, 4],
                ]

            2. a dict of worst index of each metric
                {
                    'Cosine': 1,
                    'MSE': 0,
                    'L1': 1,
                    'KL': 2,
                    'SQNR': 4,
                }

            3. badcase index input data
                {
                    0: (index_0_data, (base_cached_attr, analy_cached_attr)),
                    1: (index_1_data, (base_cached_attr, analy_cached_attr)),
                    2: (index_2_data, (base_cached_attr, analy_cached_attr)),
                    3: (index_3_data, (base_cached_attr, analy_cached_attr)),
                    4: (index_4_data, (base_cached_attr, analy_cached_attr)),
                }
    """
    model1 = model1.eval()
    model2 = model2.eval()
    if device is not None:
        model1 = model1.to(device)
        model2 = model2.to(device)

    funcmap = {
        "Cosine": torch.nn.CosineSimilarity(dim=0),
        "MSE": torch.nn.MSELoss(),
        "L1": torch.nn.L1Loss(),
        "KL": torch.nn.KLDivLoss(),
        "SQNR": lambda x, y: (
            20 * torch.log10(torch.norm(x) / torch.norm(x - y))
        ),
    }
    metric_order_seq = {
        "ascending": ["Cosine", "SQNR"],
        "descending": ["MSE", "L1", "KL"],
    }
    if custom_metric_func is not None:
        funcmap.update({"custom": custom_metric_func})
        assert custom_metric_order_seq in (
            "ascending",
            "descending",
        ), "custom_metric_order_seq is necessary to find the worst case."
        metric_order_seq[custom_metric_order_seq].append("custom")

    # dict of index -> (data, cached_attrs)
    bad_input_dict = {}

    # custom iterable object may not have len func
    # total = len(list(data_generator))
    try:
        total = len(data_generator)
    except Exception:
        total = None
    if num_steps is not None:
        assert num_steps >= 0, "num_steps must be positive."
    else:
        num_steps = sys.maxsize

    result = []
    worst_dict = {}
    spec = None
    for i, data in enumerate(data_generator):
        if i > num_steps:
            break
        data = apply_to_collection(data, torch.Tensor, lambda x: x.to(device))
        with torch.no_grad():
            data = _as_tuple(data)
            # use model1 cached attrs
            if cached_attrs is not None:
                origin_attrs = {
                    attr: copy.deepcopy(_get_attr(model1, attr))
                    for attr in cached_attrs
                }
            ret1 = model1(*copy.deepcopy(data))
            # set model2 attrs to keep same with model1
            if cached_attrs is not None:
                for k, v in origin_attrs.items():
                    _set_attr(model2, k, v)
            ret2 = model2(*copy.deepcopy(data))

        assert type(ret1) == type(ret2)
        if isinstance(ret1, OrderedDict):
            ret1 = dict(ret1)
            ret2 = dict(ret2)

        flat1, spec = tree_flatten(ret1)
        flat2, _ = tree_flatten(ret2)
        assert len(flat1) == len(flat2)

        if cached_attrs is not None:
            attrs = (origin_attrs, origin_attrs)
        else:
            attrs = ({}, {})

        current_indexes = set()
        if i == 0:
            # initialize
            for f1, f2 in zip(flat1, flat2):
                if isinstance(f1, torch.Tensor):
                    diffmap = {}
                    for k, v in funcmap.items():
                        diffmap[k] = [_compute_diff(f1, f2, v), 0]
                    result.append(diffmap)
                    # initialize worst dict
                    worst_dict = copy.deepcopy(diffmap)
                elif f1 is None:
                    # insert empty dict to avoid process error
                    result.append({})
                else:
                    result.append(f1)
            current_indexes.add(0)
        else:
            # update result diffmap
            for j, r1 in enumerate(flat1):
                if isinstance(r1, torch.Tensor):
                    for k, v in funcmap.items():
                        simi = _compute_diff(r1, flat2[j], v)
                        if k in metric_order_seq["ascending"]:
                            if simi < result[j][k][0]:
                                result[j][k] = [simi, i]
                            if worst_dict[k][0] > result[j][k][0]:
                                worst_dict[k] = copy.deepcopy(result[j][k])
                        elif k in metric_order_seq["descending"]:
                            if simi > result[j][k][0]:
                                result[j][k] = [simi, i]
                            if worst_dict[k][0] < result[j][k][0]:
                                worst_dict[k] = copy.deepcopy(result[j][k])
                        else:
                            raise RuntimeError(f"Unknown metric {k}")
                        current_indexes.add(result[j][k][1])

        # save new bad case into bad input dict
        old_indexes = set(bad_input_dict.keys())
        discard_indexes = old_indexes - current_indexes
        new_indexes = current_indexes - old_indexes
        for k in discard_indexes:
            bad_input_dict.pop(k)
        for k in new_indexes:
            bad_input_dict[k] = (copy.deepcopy(data), attrs)

        if i % 50 == 0:
            step = f"{i}/{total}" if total is not None else f"{i}"
            logger.info(step + " cases done...")

    metric_result = [
        {k: v[0] for k, v in r.items()} if isinstance(r, dict) else r
        for r in result
    ]
    index_result = [
        {k: v[-1] for k, v in r.items()} if isinstance(r, dict) else r
        for r in result
    ]
    worst_index_dict = {k: v[1] for k, v in worst_dict.items()}

    try:
        result = tree_unflatten(result, spec)
        metric_result = tree_unflatten(metric_result, spec)
        index_result = tree_unflatten(index_result, spec)
    except ValueError as e:
        logger.info(result)
        raise e

    # print index table
    regroup_index_result = []
    regroup_result = []
    regroup_metric_result = []
    _regroup(ret1, index_result, "", regroup_index_result)
    _regroup(ret1, result, "", regroup_result)
    _regroup(ret1, metric_result, "", regroup_metric_result)
    print("The bad case input index of each output:\n")
    print(
        tabulate(
            regroup_index_result,
            headers=[
                "Name/Index",
            ]
            + list(funcmap.keys()),
            numalign="left",
        )
    )
    print("\n\nThe metric results of each badcase:\n")
    print(
        tabulate(
            regroup_metric_result,
            headers=[
                "Name/Index",
            ]
            + list(funcmap.keys()),
            numalign="left",
        )
    )
    print("\n\nThe bad case input index of the worst output:\n")
    print(
        tabulate(
            worst_index_dict.items(),
            headers=["metric", "dataloader index"],
            numalign="left",
        )
    )
    out_dir = "." if out_dir is None else out_dir
    file_path = os.path.join(out_dir, "badcase.txt")
    with open(file_path, "w") as f:
        f.write("The bad case input index of each output:\n")
        f.write(
            tabulate(
                regroup_index_result,
                headers=[
                    "Name/Index",
                    "Cosine",
                    "MSE",
                    "L1",
                    "KL",
                    "SQNR",
                ],
                numalign="left",
            )
        )
        f.write("\n\nThe metric results of each badcase:\n")
        f.write(
            tabulate(
                regroup_metric_result,
                headers=[
                    "Name/Index",
                    "Cosine",
                    "MSE",
                    "L1",
                    "KL",
                    "SQNR",
                ],
                numalign="left",
            )
        )
        f.write("\n\nThe bad case input index of the worst output:\n")
        f.write(
            tabulate(
                worst_index_dict.items(),
                headers=["metric", "dataloader index"],
                numalign="left",
            )
        )

    badcase_info = (regroup_index_result, worst_index_dict, bad_input_dict)
    path = os.path.join(out_dir, "all_badcase_info.pt")
    torch.save(badcase_info, path)
    return badcase_info
