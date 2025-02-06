import copy
import logging
import warnings
from collections import OrderedDict
from numbers import Real
from typing import Mapping, Sequence

from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    apply_to_collection,
    attach_qualified_name,
    has_submodule,
    is_leaf_module,
    register_hook_on_leaf,
    remove_qualified_name,
    swap_ff_with_horizonff,
)

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from horizon_plugin_pytorch.nn import Identity
from horizon_plugin_pytorch.nn.quantized.functional import dequantize, quantize
from horizon_plugin_pytorch.qtensor import QTensor

__all__ = [
    "featuremap_similarity",
]

logger = logging.getLogger(__name__)


def _compute_similarity(inputs1, inputs2, func):
    result = []
    inputs1 = _as_tuple(inputs1)
    inputs2 = _as_tuple(inputs2)
    for input1, input2 in zip(inputs1, inputs2):
        if input1 is None or input2 is None:
            result.append(None)
            continue
        if not isinstance(
            input1, (Tensor, torch.nn.Parameter)
        ) or not isinstance(input2, (Tensor, torch.nn.Parameter)):
            logger.warning(
                f"unexpected dtype {type(input1)} and {type(input2)}.",
                extra={"call_times_context": ("message")},
            )
            result.append(None)
            continue

        # shape maybe [3] and [3,1,1,1]
        input1 = input1.dequantize() if isinstance(input1, QTensor) else input1
        input2 = input2.dequantize() if isinstance(input2, QTensor) else input2
        input2 = input2.to(input1.device)
        if input1.numel() != input2.numel() or (
            input1.numel() == 0 and input2.numel() == 0
        ):
            result.append(None)
        else:
            if not input1.is_floating_point():
                input1 = input1.float()
            if not input2.is_floating_point():
                input2 = input2.float()

            if isinstance(func, torch.nn.KLDivLoss):
                # model1 is considered as target result
                # use fp64 to improve computation precision
                dim = 1 if input1.ndim > 2 else -1
                ret = func(
                    F.log_softmax(input2.to(torch.float64).cpu(), dim=dim),
                    F.softmax(input1.to(torch.float64).cpu(), dim=dim),
                )
            elif isinstance(func, torch.nn.CosineSimilarity):
                ret = func(
                    input1.flatten().cpu().float(),
                    input2.flatten().cpu().float(),
                )
            else:
                ret = func(input1.cpu().float(), input2.cpu().float())
            result.append(ret)
    return result


def _find_leaf_modules(module, leafset, mod2name):
    """Find the modules to insert the hook and process different modules.

    For example:
        float module: softmax = nn.Softmax()
        qat/quantized module:
            softmax = SegmentLUTSoftmax(
                sub, exp, sum, reciprocal, mul
            )
    we only insert hook in 'softmax' layer, not submodules in qat softmax
    Note: if computing similarity of qat and quantized softmax modules, which
      are matched, hooks will be inserted normally in submodules.
    """
    for _, m in module.named_children():
        if is_leaf_module(m):
            leafset.add(mod2name[m])
        else:
            _find_leaf_modules(m, leafset, mod2name)
    return leafset


def compute_atol(x, y):
    return torch.abs(x - y).max().item()


def compute_rtol(x, y):
    return torch.abs(torch.nan_to_num((x - y) / x, 0)).max().item()


_func_map = OrderedDict(
    (
        ("Cosine", torch.nn.CosineSimilarity(dim=0)),
        ("MSE", torch.nn.MSELoss()),
        ("L1", torch.nn.L1Loss()),
        ("KL", torch.nn.KLDivLoss()),
        (
            "SQNR",
            lambda x, y: (
                20 * torch.log10(torch.norm(x) / torch.norm(x - y))
            ).item(),
        ),
        ("Atol", compute_atol),
        ("Rtol", compute_rtol),
    )
)


def _generate_same_input(input, target_input):
    """Generate data with same dtype and device of target_input."""
    elem_type = type(input)

    if isinstance(target_input, QTensor):
        if input.numel() != target_input.numel():
            return target_input
        device = target_input.device
        if isinstance(input, QTensor):
            return QTensor(
                input.int_repr().to(device)
                if target_input.is_quantized
                else input.dequantize().to(device),
                input.q_scale().to(device),
                input.dtype,
                input.q_per_channel_axis(),
            )
        else:
            assert isinstance(input, Tensor)
            scale = target_input.q_scale()
            zero_point = target_input.q_zero_point()
            ch_axis = target_input.q_per_channel_axis()
            dtype = target_input.dtype
            if scale is None:
                # last conv output
                return QTensor(input.to(device), scale, dtype, ch_axis)

            qvalue = quantize(
                input.to(device),
                scale,
                zero_point,
                ch_axis,
                dtype,
            )
            return QTensor(
                qvalue
                if target_input.is_quantized
                else dequantize(qvalue, scale, zero_point, ch_axis),
                scale,
                dtype,
                ch_axis,
            )

    if isinstance(target_input, Tensor):
        if input.numel() != target_input.numel():
            return target_input
        if isinstance(input, QTensor):
            return input.dequantize().to(target_input.device)
        else:
            assert isinstance(input, Tensor)
            return input.to(target_input.device)

    # Recursively apply to collection items
    if isinstance(target_input, Mapping):
        assert type(input) == type(target_input)
        assert len(input) == len(target_input)
        return elem_type(
            {
                k: _generate_same_input(input[k], target_input[k])
                for k in target_input.keys()
            }
        )

    if isinstance(target_input, Sequence) and not isinstance(
        target_input, str
    ):
        assert type(input) == type(target_input)
        assert len(input) == len(target_input)
        return elem_type(
            [_generate_same_input(i, j) for i, j in zip(input, target_input)]
        )

    return input


def _copy_data(input, inplace):
    """Copy data to avoid inplace operations affect in similarity computation."""  # noqa: E501

    return apply_to_collection(
        input, (Tensor, QTensor), lambda x: copy.deepcopy(x) if inplace else x
    )


def _get_max_diff(inputs1, inputs2):
    """Return the max diff of two result. Maybe max scale diff or float atol.

    If two QTensors and both have scales,
        return (max_scale diff, max_scale diff * scale(atol), scale)
    else if at least one QTensor scale is None or both Tensors,
        return (float atol, float atol, None)
    """
    ret = []
    eps = torch.finfo(torch.float32).eps
    inputs1 = _as_tuple(inputs1)
    inputs2 = _as_tuple(inputs2)
    for input1, input2 in zip(inputs1, inputs2):
        if (
            input1 is None
            or input2 is None
            or input1.numel() != input2.numel()
        ):
            ret.append([None, None, None])
            continue
        if isinstance(input1, QTensor) and isinstance(input2, QTensor):
            v1 = input1.dequantize()
            v2 = input2.dequantize()
            v1.masked_fill_(v1 == 0, eps)
            if input1.q_scale() is None or input2.q_scale() is None:
                diff = torch.abs(v1 - v2.to(v1.device))
                ret.append([diff.max(), diff.max(), None])
                continue
            q1 = input1.int_repr().to(torch.int32)
            q2 = input2.int_repr().to(torch.int32)
            diff = torch.abs(q1 - q2.to(q1.device))
            ret.append(
                [
                    diff.max(),
                    diff.max() * input1.q_scale().max(),
                    input1.q_scale().max(),
                ]
            )
        elif not isinstance(input1, QTensor) and not isinstance(
            input2, QTensor
        ):
            diff = torch.abs(input1 - input2.to(input1.device))
            ret.append([diff.max(), diff.max(), None])
        else:
            v1 = input1
            v2 = input2
            if isinstance(input1, QTensor):
                v1 = input1.dequantize()
                s = input1.q_scale()
                ch_axis = input1.q_per_channel_axis()
            else:
                v2 = input2.dequantize()
                s = input2.q_scale()
                ch_axis = input2.q_per_channel_axis()
            diff = torch.abs(v1 - v2.to(v1.device))
            if s is None:
                ret.append([diff.max(), diff.max(), None])
            else:
                if s.numel() == 1:
                    atol = (diff / s.to(v1.device)).max()
                    ret.append([atol, diff.max(), s])
                else:
                    s_shape = [1] * s.numel()
                    s_shape[ch_axis] = -1
                    atol = (diff / s.reshape(s_shape).to(v1.device)).max()
                    ret.append([atol, diff.max(), None])
    return ret


def featuremap_similarity(
    model1,
    model2,
    inputs,
    similarity_func="Cosine",
    threshold=None,
    devices=(None, None),
):
    """Compute the similarity of feature maps.

    The input models can be float/fused/calibration/qat/quantized model.

    Args:
        model1(Module): can be float/fused/calibration/qat/quantized model
        model2(Module): can be float/fused/calibration/qat/quantized model
        inputs(Any): the input data feed to model
        similarity_func(str, Callable): similarity computation function.
            Support "Cosine", "MSE", "L1", "KL", "SQNR" or any user-defined
            Callable object. If it is a user-defined object, it should return a
            scalar or tensor with only one number. Otherwise the result shown
            may be unexpected. Default: "Cosine"
        threshold(float, None): if similarity value exceeds or less than this
            threshold, the featuremap info will be shown on the screen. If
            threshold is none, it will be set to different values according to
            different similarity functions. Default: None
        devices: run model on which devices (cpu, gpu). If is a tuple of 2
            torch.device, the two models will be moved on specified devices
            seperatedly. It may be used to compare the CPU and GPU results
            difference.

    Returns:
        output(List(List), List(List)): Two list of each layer info. One is
        each layer info list, another is unusual layer info filtered by
        threshold comparison.
        Each info list contains:
            "Index": (int) the module index
            "Module Name": (str) prefix name of the module
            "Module Type": (type) module type
            "Similarity": (float) similarity value of the module outputs
            "Acc Error": (int) max diff (N out_scale)
            "Atol": (float) max float diff (Acc Error * Scale)
            "Scale": (float) scale of the output
            "Op Error with Same Input": (int) max diff with the same inputs
                (N out_scale)
    """
    assert (
        callable(similarity_func) or similarity_func in _func_map.keys()
    ), "Unsupport similarity computation function {}!".format(similarity_func)
    func = (
        _func_map[similarity_func]
        if type(similarity_func) == str
        else similarity_func
    )

    if threshold is None:
        threshold = 0.0 if similarity_func in ("Cosine", "SQNR") else 1.0

    # The closer the value of cosine similarity_func result gets to 1.0, the
    # two layers results are more similar. It reverses in other similarity func
    compare_func = (
        lambda x: x <= threshold
        if isinstance(x, Real)
        else False
        if similarity_func in ("Cosine", "SQNR")
        else x >= threshold
        if isinstance(x, Real)
        else False
    )

    device1, device2 = devices

    # 1. get leaf modules
    model1_mod2name = {}
    model2_mod2name = {}
    for name, mod in model1.named_modules():
        model1_mod2name[mod] = name
    for name, mod in model2.named_modules():
        model2_mod2name[mod] = name

    model1_leafs = _find_leaf_modules(model1, set(), model1_mod2name)
    model2_leafs = _find_leaf_modules(model2, set(), model2_mod2name)
    leafs = model1_leafs | model2_leafs
    # Find modules in two models
    # For example:
    #     float module: softmax = nn.Softmax()
    #     qat/quantized module:
    #         softmax = SegmentLUTSoftmax(
    #             sub, exp, sum, reciprocal, mul
    #         )
    # we only insert hook in 'softmax' layer, not submodules in qat softmax
    # Using intersection here will skip this softmax layer because it is not a
    # leaf module in qat model.
    # Note: if computing similarity of qat and quantized softmax modules, which
    #   are matched, hooks will be inserted normally in submodules.
    discard_leafs = set()
    for leaf in leafs:
        if not has_submodule(model1, leaf) or not has_submodule(model2, leaf):
            discard_leafs.add(leaf)
    leafs = tuple(leafs - discard_leafs)

    # 2. insert hook in model1 and run
    origin_model1 = model1
    model1 = copy.deepcopy(model1).to(device1)
    swap_ff_with_horizonff(model1)
    model1.eval()
    attach_qualified_name(model1, True)

    # record the output of model1
    module1_fmap = OrderedDict()
    # record the input as target input to generate data with same dtype and
    # device to compute single op error
    target_input = OrderedDict()

    # record target_input here to avoid inplace change
    def _pre_hook1(module, input):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )
        inplace = (
            hasattr(module, "inplace")
            and type(module.inplace) == bool
            and module.inplace
        )
        target_input[module_name] = _copy_data(input, inplace)  # noqa F821

    def _hook1(module, input, output):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )
        inplace = (
            hasattr(module, "inplace")
            and type(module.inplace) == bool
            and module.inplace
        )
        module1_fmap[module_name] = (  # noqa F821
            _copy_data(output, inplace),
            type(module),
        )
        module._shared_times += 1

    # explicit pass empty registered_ids to avoid a shared list of ids
    register_hook_on_leaf(
        model1,
        _hook1,
        _pre_hook1,
        check_leaf_module=lambda x: x._qualified_name in leafs,
    )
    inputs1 = _as_tuple(
        apply_to_collection(inputs, Tensor, lambda x: x.to(device1))
    )
    with torch.no_grad():
        model1(*inputs1)
    del model1

    # 3. insert hook in model2 and run. Compute similarity and delete
    # module1_fmap to save memory
    model2 = copy.deepcopy(model2).to(device2)
    swap_ff_with_horizonff(model2)
    model2.eval()
    attach_qualified_name(model2, True)

    # register hook for quantized module2
    module2_fmap = OrderedDict()
    # record model2 input to generate same input for model3
    input_fmap = OrderedDict()

    # record similarity and atol
    simi_fmap = OrderedDict()
    threshold_simi_fmap = OrderedDict()

    # record target_input here to avoid inplace change
    def _pre_hook2(module, input):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )
        inplace = (
            hasattr(module, "inplace")
            and type(module.inplace) == bool
            and module.inplace
        )
        input_fmap[module_name] = _copy_data(input, inplace)  # noqa F821

    def _hook2(module, input, output):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )
        inplace = (
            hasattr(module, "inplace")
            and type(module.inplace) == bool
            and module.inplace
        )
        module2_fmap[module_name] = [  # noqa F821
            _copy_data(output, inplace),
            type(module),
        ]

        # compute model1 and model2 result similarity
        k = module_name
        if k not in module1_fmap:  # noqa F821
            warnings.warn("key {} not found in module1_fmap".format(k))
        else:
            ret = _compute_similarity(
                module1_fmap[k][0], module2_fmap[k][0], func  # noqa F821
            )
            # diff_max, diff_atol, scale
            atols = _get_max_diff(
                module1_fmap[k][0], module2_fmap[k][0]  # noqa F821
            )
            m1 = module1_fmap[k][1]  # noqa F821
            m2 = module2_fmap[k][1]  # noqa F821
            is_m1_identity = m1 == torch.nn.Identity or m1 == Identity
            is_m2_identity = m2 == torch.nn.Identity or m2 == Identity
            module_type = m1 if not is_m1_identity else m2
            if is_m1_identity and is_m2_identity:
                suffix = "(I vs I)"
            elif is_m1_identity or is_m2_identity:
                suffix = "(I)"
            else:
                suffix = ""
            module2_fmap[k][1] = suffix  # noqa F821
            assert len(ret) == len(atols)
            for i in range(len(ret)):
                out_suffix = "" if i == 0 else f"[output-{i}]"
                diff_max, diff_atol, scale = atols[i]
                if compare_func(ret[i]):
                    threshold_simi_fmap[k + out_suffix + suffix] = [
                        module_type,
                        ret[i],
                        scale,
                        diff_atol,
                        diff_max,
                    ]
                simi_fmap[k + out_suffix + suffix] = [
                    module_type,
                    ret[i],
                    scale,
                    diff_atol,
                    diff_max,
                ]
            module1_fmap[module_name] = None  # noqa F821
        module._shared_times += 1

    register_hook_on_leaf(
        model2,
        _hook2,
        _pre_hook2,
        check_leaf_module=lambda x: x._qualified_name in leafs,
    )
    inputs2 = _as_tuple(
        apply_to_collection(inputs, Tensor, lambda x: x.to(device2))
    )
    with torch.no_grad():
        model2(*inputs2)
    del model2, module1_fmap

    # 4. run model3 with same input with model2 to get single op diff
    model3 = copy.deepcopy(origin_model1).to(device1)
    swap_ff_with_horizonff(model3)
    model3.eval()
    remove_qualified_name(model3)
    attach_qualified_name(model3, True)

    def _pre_hook3(module, input):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )
        if (
            module_name in target_input  # noqa F821
            and module_name in input_fmap  # noqa F821
        ):
            input = _generate_same_input(
                input_fmap[module_name], target_input[module_name]  # noqa F821
            )
            input_fmap[module_name] = None  # noqa F821
            target_input[module_name] = None  # noqa F821
        else:
            # two different models may report this warning
            warnings.warn(
                "Please make sure that two input models are different "
                + "stages of the same origin model."
            )
        return input

    def _hook3(module, input, output):
        module_name = module._qualified_name + (
            f"({module._shared_times})" if module._shared_times > 0 else ""
        )

        # compute model2 and model3 single op error
        k = module_name
        if k not in module2_fmap:  # noqa F821
            warnings.warn("key {} not found in module2_fmap".format(k))
        else:
            same_op_atols = _get_max_diff(
                module2_fmap[k][0], output  # noqa F821
            )
            for i in range(len(same_op_atols)):
                out_suffix = "" if i == 0 else f"[output-{i}]"
                same_in_max_qscale_diff = same_op_atols[i][0]
                key = k + out_suffix + module2_fmap[k][1]  # noqa F821
                simi_fmap[key].append(same_in_max_qscale_diff)
                if key in threshold_simi_fmap:
                    threshold_simi_fmap[key].append(same_in_max_qscale_diff)
        module2_fmap[module_name] = None  # noqa F821
        module._shared_times += 1

    register_hook_on_leaf(
        model3,
        _hook3,
        _pre_hook3,
        check_leaf_module=lambda x: x._qualified_name in leafs,
    )
    inputs3 = _as_tuple(
        apply_to_collection(inputs, Tensor, lambda x: x.to(device1))
    )
    with torch.no_grad():
        model3(*inputs3)
    del model3, module2_fmap, input_fmap, target_input

    # 5. regroup results into list
    result = []
    threshold_result = []

    def _regroup(fmap, flist):
        index = 0
        for k, v in fmap.items():
            v = [vv.item() if isinstance(vv, Tensor) else vv for vv in v]
            flist.append([index, k, *v])
            index += 1
        return flist

    result = _regroup(simi_fmap, result)
    threshold_result = _regroup(threshold_simi_fmap, threshold_result)
    return result, threshold_result
