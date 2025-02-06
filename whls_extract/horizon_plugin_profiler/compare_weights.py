import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
import torch.quantization._numeric_suite as ns
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from ._similarity import _compute_similarity, _func_map, _get_max_diff


@typechecked
def compare_weights(
    float_model: torch.nn.Module,
    qat_quantized_model: torch.nn.Module,
    similarity_func: Union[str, Callable] = "Cosine",
    with_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[List]]:
    """Compare weights of float/qat/quantized models.

    This function compares weights of each layer based on
    torch.quantization._numeric_suite.compare_weights. The weight similarity
    and atol will be print on the screen and save in "weight_comparison.txt".
    If you want to see histogram of weights, set with_tensorboard=True.

    Args:
        float_model: float model
        qat_quantized_model: qat or quantized model
        similarity_func: similarity computation function. Support "Cosine",
            "MSE", "L1", "KL", "SQNR" or any user-defined Callable object. If
            it is a user-defined object, it should return a scalar or tensor
            with only one number. Otherwise the result shown may be unexpected.
            Default: "Cosine"
        with_tensorboard: whether to use tensorboard. Default: False
        tensorboard_dir: tensorboard log file path. Default: None
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None

    Returns:
        A weight comparison dict with schema:
            * KEY (str): module name (Eg. layer1.0.conv.weight)
            * VALUE (dict): a dict of the corresponding weights in two models:
                "float": weight value in float model
                "quantized": weight value in qat/quantized model

        A list of list. Each list is each layer weight similarity in format
        [module name, similarity, atol(N scale)]
    """
    assert (
        callable(similarity_func) or similarity_func in _func_map.keys()
    ), "Unsupport similarity computation function {}!".format(similarity_func)

    func = (
        _func_map[similarity_func]
        if type(similarity_func) == str
        else similarity_func
    )
    wt_dict = ns.compare_weights(
        float_model.state_dict(), qat_quantized_model.state_dict()
    )
    similarity = []
    for k, v in wt_dict.items():
        similarity.append(
            [
                k,
                _compute_similarity(v["float"], v["quantized"], func)[0],
                _get_max_diff(v["float"], v["quantized"])[0][0],
            ]
        )

    print(
        tabulate(
            similarity,
            headers=("Weight Name", "Similarity", "Atol"),
            tablefmt="psql",
            floatfmt=".7f",
            numalign="left",
        )
    )

    out_dir = "." if out_dir is None else out_dir
    file_path = os.path.join(out_dir, "weight_comparison.txt")
    with open(file_path, "w") as f:
        f.write(
            tabulate(
                similarity,
                headers=("Weight Name", "Similarity", "Atol"),
                tablefmt="psql",
                floatfmt=".7f",
                numalign="left",
            )
        )

    if with_tensorboard:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        for k, v in wt_dict.items():
            writer.add_histogram(k, v["float"].flatten(), 0)
            writer.add_histogram(k, v["quantized"].flatten(), 1)
        writer.close()

    return wt_dict, similarity
