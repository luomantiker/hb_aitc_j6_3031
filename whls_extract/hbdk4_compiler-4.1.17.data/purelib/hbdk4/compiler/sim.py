import os
import sys
from typing import Sequence, Optional
from hbdk4.compiler.utils.process import run_program_redirect_realtime


def _get_hw_str(inputs_name: str, inputs: Sequence[Sequence[int]]) -> str:
    for i, dims in enumerate(inputs):
        if len(dims) != 2 and len(dims) != 0:
            raise RuntimeError(
                f"{inputs_name}[{i}] size should be 0 or 2(HxW) but {len(dims)} were given"
            )
    res = ",".join(["x".join([str(d) for d in dims]) for dims in inputs])
    return res


def hbm_sim(
    model: str,
    model_name: str,
    input_binarys: Sequence[str],
    output_dir: str,
    enable_perf: bool = False,
    yuv_sizes: Sequence[Sequence[int]] = None,
    yuv_strides: Sequence[Optional[int]] = None,
    yuv_roi_coords: Sequence[Sequence[int]] = None,
    yuv_roi_sizes: Sequence[Sequence[int]] = None,
    memory: int = None,
):
    """run simulator

    Args:
        * model (str): hbm file name
        * model_name (str): model name
        * input_binarys (Sequence[str]): input binary files (feature or YUV)
        * output_dir (str): output directory
        * enable_perf (bool, optional): enable dump perf info. Defaults to False.
        * yuv_sizes (Sequence[Sequence[int]], optional): size (HxW) for YUV. [] for feature.
        * yuv_strides (Sequence[Optional[int]], optional): stride for YUV. None for feature.
        * yuv_roi_coords (Sequence[Sequence[int]], optional): ROI coordinate (HxW) for YUV. [] for feature.
        * yuv_roi_sizes (Sequence[Sequence[int]], optional): ROI size (HxW) for YUV. [] for feature.
        * memory(int, optional): BPU memory pre-allocated for simulator. Unit:MB. Default 1024MB.
    """
    tool = os.path.join(os.path.dirname(__file__), "_mlir_libs", "hbdk-sim")
    cmd = [tool]
    cmd += ["-f", model]
    cmd += ["-n", model_name]
    if len(input_binarys) > 0:
        cmd += ["-i", ",".join(input_binarys)]
    cmd += ["-o", output_dir]
    if enable_perf:
        cmd += ["--perf"]
    if memory:
        cmd += ["--memory", str(memory)]
    if yuv_sizes:
        cmd += ["--yuv-size", _get_hw_str("yuv_sizes", yuv_sizes)]
    if yuv_strides:
        cmd += ["--yuv-stride", ",".join([str(d) if d else "" for d in yuv_strides])]
    if yuv_roi_coords:
        cmd += ["--yuv-roi-coord", _get_hw_str("yuv_roi_coords", yuv_roi_coords)]
    if yuv_roi_sizes:
        cmd += ["--yuv-roi-size", _get_hw_str("yuv_roi_sizes", yuv_roi_sizes)]

    p = run_program_redirect_realtime(cmd, stdout=sys.stdout, stderr=sys.stderr)
    ret = p.returncode
    if ret != 0:
        raise RuntimeError("HBDK sim FAIL, please check with HBDK Group")
    else:
        print("HBDK sim SUCCESS")

    return ret
