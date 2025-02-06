import os

from typing import Union, Any, List

from matplotlib.colors import is_color_like
from hbdk4.compiler.march import MarchBase, March
from hbdk4.compiler.overlay import Module
from hbdk4.compiler._mlir_libs import _hbdk as _hbdk_cext
from hbdk4.compiler.version import FOR_DEV_USE
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.dialects.func import FuncOp
from hbdk4.compiler._mlir_libs._hbdk import (
    _fake_convert,
    _internal_compile,
    _calibrate,
)


def dynamic_quantize_convert_per_block(exported_module, blockSize):
    op_list = []
    for module_op in exported_module.module.body.operations:
        if isinstance(module_op, (mlir.Operation, FuncOp)):
            for block in module_op.opview.body.blocks:
                for op in block:
                    if op.name == "hbir.linear" or op.name == "hbir.matmul":
                        op_list.append(op)

    for op in op_list:
        print("dynamic_quantize per-block op.view", op.opview)
        if op.name == "hbir.linear":
            _hbdk_cext._dynamic_quantize_convert(
                op, [8], [True], [True, True], [-1, -1], [True], [blockSize]
            )
        elif op.name == "hbir.matmul":
            _hbdk_cext._dynamic_quantize_convert(
                op, [8], [True], [True, True], [-1, -2], [True], [blockSize]
            )

    return exported_module


def dynamic_quantize_convert(exported_module):
    op_list = []
    for module_op in exported_module.module.body.operations:
        if isinstance(module_op, (mlir.Operation, FuncOp)):
            for block in module_op.opview.body.blocks:
                for op in block:
                    if op.name == "hbir.linear" or op.name == "hbir.matmul":
                        op_list.append(op)

    for op in op_list:
        print("dynamic_quantize op.view", op.opview)
        if op.name == "hbir.matmul":
            _hbdk_cext._dynamic_quantize_convert(
                op, [8], [True], [True, True], [-2, -1], [False], [-1]
            )
        elif op.name == "hbir.linear":
            _hbdk_cext._dynamic_quantize_convert(
                op, [8], [True], [True, True], [-2, -2], [False], [-1]
            )

    return exported_module


def kv_cache_update(exported_module):
    op_list = []
    for module_op in exported_module.module.body.operations:
        if isinstance(module_op, (mlir.Operation, FuncOp)):
            for block in module_op.opview.body.blocks:
                for op in block:
                    if op.name == "hbir.kv_cache_update":
                        op_list.append(op)

    for op in op_list:
        print("kv_cache_update op.view", op.opview)
        if op.name == "hbir.kv_cache_update":
            _hbdk_cext._kv_cache_update(op, 8, -1)

    return exported_module


def calibrate(m, gran="single", is_collect=True, batch=0):
    new_m = m.clone()
    is_collect_flag = "false"
    if is_collect:
        is_collect_flag = "true"
    args = {"gran": gran, "isCollect": is_collect_flag, "batch": str(batch)}
    _calibrate(new_m.module, m.module.context, args)

    return new_m


def fake_convert(m: Module, march: Union[MarchBase, str], perf_output_dir: str = ""):
    """Convert hbir to backend ir using fake quantization parameters.
    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-b", "nash-e", "nash-m", "nash-p"
        * perf_output_dir (str, required): Specify output directory for performance statistics.
    """

    if not isinstance(march, March):
        march = March.get(march)

    args = {
        "march": march,
        "perf_output_dir": perf_output_dir,
    }

    new_m = m.clone()
    _hbdk_cext._fake_convert(new_m.module, m.module.context, args)
    return new_m


def internal_compile(
    m: Module,
    march: Union[MarchBase, str],
    perf_output_dir: str,
    opt: int = 2,
    jobs: int = 4,
    progress_bar: bool = False,
    advice: float = 0.0,
    balance: int = 100,
):
    """Compile mlir.Module and estimate performance.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-b", "nash-e", "nash-m", "nash-p"
        * perf_output_dir (str, required): Specify output directory for performance statistics.
        * opt (int, optional): Optimization level. Defaults to 2.
        * jobs (int, optional): Number of threads launched during compiler optimization. Defaults to 4.
        * max_time_per_fc (float, optional): Set maximum time constraint (unit:us) for per funccall.
        * debug (bool, optional): Set whether to contain debug info in HBM
        * hbdk3_compatible_mode (bool, optional): Set whether to compile in hbdk3 compatible mode
        * progress_bar(bool, optional): Set whether to show progress bar
        * advice(float, optional): Print advice on ops that take longer more than the specified time (unit:us)
        * balance(int, optional): Specify a integer (recommend 2) to balance cycles and DDR access.
                                The integer should be between 0 (minimal DDR access) and 100 (minimal cycles)

    Raises:
        * ValueError: When \"balance\" is not in range of [0,100]

    """

    if not isinstance(march, March):
        march = March.get(march)

    if not (0 <= balance <= 100):
        raise ValueError(
            "balance should be between 0 (minimal DDR access) and 100 (minimal cycles)"
        )

    if perf_output_dir and not os.path.exists(perf_output_dir):
        os.makedirs(perf_output_dir, exist_ok=True)

    args = {
        "march": march,
        "opt": str(opt),
        "jobs": str(jobs),
        "progressbar": str(int(progress_bar)),
        "advice": str(advice),
        "balance": str(balance),
        "perf_output_dir": perf_output_dir and f"{perf_output_dir}" or "",
    }

    new_m = m.clone()
    _hbdk_cext._internal_compile(new_m.module, m.module.context, args)

    # hack: The json for the original float stage will be generated in CWD, we manually copy it to perf_output_dir
    original_float_json_name = "tmp_1_original_float_perf.json"
    if os.path.exists(original_float_json_name):
        import shutil

        shutil.copy(original_float_json_name, perf_output_dir)

    try:
        # In public mode
        _post_process_public_perf_output(perf_output_dir)
    except FileNotFoundError:
        # In internal or dev mode
        _post_process_perf_output(perf_output_dir)

    return


def estimate(
    jit,
    example_input: Any,
    march: Union[MarchBase, str],
    perf_output_dir: str,
    opt: int = 2,
    jobs: int = 4,
) -> None:
    """Generate perf estimation

    Args:
        * jit (torch.jit.ScriptModule): a ScriptModule created from torch.jit.trace
        * example_input (Any): input format of the ScriptModule, used for analysis
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-b", "nash-e", "nash-m", "nash-p"
        * perf_output_dir (str, required): Specify output directory for performance statistics. If this directory does not exist, it will be created
        * opt (int, optional): Optimization level. Defaults to 2.
        * jobs (int, optional): Number of threads launched during compiler optimization. Defaults to 4.
    """

    from hbdk4.compiler.torch import export

    if not isinstance(march, March):
        march = March.get(march)

    if not os.path.exists(perf_output_dir):
        os.makedirs(perf_output_dir)

    m = export(jit, example_input)
    converted_m = fake_convert(m, march, perf_output_dir)
    _ = internal_compile(converted_m, march, perf_output_dir, opt, jobs)


def _post_process_public_perf_output(perf_output_dir):
    from hbdk4.compiler.tools.public_perf import gen_html

    # In the public release mode, only one json file named "public_perf.json" will be generated
    gen_html(os.path.join(perf_output_dir, "public_perf.json"))


def _post_process_perf_output(perf_output_dir: str):
    import argparse
    from hbdk4.compiler.tools.perf import sub_command_perf

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="sub-commands",
        description="use `hbdk-view {sub-command} -h` for each sub-command's arguments",
    )
    sub_command_perf(subparsers)

    # parse and run
    args = parser.parse_args(["perf", perf_output_dir])
    if hasattr(args, "func"):  # specified a sub-command
        args.func(args)
    else:  # specified no sub-command
        parser.print_help()
    for root, _, files in os.walk(perf_output_dir):
        for f in files:
            filename = os.path.join(root, f)
            if filename.endswith(".html"):
                print(f"Perf output dumped to {filename}")
                return
    raise RuntimeError("Perf output not found")


def post_process_codegen_replay_result(codegen_replay_output_dir: str):
    """Gen html file for each layer group result
    For hbdk4 internal use only
    """
    for root, dirs, _ in os.walk(codegen_replay_output_dir):
        if dirs:
            continue

        is_valid_layer_group = False
        for file in os.listdir(root):
            if (
                os.path.isfile(os.path.join(root, file))
                and file.startswith("tmp_")
                and file.endswith("json")
            ):
                is_valid_layer_group = True

        if is_valid_layer_group:
            _post_process_perf_output(root)


def set_cl_options(argv: List[str]):
    """Set command line options for debugging purpose
    For hbdk4 developer use only
    """
    if not FOR_DEV_USE:
        raise RuntimeError("set_cl_options API is for developer use only")

    _hbdk_cext._set_cl_options(argv)
