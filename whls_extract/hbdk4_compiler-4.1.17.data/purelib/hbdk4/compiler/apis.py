import os
from sys import stdout, stderr
from tempfile import NamedTemporaryFile
from typing import Optional, Union, List
from tempfile import TemporaryDirectory
from hbdk4.compiler.march import MarchBase, March, MarchSeries
from hbdk4.compiler.utils.visualize import OnnxConvertor
from hbdk4.compiler.overlay import Module
from hbdk4.compiler.hbm import Hbm, Hbo
from hbdk4.compiler.utils.process import run_program_redirect_realtime
from hbdk4.compiler._mlir_libs._hbdk import _compile, _convert
from hbdk4.compiler.utils.default import handle_diagnostic
import json


def load(path: str) -> Module:
    """load mlir text or bytecode to mlir.Module

    Args:
        * path (str): A filesystem path to load bytecode ended with \".bc\"

    Raises:
        * ValueError: When \"path\" is not ended with \".bc\"

    Returns:
        * Module: a helper for mlir.Module that manages hbdk operations
    """
    if path.endswith(".bc") or path.endswith(".mlir"):
        with open(path, "rb") as f:
            return Module.parse(f.read())
    raise ValueError("invalid file. should end with .bc")


def save(m: Module, path: str) -> None:
    """save mlir.Module to mlir bytecode

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * path (str): A filesystem path to save bytecode. Must end with \".bc\"

    Raises:
        * ValueError: When \"path\" is not ended with \".bc\"
    """
    if not path.endswith(".bc"):
        raise ValueError("invalid file. should end with .bc")
    with open(path, "wb") as f:
        m.module.write_bytecode(f)


@handle_diagnostic
def convert(
    m: Module, march: Union[MarchBase, str], advice=False, advice_path="", **kwargs
) -> Module:
    """Convert hbir to backend ir.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-e", "nash-m", "nash-p"
        * advice (bool, optional): Set whether to enable op check
        * advice_path (str, optional): path to store op check info. Defaults to empty, print op check info directly without saving the file
    """

    if not isinstance(march, March):
        march = March.get(march)

    args = {
        "march": march,
        "perf_output_dir": ".",
        "advice": str(advice),
        "advice_path": advice_path,
        "expected_backend": ("bpu" if march == March.bayes else "bpu,vpu"),
    }
    cpp_args = {**args, **kwargs}

    new_m = m.clone()
    _convert(new_m.module, m.module.context, cpp_args)
    return new_m


def statistics(m: Module) -> list:
    """Print op statics of given mlir module.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
    """

    ret = []
    for func in m.functions:
        ret.append(func.statistics())
    return ret


def link(hbo_list: List[Hbo], output_path: str, desc: Optional[str] = None):
    """Link hbo to hbm

    Args:
        * hbo_list (List[Hbo): A List of Hbo, which is general by compile
        * output_path (str, required): A filesystem path to save hbm. Must ends with \".hbm\"
        * desc (str optional): Generate a description of hbm when linking, if this parameter is not given, the description information of hbm will not be generated
    """
    if not output_path.endswith(".hbm"):
        raise ValueError(f'{output_path} must ends with ".hbm"')

    if not hbo_list:
        raise RuntimeError("Inputs must not be empty")

    for hbo in hbo_list:
        hbo_name = hbo.get_name()
        if not hbo_name.endswith(".hbo"):
            raise ValueError(f'{hbo_name} must ends with ".hbo"')

    output_dir = os.path.dirname(output_path)
    with NamedTemporaryFile(mode="w", dir=output_dir, delete=True) as temporary_file:
        merge_output_path = temporary_file.name
        hbdk_lld = os.path.join(os.path.dirname(__file__), "_mlir_libs", "hbdk-lld")
        cmd = [hbdk_lld]
        for hbo in hbo_list:
            hbo_name = hbo.get_name()
            cmd += [hbo_name]
        cmd += ["-o"]
        cmd += [merge_output_path]
        if desc is not None:
            cmd += ["-d"]
            cmd += [desc]
        p = run_program_redirect_realtime(cmd, stdout=stdout, stderr=stderr)
        ret = p.returncode
        if ret != 0:
            raise RuntimeError("failed to run hbdk-lld")

        ld_lld = os.path.join(os.path.dirname(__file__), "_mlir_libs", "ld.lld")
        cmd = [ld_lld] + ["-w"] + ["--gc-sections"]
        for hbo in hbo_list:
            hbo_name = hbo.get_name()
            cmd += [hbo_name]
        cmd += [merge_output_path]
        cmd += ["-o"]
        cmd += [output_path]
        p = run_program_redirect_realtime(cmd, stdout=stdout, stderr=stderr)
        ret = p.returncode
        if ret != 0:
            raise RuntimeError("failed to run ld.lld")
        return Hbm(output_path)


@handle_diagnostic
def compile(
    m: Module,
    path: str,
    march: Union[MarchBase, str],
    opt: int = 2,
    jobs: int = 4,
    max_time_per_fc: float = 0.0,
    debug: bool = False,
    hbdk3_compatible_mode: bool = False,
    progress_bar: bool = False,
    advice: float = 0.0,
    balance: int = 100,
    input_no_padding: bool = False,
    output_no_padding: bool = False,
) -> Union[Hbm, Hbo]:
    """Compile hbir module to hbm or hbo.

    If the suffix of the output is 'hbo', it will compile to generate an hbo file.
    Afterward, you need to call the link function to perform linking or packing operations.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-e", "nash-m", "nash-p"
        * path (str, required): A filesystem path to save hbm or hbo. Must ends with \".hbm\" or \".hbo\"
        * opt (int, optional): Optimization level. Defaults to 2.
        * jobs (int, optional): Number of threads launched during compiler optimization. Defaults to 4.
        * max_time_per_fc (float, optional): Set maximum time constraint (unit:us) for per funccall.
        * debug (bool, optional): Set whether to contain debug info in HBM
        * hbdk3_compatible_mode (bool, optional): Set whether to compile in hbdk3 compatible mode, True use hbm3 and False use hbm4
        * progress_bar(bool, optional): Set whether to show progress bar
        * advice(float, optional): Print advice on ops that take longer more than the specified time (unit:us)
        * balance(int, optional): Specify a integer (recommend 2) to balance cycles and DDR access.
                                The integer should be between 0 (minimal DDR access) and 100 (minimal cycles)
        * input_no_padding (bool, optional): Set whether model input is native without padding
        * output_no_padding (bool, optional): Set whether model output is native without padding

    Raises:
        * ValueError: When \"path\" is not ended with \".hbm\"
        * ValueError: When \"balance\" is not in range of [0,100]
    """
    # Record compilation options in hbm
    # To use the original options, place this code at the begin
    localvars = list(locals().items())
    compile_options = {}
    for k, v in localvars:
        if k == "march":
            if isinstance(v, str):
                compile_options[k] = v
            elif isinstance(v, MarchBase):
                compile_options[k] = str(v)
        elif k != "path" and k != "m":
            compile_options[k] = v
    # Ensure that compile_options can be serialized properly
    assert all(
        isinstance(value, (type(None), str, int, float, bool))
        for value in compile_options.values()
    )

    if not isinstance(march, March):
        march = March.get(march)

    args = {
        "march": march,
        "opt": str(opt),
        "jobs": str(jobs),
        "progressbar": str(int(progress_bar)),
        "advice": str(advice),
        "balance": str(balance),
        "max_time_per_fc": max_time_per_fc and f"{max_time_per_fc}" or "",
        "path": path,
        "debug": str(debug),
        "hbdk3_compatible_mode": str(hbdk3_compatible_mode),
        "input-no-padding": str(input_no_padding),
        "output-no-padding": str(output_no_padding),
        "compile_options": json.dumps(compile_options),
    }

    new_m = m.clone()
    if path.endswith(".hbm"):
        # If hbdk3_compatible_mode=True, use hbm3, otherwise use hbm4
        if hbdk3_compatible_mode:
            _compile(new_m.module, m.module.context, args)
            if march.series in (MarchSeries.bayes, MarchSeries.nash):
                return Hbm(path)
            return None
        else:
            hbo_path = path[:-4] + ".hbo"
            hbm = None
            try:
                args["path"] = hbo_path
                _compile(new_m.module, m.module.context, args)
                if march.series in (MarchSeries.bayes, MarchSeries.nash):
                    hbo = Hbo(hbo_path)
                    hbm = link([hbo], path)
            finally:
                if os.path.exists(hbo_path):
                    os.remove(hbo_path)
            return hbm
    elif path.endswith(".hbo"):
        _compile(new_m.module, m.module.context, args)
        if march.series in (MarchSeries.bayes, MarchSeries.nash):
            return Hbo(path)
        return None

    else:
        # Code for check file suffix is before
        raise RuntimeError("Should not reach here")


def visualize(
    m: Module, onnx_file: Optional[str] = None, use_netron: Optional[bool] = False
):
    """Generate visualizable mlir onnx and view it in netron.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * onnx_file (str): path to store onnx proto if it is None then create directory in /tmp
        * use_netron (bool): if True, start netron server to view onnx, otherwise just generate onnx

    """

    import socket
    import time

    servers = []

    for func in m.functions:
        cvt = OnnxConvertor(func)

        if onnx_file is None:
            onnx_file = "./" + func.name + ".onnx"

        cvt.gen_onnx(onnx_file)

        print("\033[92mTemporary onnx file saved to {}\033[0m".format(onnx_file))

        if use_netron:
            import netron

            server = netron.server
            host, port = server.serve(
                onnx_file, None, address=(socket.gethostname(), 0)
            )

            print("\033[92mVisit http://{}:{}\033[0m".format(host, port))

            servers.append(server)

    if use_netron:
        print('\033[92mEnter "c" to shutdown all servers and continue...\033[0m')

        while True:
            time.sleep(2)
            key = input()
            if key.strip() == "c":
                print("\033[92mStopping server...\033[0m")
                for server in servers:
                    server.stop()
                break
