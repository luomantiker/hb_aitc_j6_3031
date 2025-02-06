import os
import sys
import json
import tempfile
from typing import Sequence, Union

from hbdk4.compiler.utils.process import run_program_redirect_realtime
from hbdk4.compiler.march import MarchBase, March


def is_elf(file_path):
    elf_magic = b"\x7fELF"
    with open(file_path, "rb") as f:
        file_header = f.read(16)
        if file_header[:4] == elf_magic:
            return True
        else:
            return False


def _run_hbm_tool_with_cmd(tool_name: str, cmd: Sequence[str]):
    assert os.path.isabs(cmd[0])
    p = run_program_redirect_realtime(cmd, stdout=sys.stdout, stderr=sys.stderr)
    ret = p.returncode
    if ret != 0:
        raise RuntimeError(f"HBDK {tool_name} FAIL, please check with HBDK Group")
    else:
        print(f"HBDK {tool_name} SUCCESS")
    return ret


def _binary_desc_to_bytes(desc_dict: dict):
    for k, v in desc_dict.items():
        if isinstance(v, dict):
            _binary_desc_to_bytes(v)
        if k == "desc_type" and v == "binary":
            if "desc" not in desc_dict:
                raise ValueError(
                    "desc dict contains desc_type but desc content is missing"
                )
            desc_dict["desc"] = bytes.fromhex(desc_dict["desc"])


def hbm_extract_desc(model: str) -> dict:
    """Extract hbm desc info

    DEPRECATED: It will be removed in the future

    Args:
        * model (str): Hbm model file name

    Return:
        * desc_dict (dict): Hbm desc info
    """
    tool_name = "hbdk-hbm-attach"
    if is_elf(model):
        tool_name = "hbdk-hbm-desc"

    tool = os.path.join(os.path.dirname(__file__), "_mlir_libs", tool_name)
    cmd = [tool, model]

    with tempfile.NamedTemporaryFile(suffix="_desc.json") as f:
        filename = f.name
        cmd += ["-o", filename]
        _run_hbm_tool_with_cmd("hbm extract desc", cmd)
        desc_dict = json.load(f)
        _binary_desc_to_bytes(desc_dict)
        return desc_dict


# NOTE: json does not support to serialize python bytes by default.
# Convert it to hex string and process in hbdk-hbm-desc c++ tool.
class BytesToHexEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            return o.hex()
        return super().default(o)


def hbm_update_desc(model: str, desc_dict: dict):
    """Update hbm desc info

    DEPRECATED: It will be removed in the future

    Args:
        * model (str): Hbm model file name
        * desc_dict (dict): Hbm desc info
    """
    tool_name = "hbdk-hbm-attach"
    if is_elf(model):
        tool_name = "hbdk-hbm-desc"

    tool = os.path.join(os.path.dirname(__file__), "_mlir_libs", tool_name)
    cmd = [tool, model, "-u"]

    with tempfile.NamedTemporaryFile(suffix="_desc.json") as f:
        filename = f.name
        with open(filename, "w") as ff:
            json.dump(desc_dict, ff, indent=4, cls=BytesToHexEncoder)
        cmd += ["-i", filename]
        _run_hbm_tool_with_cmd("hbm update desc", cmd)


def hbm_perf(model: str, output_dir: str = None):
    """HBM performance analysis

    Args:
        * model (str): Hbm model file name
        * output_dir (str): Output directory to hold the results, default to the current path

    Return:
        * 0 if no error
    """

    tool = os.path.join(os.path.dirname(__file__), "_mlir_libs", "hbdk4-perf")
    cmd = [tool, model]

    if output_dir:
        cmd += ["-o", output_dir]

    ret = _run_hbm_tool_with_cmd("hbm perf", cmd)
    return ret


def hbm_pack(
    models: Sequence[str], output: str = None, tag: str = None, reuse_param: bool = True
):
    """Pack hbm model files

    DEPRECATED: Use `hbdk4.compiler.apis.link` instead

    Args:
        * models (Sequence): Hbm model file names
        * output (str): Output hbm file name
        * tag (str): Set hbm Tag
        * reuse_param (bool): Enable param sharing

    Return:
        * 0 if no error
    """

    tool = os.path.join(os.path.dirname(__file__), "_mlir_libs", "hbdk-pack")
    cmd = [tool]
    for m in models:
        if is_elf(m):
            raise RuntimeError(
                "Input file is based on new file format. Call `hbdk4.compiler.apis.link` instead"
            )
        cmd += [m]
    if output:
        cmd += ["-o", output]
    if not reuse_param:
        cmd += ["--fno-reuse-param"]
    if tag:
        cmd += ["--tag", tag]

    ret = _run_hbm_tool_with_cmd("hbm pack", cmd)
    return ret
