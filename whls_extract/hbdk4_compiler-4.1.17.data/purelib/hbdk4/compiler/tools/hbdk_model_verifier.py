#!/usr/bin/env python
from typing import List, Union
from hbdk4.compiler import Module
from hbdk4.compiler.utils import hbm_info
from hbdk4.compiler.hbm import Hbm, VariableInputSemantic
from hbdk4.compiler.march import March
from hbdk4.compiler import hbtl
from hbdk4.compiler import RemoteBPU

import numpy as np
import datetime
import subprocess
import sys
import os
import re
import traceback
import shutil
import logging


class Verifier:
    def __init__(
        self,
        hbm: str,
        converted_module: Module = None,
        input: List[Union[str, np.ndarray]] = None,
        ip: str = "",
        port: int = 22,
        model_name: str = None,
        memory: int = None,
        yuv_shape: List[List[int]] = None,
        yuv_roi: List[List[int]] = None,
        image_stride: List[int] = None,
        local_work_path: str = ".",
        remote_work_root: str = "/tmp/",
        username: str = "root",
        password: str = "",
        times: int = 1,
        hbrt_log_level: int = 0,
        verbose: bool = False,
    ):
        """Tool to verify the consistency of deployed model for Horizon BPU.

        Create an instance of the Verifier and initialize it.
        Args:
            hbm: The hbm file path.
            converted_module: The mlir converted from hbir to march,
                                can be obtained by calling hbdk4.compiler.apis.convert().
                                If run by hbir interpreter, it's necessary.
            input: The input file path.
            ip: IP address of remote BPU. If run on bpu board, it's necessary.
            port: Port to login remote BPU. If run on bpu board, it's necessary.
            model_name: Name of model to verify.
            memory: Specify the max memory usage(unit: MiB) of bpu in simulator.
                    Ignored if run_sim = False.
            yuv_shape: YUV shape for resizer/pyramid, [H,W].
                        If the model has multiple inputs from different sources, some without --yuv-shape,
                        give empty List as placeholder, like [[],[H,W],[]].
            yuv_roi: ROI coordinate for resizer/pyramid model, [h0,w0,h1,w1].
                        (h0,w0) is the lower left corner, and (h1,w1) is the upper right corner.
                        Placeholder similar to yuv_shape.
            image_stride: W stride for resizer/pyramid. Placeholder similar to yuv_shape.
            local_work_path: Local path to save temporary data and final results.
            remote_work_root: Remote bpu path to store temporary data.
            username: Username to login remote BPU.
            password: Password to login remote BPU.
            times: Times to run on BPU.
            hbrt_log_level: The log level for hbrt.
            verbose: Show more all stdout/stderr outputs of subprograms called
        """

        self.arg_dict = {}
        self.arg_dict["hbm"] = os.path.abspath(hbm)
        self.arg_dict["converted_module"] = converted_module
        self.arg_dict["input"] = input
        self.arg_dict["ip"] = ip
        self.arg_dict["port"] = port
        self.arg_dict["model_name"] = model_name
        self.arg_dict["memory"] = memory
        self.arg_dict["yuv_shape"] = yuv_shape
        self.arg_dict["yuv_roi"] = yuv_roi
        self.arg_dict["image_stride"] = image_stride
        self.arg_dict["local_work_path"] = local_work_path
        self.arg_dict["remote_work_root"] = remote_work_root
        self.arg_dict["username"] = username
        self.arg_dict["password"] = password
        self.arg_dict["times"] = times
        self.arg_dict["hbrt_log_level"] = hbrt_log_level
        self.arg_dict["verbose"] = verbose
        self.arg_dict["no_bpu_clean"] = False
        self.arg_dict["clean_remote_work_path"] = True
        check_tool_existence(self)
        generate_input(self)

    def compare_result(self, run_sim=True, threshold=0.999):
        """Compare hbir interpreter/simulator/bpu board inference results.

        Args:
            run_sim (bool, optional): Choose whether to run the simulator and compare its results.
                                    "True" by default.
            threshold (float, optional): Threshold for cosine similarity, as close to 1 as possible.
                                    "0.999" by default.
        """

        result_sim = None
        result_hbir = None
        result_bpu = None

        if self.arg_dict["converted_module"] is not None:
            result_hbir = run_by_hbir_interpreter(self)
        if run_sim:
            result_sim = run_by_simulator(self)
        if self.arg_dict["ip"] != "":
            result_bpu = run_on_bpu_board(self)

        if result_hbir and result_sim:
            logging.info("======> Compare Results of Hbir Interpreter vs Simulator")
            calculate_cos_similarity(result_hbir, result_sim, threshold)
        if result_hbir and result_bpu:
            logging.info("======> Compare Results of Hbir Interpreter vs BPU Board")
            calculate_cos_similarity(result_hbir, result_bpu, threshold)
        if result_sim and result_bpu:
            logging.info("======> Compare Results of Simulator vs BPU Board")
            calculate_cos_similarity(result_sim, result_bpu, threshold)


def check_tool_existence(verifier: Verifier):
    logging.info("======> Check tool existence")
    verifier.arg_dict["x86_run_model_program_path"] = "hbdk-run-model-x86"

    # Note: BPU runtime tools are defined in `remote_bpu.py`, they are imported in initializer of RemoteBPU.

    tool = verifier.arg_dict
    tool_list = [
        shutil.which(tool["x86_run_model_program_path"]),
    ]

    for tool_path in tool_list:
        assert os.path.exists(tool_path), tool_path + " not found"
    logging.info("======> Tools all detected")


input_infos = []


def generate_input(verifier: Verifier):
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, "%m%d%H%M%S")
    path_prefix = "verifier_" + time_str
    os.makedirs(path_prefix, exist_ok=True)
    local = os.path.abspath(verifier.arg_dict["local_work_path"])
    verifier.arg_dict["local_work_path"] = os.path.join(local, path_prefix)
    verifier.arg_dict["input_data"] = []

    input_infos.clear()
    build_info(verifier)

    for info in input_infos:
        info.generate_random_input_file(
            verifier.arg_dict["local_work_path"],
            verifier.arg_dict["converted_module"],
            verifier.arg_dict["input_data"],
        )


def calculate_cos_similarity(arr1, arr2, threshold):
    assert len(arr1) == len(arr2), "Inconsistent number of output files!"

    if True in np.isnan(arr1):
        raise ValueError("The first output has NAN values!")
    if True in np.isnan(arr2):
        raise ValueError("The second output has NAN values!")

    compare_flag = True
    similarities = []
    for idx in range(len(arr1)):
        res1 = arr1[idx].flatten()
        res2 = arr2[idx].flatten()

        # prevent data overflow
        res1 = res1.astype(np.float64)
        res2 = res2.astype(np.float64)
        dot = np.dot(res1, res2)
        norm1 = np.linalg.norm(res1)
        norm2 = np.linalg.norm(res2)
        similarity = dot / (norm1 * norm2)
        similarities.append(similarity)

        if similarity < threshold:
            compare_flag = False
            logging.error(
                "The cosine similarity is {:.3f}, which is below the threshold of {:.3f}.".format(
                    similarity, threshold
                )
            )

    if compare_flag:
        logging.info("VERIFY SUCCESS!")
    else:
        deviations = [abs(1 - similarity) for similarity in similarities]
        max_deviation = max(deviations)
        avg_deviation = np.mean(deviations)
        logging.error(
            "The maximum deviation is {:.3f}, and the average deviation is {:.3f}.".format(
                max_deviation, avg_deviation
            )
        )
        logging.error("VERIFY FAILED!")


def run_by_hbir_interpreter(verifier: Verifier):
    logging.info("======> Run Model by Hbir")
    verifier.arg_dict["hbir_output_path"] = os.path.join(
        verifier.arg_dict["local_work_path"], "hbir_output"
    )
    os.makedirs(verifier.arg_dict["hbir_output_path"], exist_ok=True)

    model_name = verifier.arg_dict["model_name"]
    func = verifier.arg_dict["converted_module"][model_name]
    input_dict = {}
    res = []
    for idx in range(len(func.inputs)):
        input_dict[func.inputs[idx].name] = verifier.arg_dict["input_data"][idx]
    res_dict = func.feed(input_dict)
    for idx in range(len(func.outputs)):
        res.append(res_dict[func.outputs[idx].name])

    for i, arr in enumerate(res):
        flat_arr = arr.reshape(arr.shape[0], -1)
        filename = os.path.join(
            verifier.arg_dict["hbir_output_path"], f"hbir_interpreter_output_{i}.txt"
        )
        np.savetxt(filename, flat_arr)

    logging.info("======> Run Model by Hbir Done")
    return list(res)


def run_by_simulator(verifier: Verifier):
    logging.info("======> Run Model by Simulator")
    verifier.arg_dict["sim_output_path"] = os.path.join(
        verifier.arg_dict["local_work_path"], "sim_output"
    )
    os.makedirs(verifier.arg_dict["sim_output_path"], exist_ok=True)

    cmd = gen_run_cmd_param(verifier, for_sim=True)
    orig_path = os.getcwd()
    os.chdir(verifier.arg_dict["sim_output_path"])
    out, err, returncode = execute_shell_cmd(cmd)

    if out and verifier.arg_dict["verbose"]:
        logging.warning("receive following output from simulator:")
        logging.info(out)
    if err:
        logging.warning("receive following warning from simulator:")
        logging.warning(err)
    if returncode != 0:
        logging.critical("Fail to execute {}".format(cmd))
        sys.exit(1)

    output_generated = False
    for file in os.listdir(verifier.arg_dict["sim_output_path"]):
        if file.find("hbdk_output") != -1:
            output_generated = True
    if not output_generated:
        logging.critical(
            "output is not generated by simulator. Please check log for more details."
        )
        sys.exit(1)

    os.chdir(orig_path)
    logging.info("======> Run Model by Simulator Done.")
    return get_sim_output_array(verifier)


def run_on_bpu_board(verifier: Verifier):
    logging.info("======> Run Model by remote BPU")
    input_filenames = []
    for i in input_infos:
        input_filenames.append(i.bpu_input_file)
    logging.info("Remote BPU input filenames: " + "{}".format(input_filenames))

    bpu_obj = RemoteBPU(
        verifier.arg_dict["hbm"],
        verifier.arg_dict["march"],
        verifier.arg_dict["model_name"],
        input_filenames,
        verifier.arg_dict["ip"],
        verifier.arg_dict["port"],
        verifier.arg_dict["local_work_path"],
        verifier.arg_dict["remote_work_root"],
        verifier.arg_dict["username"],
        verifier.arg_dict["password"],
        verifier.arg_dict["times"],
        verifier.arg_dict["verbose"],
    )

    ret_path = bpu_obj.run_remote_bpu_board()
    logging.info("Download Remote BPU running results in path: " + ret_path)
    logging.info("======> Run Model by remote BPU Done.")
    verifier.arg_dict["bpu_output_path"] = ret_path
    return get_bpu_output_array(verifier)


def build_info(verifier: Verifier):
    hbmObj = Hbm(verifier.arg_dict["hbm"])
    verifier.arg_dict["march"] = hbmObj.march
    hbmGraphs = hbmObj.graphs
    if verifier.arg_dict["model_name"] is not None:
        model = [d for d in hbmGraphs if d.name == verifier.arg_dict["model_name"]][0]
    else:
        model = hbmGraphs[0]
        verifier.arg_dict["model_name"] = [d.name for d in hbmGraphs][0]

    input_files = None
    if verifier.arg_dict["input"] is not None:
        input_files = verifier.arg_dict["input"]
        if len(input_files) != len(model.inputs):
            logging.critical(
                "input file number "
                + str(len(input_files))
                + " != model input tensor number "
                + str(len(model.inputs))
            )
            sys.exit(1)

    # Note: `output_name` is only used for matching and loading remote bpu running results.
    verifier.arg_dict["output_name"] = []

    verifier.arg_dict["output_type"] = []
    for idx in range(len(model.outputs)):
        element_type = model.outputs[idx].type.np_dtype
        verifier.arg_dict["output_type"].append(element_type)
        name = model.outputs[idx].name
        verifier.arg_dict["output_name"].append(name)

    for idx in range(len(model.inputs)):

        input_semantic = None
        image_shape = None
        yuv_roi = None
        image_stride = None
        image_mode = None

        if not model.inputs[idx].children:
            feature_info = model.inputs[idx]
            input_semantic = "normal"
        else:
            feature_info = model.inputs[idx].children[0]
            if (
                model.inputs[idx].children[-1].input_semantic
                == VariableInputSemantic.ImageRoi
            ):
                input_semantic = "resizer"
            else:
                input_semantic = "pyramid"
            if (
                len(model.inputs[idx].children) > 1
                and model.inputs[idx].children[1].input_semantic
                == VariableInputSemantic.ImageUv
            ):
                image_mode = "nv12"
            else:
                image_mode = "gray"

            if verifier.arg_dict["image_stride"]:
                image_stride = verifier.arg_dict["image_stride"][idx]
            if verifier.arg_dict["yuv_roi"]:
                yuv_roi = verifier.arg_dict["yuv_roi"][idx]
            else:
                yuv_roi = (0, 0, 64, 64)

        input_file = None
        file_suffix = ".bin"
        if input_files is not None:
            if hbtl.is_tensor(input_files[idx]):
                data = input_files[idx]
                if hbtl.is_torch_tensor(data):
                    data = data.numpy()
                data = data.flatten()
                if image_mode == "nv12":
                    file_suffix = ".yuv"
                elif image_mode == "gray":
                    file_suffix = ".y"
                input_files[idx] = os.path.join(
                    verifier.arg_dict["local_work_path"],
                    "input_" + str(idx) + str(file_suffix),
                )
                file = open(input_files[idx], "w")
                file.close()
                data.tofile(input_files[idx])
            input_file = os.path.abspath(input_files[idx])

        ext = None
        if input_file is not None:
            ext = input_file.split(".")[-1]

            if ext in ("y", "yuv"):
                assert verifier.arg_dict["yuv_shape"][
                    idx
                ], "resizer input must specify yuv_shape"
                image_shape = verifier.arg_dict["yuv_shape"][idx]
                if ext == "y":
                    image_shape.append(1)
                else:
                    image_shape.append(3)

        feature_name = feature_info.name
        element_type = feature_info.type.np_dtype
        dims = feature_info.type.dims

        input_info = hbm_info.InputInfo(
            input_idx=idx,
            feature_name=feature_name,
            filename=input_file,
            dims=dims,
            input_semantic=input_semantic,
            element_type=element_type,
            image_shape=image_shape,
            image_roi=yuv_roi,
            image_stride=image_stride,
            image_mode=image_mode,
        )
        input_infos.append(input_info)


def get_sim_output_array(verifier: Verifier):
    output_path = verifier.arg_dict["sim_output_path"]

    files = os.listdir(output_path)
    res = []
    for idx in range(len(verifier.arg_dict["output_type"])):
        if verifier.arg_dict["output_type"][idx] == "float32":
            file_suffix = "float.txt"
        else:
            file_suffix = "int.txt"

        for f in files:
            if re.match(".*output" + str(idx) + ".*" + file_suffix, f):
                res.append(
                    read_output_file(
                        os.path.join(output_path, f),
                        verifier.arg_dict["output_type"][idx],
                    )
                )
                continue
    return res


def get_bpu_output_array(verifier: Verifier) -> list:
    output_path = verifier.arg_dict["bpu_output_path"]

    files = os.listdir(output_path)
    res = []

    for idx, output_name in enumerate(verifier.arg_dict["output_name"]):
        tmp_file_name = output_name + ".bin"
        if tmp_file_name in files:
            output_type = verifier.arg_dict["output_type"][idx]
            if output_type is None:
                raise ValueError(
                    "The graph output type cannot be `None`, please check HBM compiler result"
                )
            res.append(
                np.fromfile(os.path.join(output_path, tmp_file_name), output_type)
            )

    return res


def read_output_file(path, output_type):
    tmp_output = os.path.join(os.path.dirname(path), "output.txt")
    with open(path, "r") as f_in, open(tmp_output, "w") as f_out:
        for line in f_in:
            if not line.startswith("#"):
                f_out.write(line)

    return np.loadtxt(tmp_output, output_type)


def get_arg_from_input(arg_str, func_to_get_more_arg):
    s = ""
    is_empty_arg = True
    for input_info in input_infos:
        this = func_to_get_more_arg(input_info)
        if this:
            is_empty_arg = False
            s += this
        s += ","
    if s:
        s = s.strip(",")
    if is_empty_arg:
        return ""
    else:
        return arg_str + " " + s


def execute_shell_cmd(cmd, silent=False):
    if not silent:
        logging.info("executing cmd: " + cmd)
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    if err is not None:
        if err.decode():
            err = err.decode(errors="replace")
    if out is not None:
        out = out.decode(errors="replace")
    return out, err, p.returncode


def gen_run_cmd_param(verifier: Verifier, for_sim):
    res = "export HBRT_LOG_LEVEL=%s && " % str(verifier.arg_dict["hbrt_log_level"])
    if for_sim:
        res += verifier.arg_dict["x86_run_model_program_path"]
        res += " -i "
        for input_info in input_infos:
            res += input_info.bpu_input_file + ","
        res = res[0:-1]
    else:
        res += "./" + os.path.basename(
            verifier.arg_dict["aarch64_run_model_program_path"]
        )
        for input_info in input_infos:
            res += " -i " + os.path.basename(input_info.bpu_input_file) + " "

    try:
        res += " -n " + verifier.arg_dict["model_name"]

        if for_sim:
            res += " -f " + verifier.arg_dict["hbm"]
            res += " -o " + verifier.arg_dict["sim_output_path"]
        else:
            march = verifier.arg_dict["march"]
            if march == March.nash_e or march == March.nash_m:
                res += " -m Bpu30g"
            res += " -f " + os.path.basename(verifier.arg_dict["hbm"])
            res += " -o " + verifier.arg_dict["bpu_output_path"]

        res += get_arg_from_input(
            " --yuv-img-size",
            lambda info: (str(info.image_shape[0]) + "x" + str(info.image_shape[1]))
            if info.image_shape
            else "",
        )
        res += get_arg_from_input(
            " --yuv-stride",
            lambda info: str(info.image_stride)
            if info.image_stride and info.stride_speicified_manually
            else "",
        )
        res += get_arg_from_input(
            " --yuv-roi",
            lambda info: (
                str(info.image_roi[0])
                + "x"
                + str(info.image_roi[1])
                + "x"
                + str(info.image_roi[2])
                + "x"
                + str(info.image_roi[3])
            )
            if info.image_roi and info.input_semantic == "resizer"
            else "",
        )
        res += get_arg_from_input(
            " --yuv-roi-coord",
            lambda info: (str(info.image_roi[0]) + "x" + str(info.image_roi[1]))
            if info.image_roi and info.input_semantic == "pyramid"
            else "",
        )
        res += get_arg_from_input(
            " --yuv-roi-size",
            lambda info: (str(info.image_roi[2]) + "x" + str(info.image_roi[3]))
            if info.image_roi and info.input_semantic == "pyramid"
            else "",
        )

        if for_sim and verifier.arg_dict["memory"]:
            res += " --memory " + str(verifier.arg_dict["memory"])
        if verifier.arg_dict["times"] != 1:
            res += " --dev-times %d" % verifier.arg_dict["times"]

    except Exception:
        traceback.print_exc()
        logging.critical("can not generate cmd parameters from arg dict!")
        sys.exit(1)
    return res
