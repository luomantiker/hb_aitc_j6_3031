import os
from typing import List
import traceback
import socket
import time
import logging
import subprocess
from collections import OrderedDict

from hbdk4.compiler.march import March, MarchSeries


class RemoteBPU:
    """
    Examples:
        from hbdk4.compiler import Hbm
        from hbdk4.compiler.remote_bpu import RemoteBPU

        hbm_obj = Hbm("add.hbm")
        print(hbm_obj.graphs[0].name)

        remote_board = RemoteBPU(
            "add.hbm",
            hbm_obj.march,
            hbm_obj.graphs[0].name,
            ["input0.bin", "input1.bin"],
            "horizon.cc", # 192.168.1.10
            22,
            "remote_bpu/",
            "/tmp/"
        )
        remote_board.run_remote_bpu_board()
    """

    def __init__(
        self,
        hbm: str,
        march: March,
        model_name: str = None,
        input_filenames: List[str] = None,
        ip: str = "",
        port: int = 22,
        local_work_path: str = "remote_bpu/",
        remote_work_root: str = "/tmp/",
        username: str = "root",
        password: str = "",
        times: int = 1,
        verbose: bool = False,
    ) -> None:
        """Tool to run Hbm4 model file on remote BPU.

        Create an instance of RemoteBPU and initialize it.
        Args:
            hbm: The hbm file path. It can be abs path or relative path.
            march: Remote BPU march. It should match the march of hbm
            input_filenames: The input file name list. The can be abs path or relative path.
            ip: IP address of remote BPU. If run on bpu board, it's necessary.
            port: Port to login remote BPU. If run on bpu board, it's necessary.
            model_name: Name of model graph to run.
            local_work_path: Local path to save temporary data and final results.
            remote_work_root: Remote bpu path to store temporary data.
            username: Username to login remote BPU.
            password: Password to login remote BPU.
            times: Times to run on BPU, deprecated!
            verbose: Show more all stdout/stderr outputs of subprograms called
        """
        self.__hbm = hbm
        self.__march = march

        self.__input_filenames = input_filenames

        # Remote configuration
        self.__ip = ip
        self.__port = port
        self.__model_name = model_name
        self.__local_work_path = local_work_path
        self.__remote_work_root = remote_work_root
        self.__username = username
        self.__password = password
        self.__times = times
        self.__verbose = verbose

        os.makedirs(self.__local_work_path, exist_ok=True)
        self.__local_work_path = os.path.abspath(self.__local_work_path)

        self.__gen_run_model_path()

    def __gen_run_model_path(self):
        self.__tools_dict = {}
        if self.__march.series == MarchSeries.bayes:
            try:
                from hbdk4.runtime.aarch64_unknown_linux_gnu import bayes as rtlib

                self.__tools_dict["run_model"] = rtlib.get_run_model_bayes_path()
            except ImportError:
                raise RuntimeError(
                    "Cannot import hbdk4.runtime, please install it with bayes march."
                )
        elif self.__march.series == MarchSeries.nash:
            try:
                from hbdk4.runtime.aarch64_unknown_linux_gnu import nash as rtlib

                self.__tools_dict["run_model"] = rtlib.get_run_model_nash_path()
            except ImportError:
                raise RuntimeError(
                    "Cannot import hbdk4.runtime, please install it with nash march."
                )
        else:
            raise RuntimeError("March is illegal in `import hbdk4.runtime`")
        self.__tools_dict["libhbtl"] = rtlib.get_libhbtl_path()

        # Check tools existing
        for path in self.__tools_dict.values():
            assert os.path.exists(path), path + " not found"

    def __gen_run_model_cmd(self) -> str:
        cmd = " ./" + os.path.basename(self.__tools_dict["run_model"])
        # FIXME(wuruidong): Please check `nash_b` and `nash_p` cmd in hbrt4-run-model
        cmd += " -m "
        if self.__march == March.bayes:
            cmd += "Bayes2"
        elif self.__march == March.nash_b:
            cmd += "Bpu30b"
        elif self.__march == March.nash_e or self.__march == March.nash_m:
            cmd += "Bpu30g"
        elif self.__march == March.nash_p:
            cmd += "Bpu30p"
        else:
            raise ValueError(f"Bad value {self.__march}")
        cmd += " -f " + os.path.basename(self.__hbm)
        cmd += " -n " + self.__model_name
        for input_file in self.__input_filenames:
            cmd += " -i " + os.path.basename(input_file)
        cmd += " -o " + self.__bpu_output_path

        # optional log level
        # cmd += " --log-level Trace"
        return cmd

    def __execute_shell_cmd(self, cmd, silent=False):
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

    def run_remote_bpu_board(self) -> str:
        try:
            from paramiko import ssh_exception
            from hbdk4.compiler.utils import bpu_connect
        except ImportError:
            print("Cannot import ssh_exception from paramiko, please install it.")

        any_files_may_have_uploaded = False
        timeout = 1000
        lock_timeout = 0
        total_time = 0
        total_time_to_connect = 0
        total_time_to_acquire_lock = 0
        bpu, bpu_lock_result = None, None
        may_retry = True
        start_time = time.time()
        time_before_lock = time.time()
        time_after_lock = time.time()
        run_result = None
        uploaded_files = set()
        program_to_print_pid = "echo The PID is $PPID."
        regex_to_parse_pid = r"The PID is (\d+)\."

        self.__remote_work_path = os.path.join(
            self.__remote_work_root,
            os.path.basename(self.__local_work_path),
        )

        while True:
            if total_time > timeout:
                raise RuntimeError("bpu execution has timeout(%s)" % str(timeout))
            try:
                time_before_lock = time.time()
                if (not bpu) or (not bpu_lock_result):
                    (
                        bpu,
                        bpu_lock_result,
                        total_time_to_connect,
                        total_time_to_acquire_lock,
                    ) = bpu_connect.connect_to_bpu_and_get_lock(
                        timeout,
                        lock_timeout,
                        self.__ip,
                        self.__port,
                        self.__username,
                        self.__password,
                        self.__remote_work_path,
                        self.__local_work_path,
                        total_time_to_connect,
                        total_time_to_acquire_lock,
                        program_to_print_pid,
                        regex_to_parse_pid,
                    )
                time_after_lock = time.time()
                bpu.exec_command(
                    "rm -rf %s/verifier_*" % self.__remote_work_root,
                    timeout=timeout,
                )
                any_files_may_have_uploaded = True

                bpu.exec_command("mkdir -p " + self.__remote_work_path, timeout=timeout)
                self.__bpu_output_path = os.path.join(
                    self.__remote_work_path, "bpu_output"
                )
                bpu.exec_command("mkdir -p " + self.__bpu_output_path, timeout=timeout)

                uploaded_files_is_executable = OrderedDict()

                def upload_file(filename, dest_folder):
                    real_filename = os.path.abspath(filename)
                    if real_filename in uploaded_files:
                        return
                    is_executable = os.access(filename, os.X_OK)
                    basename = os.path.basename(filename)
                    dest_with_filename = os.path.join(dest_folder, basename)
                    relative_dest_with_filename = os.path.relpath(
                        dest_with_filename, self.__remote_work_path
                    )
                    uploaded_files_is_executable[
                        relative_dest_with_filename
                    ] = is_executable
                    bpu.upload(filename, dest_folder)
                    uploaded_files.add(real_filename)

                def upload(folder_or_file):
                    dest_dir = self.__remote_work_path
                    if os.path.isfile(folder_or_file):
                        return upload_file(folder_or_file, dest_dir)
                    dest_dir = os.path.abspath(
                        os.path.join(dest_dir, os.path.basename(folder_or_file))
                    )
                    for root, __dirs, files in os.walk(folder_or_file):
                        cur_dest_dir = os.path.abspath(
                            os.path.join(
                                dest_dir, os.path.relpath(root, folder_or_file)
                            )
                        )
                        if root not in uploaded_files:
                            bpu.exec_command(
                                "mkdir -p " + cur_dest_dir, timeout=timeout
                            )
                        uploaded_files.add(root)
                        for f in files:
                            upload_file(os.path.join(root, f), cur_dest_dir)

                bpu.upload(self.__hbm, self.__remote_work_path)

                # Please check if there are any other libraries or executable files that need to upload
                upload(self.__tools_dict["run_model"])
                upload(self.__tools_dict["libhbtl"])

                for input_file in self.__input_filenames:
                    abs_file = os.path.abspath(input_file)
                    upload(abs_file)

                chmod_cmd = (
                    "flock -x /tmp/hbdk_model_verifier.lock2 sh -c "
                    + "'%s; cd %s ; %s '"
                    % (
                        program_to_print_pid,
                        self.__remote_work_path,
                        " ; ".join(
                            [
                                ("chmod +x %s" % x)
                                for x in uploaded_files_is_executable.keys()
                                if uploaded_files_is_executable[x]
                            ]
                        ),
                    )
                )
                run_result = bpu.exec_command(
                    chmod_cmd,
                    timeout=timeout,
                    log_stdout=self.__verbose,
                    log_stderr=True,
                    parse_pid_regex=regex_to_parse_pid,
                    name="Chmod",
                )

                # Another lock to ensure locking when hbdk-model-verifier is interrupted
                # The global flock would receive SIGHUP before the following command
                cmd = "flock -x /tmp/hbdk_model_verifier.lock2 sh -c '"
                cmd += "%s; " % program_to_print_pid
                cmd += "echo BPU execution begins; cd " + self.__remote_work_path
                cmd += " ; echo the local work path is %s" % self.__local_work_path
                cmd += " ; ulimit -c 0; sync"
                cmd += ' ; echo BPU model execution begins; if [ -f "/etc/profile" ]; then\n source /etc/profile 2>/dev/null \n fi;'
                cmd += self.__gen_run_model_cmd()
                cmd += " && echo BPU model execution ends"
                # Save as much disk space as possible before making tar
                cmd += " && rm -rf %s" % " ".join(uploaded_files_is_executable.keys())

                cmd += " && echo Begin to compress model output && tar -czvf bpu_output.tar.gz bpu_output"
                cmd += "'"
                cmd += " && rm -rf bpu_output "
                cmd += ";exit_code=$?; echo BPU execution ends; exit $exit_code"
                run_result = bpu.exec_command(
                    cmd,
                    timeout=timeout,
                    log_stdout=self.__verbose,
                    log_stderr=True,
                    parse_pid_regex=regex_to_parse_pid,
                    name="Model Executor",
                )

                def download_result():
                    bpu.download(
                        os.path.join(self.__remote_work_path, "bpu_output.tar.gz"),
                        self.__local_work_path,
                    )

                if run_result.exit_code != 0:
                    msg = "bpu execution returns non-zero exit code " + str(
                        run_result.exit_code
                    )
                    logging.critical(msg)
                    raise RuntimeError(msg)
                else:
                    max_retry_times = 5
                    for i in range(max_retry_times):
                        try:
                            download_result()
                            break
                        except IOError:
                            if i >= max_retry_times - 1:
                                may_retry = False
                                raise
                            else:
                                time.sleep(10)
                break
            except (
                ssh_exception.SSHException,
                socket.error,
                EOFError,
                OSError,
                socket.timeout,
            ) as e:
                if e in (
                    ssh_exception.AuthenticationException,
                    ssh_exception.BadAuthenticationType,
                    ssh_exception.PasswordRequiredException,
                    ssh_exception.NoValidConnectionsError,
                    ssh_exception.BadHostKeyException,
                ):
                    raise
                if not may_retry:
                    raise
                total_time += time.time() - time_before_lock
                if total_time > timeout:
                    raise RuntimeError("bpu execution has timeout(%s)" % str(timeout))
                traceback.print_exc()
                logging.warning(
                    "Connection error while working on bpu. Retrying until the timeout exceeds"
                )
                time.sleep(5)
            finally:
                try:
                    if bpu_lock_result:
                        bpu_lock_result.destroy()
                    if bpu:
                        if any_files_may_have_uploaded:
                            cmd = " rm -rf " + self.__remote_work_path + "; "
                            bpu.exec_command(cmd, timeout=timeout)
                        bpu.close()
                except Exception:
                    traceback.print_exc()
                    logging.warning(
                        "Unexpected exception while doing bpu cleanup. The exception is printed, but ignored"
                    )
                finally:
                    bpu_lock_result = None
                    bpu = None
                    uploaded_files.clear()
                    any_files_may_have_uploaded = False

        logging.info(
            "Verifier Total Time on dev board with connection time"
            " (including BPU, CPU, IO, network and time to wait for lock): %f ms"
            % (time.time() - start_time)
        )
        logging.info(
            "Verifier Total Time on dev board without connection time"
            " (including BPU, CPU, IO and network time): %f ms"
            % (time.time() - time_after_lock)
        )

        # Save cwd before chdir
        saved_cwd = os.getcwd()
        os.chdir(self.__local_work_path)

        self.__execute_shell_cmd("tar -xvf bpu_output.tar.gz")
        self.__execute_shell_cmd("rm bpu_output.tar.gz")

        # chdir back to the original cwd
        os.chdir(saved_cwd)

        return os.path.join(self.__local_work_path, "bpu_output")
