#!/usr/bin/env python
import re
import time
import traceback
import paramiko
import os
import logging
import threading
import socket
from paramiko import ssh_exception
from paramiko.message import Message as ParamikoMessage
from paramiko.common import cMSG_CHANNEL_REQUEST


class BpuExecResult:
    def __init__(
        self,
        *,
        stdin=None,
        stdout=None,
        stderr=None,
        exit_code=-1,
        channel=None,
        bpu_board=None,
        need_cleanup=True,
        pid=None,
        command=None,
        name=""
    ):
        self.command = command
        self.closed = False
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.channel = channel
        self.bpu_board = bpu_board
        self.need_cleanup = need_cleanup
        self.pid = pid
        self.name = name or "Unnamed Command"

    def _logging_cleanup_failure(self):
        logging.warning("Error while cleaning up the command %s" % str(self.command))

    def _close_send_hup_signal(self):
        if (
            self.need_cleanup
            and self.exit_code < 0
            and self.channel
            and not self.channel.closed
        ):
            logging.info('REMOTE: Sending SIGHUP for "%s"' % str(self.name))
            try:
                send_hup_signal_to_channel(self.channel)
            except Exception:
                self._logging_cleanup_failure()
                raise

    def _close_run_kill_cmd(self):
        if self.need_cleanup and self.exit_code < 0 and self.bpu_board and self.pid:
            logging.info(
                'REMOTE: Killing pid %s for "%s"' % (str(self.pid), str(self.name))
            )
            max_retries = 10
            for i in range(max_retries):
                try:
                    if i != 0:
                        time.sleep(3)
                        self.bpu_board.connect()
                    self.bpu_board.exec_command(
                        "kill -s SIGHUP " + str(self.pid),
                        no_wait=False,
                        need_cleanup=False,
                        log_info=False,
                    )
                except Exception:
                    traceback.print_exc()
                    self._logging_cleanup_failure()
                    if i < max_retries - 1:
                        logging.warning("Retry after several seconds")
                    else:
                        raise
                break

    def close(self):
        if self.closed:
            return
        try:
            try:
                self._close_send_hup_signal()
            finally:
                self._close_run_kill_cmd()
        finally:
            if self.stdin and hasattr(self.stdin, "close"):
                self.stdin.close()
            if self.stdout and hasattr(self.stdout, "close"):
                self.stdout.close()
            if self.stderr and hasattr(self.stderr, "close"):
                self.stderr.close()
            if self.channel:
                self.channel.close()
        self.closed = True

    def destroy(self):
        self.close()
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.channel = None
        self.bpu_board = None

    def __del__(self):
        self.destroy()


def send_hup_signal_to_channel(channel):
    """
    See ssh protocol spec:
    https://tools.ietf.org/html/rfc4254#section-8
    """
    msg = ParamikoMessage()
    msg.add_byte(cMSG_CHANNEL_REQUEST)
    msg.add_int(channel.remote_chanid)
    msg.add_string("signal")
    msg.add_boolean(False)
    msg.add_string("HUP")
    channel.transport._send_user_message(msg)


class BPUBoard(object):
    def __init__(self, host, port, username, password):
        self._host = host
        self._username = username
        self._password = password
        self._port = port
        self._client = None
        self._sftp = None
        self.is_connected = False
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if self._password == "":
            self._password = self._password.strip()

    def connect(self):
        try:
            self._client.load_system_host_keys()
            self._client.connect(
                hostname=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
            )
            transport = self._client.get_transport()
            transport.set_keepalive(60)
            self._sftp = paramiko.SFTPClient.from_transport(transport)
            self.is_connected = True
        except paramiko.BadHostKeyException:
            logging.critical(
                "can not establish ssh connection to bpu board."
                " invalid user name, password, or invalid host key!"
                " Make sure you can connect to the board using the 'ssh' command"
            )
            raise
        except paramiko.AuthenticationException:
            if self._password == "":
                try:
                    self._client.get_transport().auth_none(username=self._username)
                    self._sftp = paramiko.SFTPClient.from_transport(
                        self._client.get_transport()
                    )
                    self.is_connected = True
                except Exception:
                    logging.critical(
                        "can not establish ssh connection to bpu board. authentication failed!"
                    )
                    raise
            else:
                logging.critical(
                    "can not establish ssh connection to bpu board. authentication failed!"
                )
                raise
        except paramiko.SSHException:
            logging.critical("can not establish ssh connection to bpu board.")
            raise

    def download(self, remotepath, localpath):
        try:
            localpath = os.path.join(localpath, os.path.basename(remotepath))
            logging.info("Downloading " + remotepath + " to " + localpath)
            self._sftp.get(remotepath, localpath)
        except Exception:
            logging.critical(
                "Error occurs when downloading file " + remotepath + " from BPU"
            )
            raise

    def upload(self, localpath, remotepath, dest_is_dir=True):
        try:
            if dest_is_dir:
                remotepath = os.path.join(remotepath, os.path.basename(localpath))
            logging.info("Uploading " + localpath + " to " + remotepath)
            self._sftp.put(localpath, remotepath)
        except Exception:
            logging.critical(
                "Error occurs when uploading file " + localpath + " to BPU"
            )
            raise

    def exec_command(
        self,
        command,
        timeout=None,
        no_wait=False,
        bufsize=-1,
        get_pty=False,
        log_stdout=False,
        log_stderr=False,
        need_cleanup=True,
        parse_pid_regex=None,
        log_info=True,
        name="",
    ):
        if no_wait and (log_stdout or log_stderr):
            raise RuntimeError(
                "log_stdout and log_stderr is only implemented for no_wait==False"
            )
        # NOTE: get_pty=True will redirect stderr to stdout
        # NOTE: get_pty=True will ensure command receives SIGHUP upon disconnection
        stdin, stdout, stderr = self._client.exec_command(
            command, timeout=timeout, get_pty=get_pty, bufsize=bufsize
        )
        channel = stdout.channel or stderr.channel or None
        if log_info:
            logging.info("REMOTE: executing [" + command + "]")
        # Must return stdin to avoid it to be garbage collected
        result = BpuExecResult(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            channel=channel,
            bpu_board=self,
            need_cleanup=need_cleanup,
            command=command,
            name=name,
        )
        regex = None
        if parse_pid_regex:
            regex = re.compile(parse_pid_regex)
        if not no_wait:
            # Use threading to avoid remote output to be larger than the current Transport or session's window_size
            result.stdout = []
            result.stderr = []
            if log_stdout or regex:

                def get_stdout():
                    nonlocal regex
                    lines = []
                    for line in stdout:
                        if regex:
                            match = re.search(regex, line)
                            if match:
                                try:
                                    result.pid = int(match.groups()[0])
                                    regex = None
                                    logging.info(
                                        "REMOTE: command PID is " + str(result.pid)
                                    )
                                except (IndexError, ValueError):
                                    traceback.print_exc()
                                    logging.warning(
                                        "Fail to parse pid from string %s with regex %s"
                                        % (line, parse_pid_regex)
                                    )
                        lines.append(line)
                        if log_stdout:
                            logging.info(line.strip())
                    result.stdout = lines
                    return lines

            else:

                def get_stdout():
                    result.stdout = stdout.readlines()
                    return result.stdout

            if log_stderr:

                def get_stderr():
                    lines = []
                    for line in stderr:
                        lines.append(line)
                        logging.warning(line.strip())
                    result.stderr = lines

            else:

                def get_stderr():
                    result.stderr = stderr.readlines()
                    return result.stderr

            try:
                f_stdout = threading.Thread(target=get_stdout, daemon=True)
                f_stderr = threading.Thread(target=get_stderr, daemon=True)
                f_stdout.start()
                f_stderr.start()
                f_stdout.join()
                f_stderr.join()
                result.exit_code = channel.recv_exit_status()
            finally:
                result.close()
            if log_info:
                logging.info("REMOTE: command exit code is " + str(result.exit_code))
        return result

    def close(self):
        self.is_connected = False
        if self._client:
            self._client.close()

    def __del__(self):
        self.close()


def connect_to_bpu_and_get_lock(
    timeout,
    lock_timeout,
    ip,
    port,
    username,
    password,
    remote_work_path,
    local_work_path,
    total_time_to_connect,
    total_time_to_acquire_lock,
    program_to_print_pid,
    regex_to_parse_pid,
):

    bpu_flock_cmd = "flock -x /tmp/hbdk_model_verifier.lock"
    bpu = None
    bpu_lock_result = None
    logging.info("======> Try to connect BPU")
    bpu_connected_and_lock_acquired = False
    bpu = None
    while True:
        if total_time_to_connect > timeout:
            raise RuntimeError("Timeout exceeds while connecting to bpu")
        if lock_timeout and total_time_to_acquire_lock > lock_timeout:
            raise RuntimeError("Lock timeout exceeds while connecting to bpu")
        try:
            start_connect_time = time.time()
            try:
                bpu = BPUBoard(
                    ip,
                    port,
                    username,
                    password,
                )
                bpu.connect()
                logging.info("======> BPU connected")
            finally:
                total_time_to_connect += time.time() - start_connect_time

            bpu_lock_start_time = time.time()
            try:
                max_bpu_lock_time = max(3600, timeout + 1800)
                bpu_acquiring_lock_msg = "Acquiring lock for model verifier on bpu"
                bpu_lock_acquired_msg = (
                    "Model verifier %s from %s lock on bpu has been acquired"
                    % (
                        remote_work_path,
                        local_work_path,
                    )
                )
                logging.info(bpu_acquiring_lock_msg)
                bpu_lock_result = bpu.exec_command(
                    bpu_flock_cmd
                    + " sh -c '%s; echo %s; sleep %s'"
                    % (
                        program_to_print_pid,
                        bpu_lock_acquired_msg,
                        str(max_bpu_lock_time),
                    ),
                    bufsize=0,
                    no_wait=True,
                    get_pty=True,
                    name="Verifier Global Lock",
                )

                bpu_lock_handled_event = threading.Event()
                is_bpu_lock_acquired = False

                def acquire_bpu_lock():
                    nonlocal is_bpu_lock_acquired
                    try:
                        while bpu_lock_result.stdout:
                            line = bpu_lock_result.stdout.readline()
                            if bpu_lock_handled_event.is_set():
                                break
                            if not line:
                                break
                            pid_match = re.search(regex_to_parse_pid, line)
                            if pid_match:
                                try:
                                    bpu_lock_result.pid = int(pid_match.groups()[0])
                                    logging.info(
                                        "REMOTE: The PID of the lock is %s"
                                        % str(bpu_lock_result.pid)
                                    )
                                except (IOError, ValueError):
                                    traceback.print_exc()
                                    logging.warning(
                                        "Fail to get the pid of the lock program"
                                    )
                            if line.find(bpu_lock_acquired_msg) != -1:
                                is_bpu_lock_acquired = True
                                break
                    finally:
                        bpu_lock_handled_event.set()
                    return is_bpu_lock_acquired

                def helper_no_lock_for_a_while():
                    nonlocal bpu_lock_handled_event
                    bpu_lock_handled_event.set()

                bpu_lock_retry_internal = 60
                time_until_timeout = bpu_lock_retry_internal
                if lock_timeout:
                    time_until_timeout = min(
                        bpu_lock_retry_internal,
                        max(1, lock_timeout - total_time_to_acquire_lock),
                    )

                bpu_lock_timer = threading.Timer(
                    time_until_timeout, helper_no_lock_for_a_while
                )
                bpu_lock_timer.daemon = True
                bpu_lock_timer.start()
                bpu_lock_thread = threading.Thread(target=acquire_bpu_lock)
                bpu_lock_thread.start()

                while not bpu_lock_handled_event.is_set():
                    bpu_lock_handled_event.wait()

                bpu_lock_timer.cancel()
                if is_bpu_lock_acquired:
                    bpu_lock_thread.join()
                    bpu_connected_and_lock_acquired = True
                    break
                else:
                    # close lock related stdin/stdout and join thread, to avoid resource leak
                    bpu_lock_result.destroy()
                    bpu_lock_thread.join()
                    bpu_lock_result = None
                    if lock_timeout and total_time_to_acquire_lock > lock_timeout:
                        raise RuntimeError(
                            "Lock timeout exceeds while connecting to bpu"
                        )
                    logging.info(
                        "Fail to get bpu lock within %s. Retry connection"
                        % str(bpu_lock_retry_internal)
                    )
            finally:
                total_time_to_acquire_lock += time.time() - bpu_lock_start_time
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
            if total_time_to_connect > timeout:
                raise RuntimeError("Timeout exceeds while connecting to bpu")
            if lock_timeout and total_time_to_acquire_lock > lock_timeout:
                raise RuntimeError("Lock timeout exceeds while connecting to bpu")
            traceback.print_exc()
            logging.warning(
                "Connection error while connecting to bpu. The error has been printed. "
                "Retrying until timeout exceeds "
            )
        finally:
            if not bpu_connected_and_lock_acquired:
                try:
                    if bpu_lock_result:
                        bpu_lock_result.destroy()
                    if bpu:
                        bpu.close()
                except Exception:
                    traceback.print_exc()
                    logging.warning(
                        "Unexpected exception while doing bpu cleanup. The exception is printed, but ignored"
                    )
                finally:
                    bpu_lock_result = None
                    bpu = None

    if (not bpu) or (not bpu.is_connected):
        raise RuntimeError("Fail to connect to bpu")
    return bpu, bpu_lock_result, total_time_to_connect, total_time_to_acquire_lock
