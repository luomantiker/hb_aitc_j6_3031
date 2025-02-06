import subprocess
import threading


def run_program_redirect_realtime(args, stdout=None, stderr=None, env=None, **kwargs):
    """
    Run subprocess while argument is dumping to stdout/stderr in realtime, before subprcess ends
    """
    p_stdout = None
    if stdout is not None:
        p_stdout = subprocess.PIPE
    p_stderr = None
    if stderr is not None:
        p_stderr = subprocess.PIPE

    def redirect_stream(src, dest):
        try:
            for line in iter(src.readline, b""):
                if not line:
                    break
                if hasattr(dest, "buffer"):
                    dest.buffer.write(line)
                    dest.buffer.flush()
                elif (hasattr(dest, "mode")) and (dest.mode.find("b") != -1):
                    dest.write(line)
                    dest.flush()
                else:
                    dest.write(line.decode())
                    dest.flush()
        except (IOError, OSError):
            pass

    p = subprocess.Popen(args, stdout=p_stdout, stderr=p_stderr, env=env, **kwargs)
    read_stdout_thread = None
    read_stderr_thread = None
    if stdout is not None:
        read_stdout_thread = threading.Thread(
            target=redirect_stream, args=(p.stdout, stdout)
        )
        read_stdout_thread.start()
    if stderr is not None:
        read_stderr_thread = threading.Thread(
            target=redirect_stream, args=(p.stderr, stderr)
        )
        read_stderr_thread.start()
    p.wait()
    if stdout is not None:
        read_stdout_thread.join()
    if stderr is not None:
        read_stderr_thread.join()
    return p
