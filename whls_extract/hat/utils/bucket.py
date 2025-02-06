"""The client to manipulate files in bucket.

Copied from `hatbc` repo and make modifications,
retaining only the URL path conversion functions.
originally from `hatbc.filestream.bucket.client`.
"""
import logging
import os
import subprocess
import sys
import threading
from typing import Callable, Dict, Optional, Tuple

from hat.utils.aidi import is_running_on_aidi

__all__ = [
    "BucketClient",
    "get_bucket_client",
    "url_to_local_path",
    "local_path_to_url",
]


logger = logging.getLogger(__name__)

DMP_PREFIX = "dmpv2://"
HTTPS_PREFIX = "https://"
GFPS_PREFIX = "/gpfs/"
HTTPS_PATH = "https://us-data.hobot.cc/"


B_PER_MB = 2 ** 20


class KThread(threading.Thread):
    """A subclass of threading.Thread, with a kill() method.

    Come from:

    Kill a thread in Python:

    http://mail.python.org/pipermail/python-list/2004-May/260937.html
    """

    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)

        self.killed = False

    def start(self):
        """Start the thread."""

        self.__run_backup = self.run

        self.run = self.__run  # Force the Thread to install our trace.

        threading.Thread.start(self)

    def __run(self):
        """Hacked run function, which installs the trace."""

        sys.settrace(self.globaltrace)

        self.__run_backup()

        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == "call":
            return self.localtrace

        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == "line":
                raise SystemExit()

        return self.localtrace

    def kill(self):
        self.killed = True


def timeout_call(
    seconds: float,
    func: Callable,
    args=tuple(),  # noqa C408
    kwargs: Dict = None,
    message: str = None,
):  # noqa D400
    """Call a function and raise a TimeoutError when executing time
        exceed timeout duration.

    Args:
        seconds: In seconds.
        func : Function to exec.
        args : Function args, by default tuple().
        kwargs : Function kwargs, by default None.
        message : TimeoutError message, by default None.
    """

    if message is None:
        message = f"Calling {func} take time exceed {seconds}s"

    if kwargs is None:
        kwargs = dict()  # noqa C408

    result = []
    exception = None

    def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
        try:
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))
        except BaseException as e:
            nonlocal exception
            exception = e

    # create new args for _new_func, because we want to get the func return val to result list  # noqa
    new_kwargs = {
        "oldfunc": func,
        "result": result,
        "oldfunc_args": args,
        "oldfunc_kwargs": kwargs,
    }

    thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)

    thd.start()

    thd.join(seconds)

    alive = thd.is_alive()

    thd.kill()  # kill the child thread

    if alive:
        raise TimeoutError(message)
    elif exception is not None:
        raise exception
    else:
        return result[0]


def _parse_mount_root(msg, check_access=True):
    mount_keys = [
        "hogpu.cc",
        "JuiceFS",
        "gpfs/plat_dmp",
        "gpfs2/plat_dmp",
        "gpfs-test/plat_dmp",
        "gpfs",
    ]

    bucket2root = dict()  # noqa C408

    for result_i in msg.strip().split("\n"):
        parts = result_i.strip().split()
        if len(parts) >= 6:
            device = parts[0]
            mount_point = parts[2]
            for k in mount_keys:
                if k in device:
                    bucket_name = mount_point.split("/")[-1]
                    if check_access:
                        if os.access(mount_point, os.R_OK):
                            bucket2root[bucket_name] = mount_point
                    else:
                        bucket2root[bucket_name] = mount_point

    return bucket2root


def get_bucket_mount_root() -> Dict[str, str]:
    if is_running_on_aidi():
        bucket2root = dict()  # noqa C408
        prefix = "bucket_"
        for key_i, val_i in os.environ.items():
            if key_i.startswith(prefix):
                bucket2root[key_i[len(prefix) :]] = val_i
        return bucket2root

    else:
        results = subprocess.run(["mount"], capture_output=True, text=True)
        results = results.stdout
        return _parse_mount_root(results)


def is_dmp_url(url: str) -> bool:
    return url.startswith(DMP_PREFIX)


def is_https_url(url: str) -> bool:
    return url.startswith(HTTPS_PREFIX)


def check_is_dmp_url(url: str):
    if not is_dmp_url(url):
        raise Exception(f"{url} is not a dmp url")


def split_dmp_url(url: str) -> Tuple[str, str]:
    check_is_dmp_url(url)
    dmp_path = url[len(DMP_PREFIX) :]
    bucket_name_pos = dmp_path.find("/")
    bucket_name = dmp_path[:bucket_name_pos]
    bucket_file_path = dmp_path[bucket_name_pos:]
    return bucket_name, bucket_file_path


_client = None


def get_bucket_client():
    global _client
    if _client is not None:
        return _client
    _client = BucketClient()
    return _client


class NotValidLocalPathError(FileNotFoundError):
    """This error is raised when the input path is not a valid local path."""


class BucketClient(object):
    def __init__(
        self,
        max_retry=5,
        check_mounted_buckets_visiable=True,
    ):
        self._max_retry = max_retry
        self.bucket2root = get_bucket_mount_root()

        self.check_mounted_buckets_visiable = check_mounted_buckets_visiable
        self.checked_buckets = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["bucket2root"] = None
        state["checked_buckets"] = set()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        # NOTE: reset this value because the local and remote value
        # is different
        self.bucket2root = get_bucket_mount_root()

    def _check_bucket(self, bucket_name):  # noqa D400
        """
        Sometimes, bucket may be blocked, for the first time of visiting,
        check whether can be visited or not.
        """
        if bucket_name in self.checked_buckets:
            return
        assert (
            bucket_name in self.bucket2root
        ), f"{bucket_name} does not mount!"  # noqa
        bucket_root = self.bucket2root[bucket_name]
        try:
            timeout_call(120, os.listdir, (bucket_root,))
        except TimeoutError:
            raise OSError(
                f"Cannot visit the mounted bucket {bucket_name}, exceed 120 seconds"  # noqa
            )  # noqa
        self.checked_buckets.add(bucket_name)

    def valid_url(self, url: str) -> bool:
        return url.startswith(DMP_PREFIX) or url.startswith(HTTPS_PREFIX)

    def mount_buckets(self):
        return list(self.bucket2root.keys())

    def get_mount_root(self, bucket_name) -> str:
        if bucket_name not in self.bucket2root:
            raise FileNotFoundError(f"bucket not mounted: {bucket_name}")
        if self.check_mounted_buckets_visiable:
            self._check_bucket(bucket_name)
        return self.bucket2root[bucket_name]

    def url_to_local(
        self,
        url: str,
        root_only: Optional[bool] = False,
        check_visiable: Optional[bool] = True,
    ) -> str:
        if is_https_url(url):
            local_root = url.replace(HTTPS_PATH, GFPS_PREFIX)
            return local_root
        else:
            bucket_name, bucket_file_path = split_dmp_url(url)
            if check_visiable:
                local_root = self.get_mount_root(bucket_name)
            else:
                local_root = "/horizon-bucket/{}".format(bucket_name)
            return local_root if root_only else local_root + bucket_file_path

    def local_to_url(self, path: str) -> str:
        path = os.path.abspath(path)
        if path.startswith(GFPS_PREFIX):
            return path.replace(GFPS_PREFIX, HTTPS_PATH)
        for bucket_name in self.bucket2root:
            local_root = self.bucket2root[bucket_name]
            if path.startswith(local_root):
                path = f"{DMP_PREFIX}{bucket_name}{path[len(local_root):]}"
                return path
        else:
            raise NotValidLocalPathError(
                "Input path not valid local bucket path: %s" % path
            )

    def isfile(self, url: str) -> bool:
        if not self.exists(url):
            return False
        file_info = self.file_info(url)
        return file_info is not None and file_info.folder is False


def url_to_local_path(
    path: str,
    bucket_client: "BucketClient" = None,
    *,
    allow_invalid_bucket_url: bool = True,
):
    """Change dmp url path to local path.

    Args:
        path: dmp url path
        bucket_client: bucket client, default None
        allow_invalid_bucket_url: whether allow invalid bucket url, default True # noqa

    Returns:
        path: local path.
    """
    # get bucket client
    bucket_client = get_bucket_client()
    # judge the validity of input path
    if bucket_client.valid_url(path):
        return bucket_client.url_to_local(path)
    elif allow_invalid_bucket_url:
        return path
    else:
        raise ValueError(f"Invalid dmp url: {path}")


def local_path_to_url(
    path: str,
    bucket_client: "BucketClient" = None,
    *,
    allow_invalid: bool = False,
):
    """Change local path to url path.

    Args:
        path: local path
        bucket_client: bucket client, default None
        allow_invalid: whether allow invalid local path, default False # noqa

    Returns:
        path: dmp url path.
    """
    # get bucket client
    bucket_client = get_bucket_client()
    # judge the validity of input path
    if not bucket_client.valid_url(path):
        try:
            path = bucket_client.local_to_url(path)
        except FileNotFoundError:
            if allow_invalid:
                return path
            else:
                raise ValueError(f"Invalid local url: {path}")
    return path
