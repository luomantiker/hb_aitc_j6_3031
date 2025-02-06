import importlib
import logging
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Union

try:
    from hatbc.aidi.env import HOSTFILE
    from hatbc.distributed.cli import url2ip
    from hatbc.workflow import Operator
except ImportError:
    HOSTFILE, url2ip, Operator = None, None, None

from hat.utils.apply_func import _as_list
from hat.utils.bucket import url_to_local_path
from hat.utils.package_helper import require_packages

__all__ = [
    "HATOp",
    "HATTrainPipeline",
    "HATPredictor",
]


logger = logging.getLogger(__name__)


class HATOp(Operator):
    @require_packages("hatbc")
    def __init__(
        self,
        entrance: Optional[str],
        enable_tracking: Optional[str] = False,
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__()

        self.entrance = entrance
        self.enable_tracking = enable_tracking
        self.hostfile = hostfile
        self.port = port
        self.launcher = launcher
        self.working_meta = {}
        self.working_env = {}

    def forward(
        self,
        stage: str,
        config_path: str,
        device_ids: Optional[Union[int, List[int]]] = 0,
        num_machines: Optional[int] = 1,
        pipeline_test: Optional[bool] = False,
        working_path: Optional[str] = None,
        working_env: Optional[dict] = None,
        dry_run: Optional[bool] = False,
    ):
        """Hat op.

        Args:
            stage: Stage.
            config_path: Config.
            device_ids: GPU id.
            num_machines: Num of machines.
            pipeline_test: Pipeline test.
            working_path: Add to PYTHONPATH.
            working_env: Env variable.
            dry_run:
                Skip actual cmd run.

        Returns:
            None
        """
        # add working path to pythonpath
        self.update_working_path(working_path=working_path)
        # update entrance with abs path
        self.update_entrance(working_path=working_path)
        # update working_env
        self.update_working_env(working_env)

        config_path = url_to_local_path(config_path)
        device_ids = _as_list(device_ids)

        cmd = [
            "python3",
            "-W",
            "ignore",
        ]
        if not self.entrance.endswith(".py"):
            # execute with pypi cli
            cmd.append("-m")

        cmd.extend(
            [
                self.entrance,
                "-c",
                config_path,
                "-s",
                stage,
                "--device-ids",
                ",".join([str(x) for x in device_ids]),
            ]
        )

        if self.enable_tracking:
            cmd.append("--enable-tracking")

        if pipeline_test:
            cmd.append("--pipeline-test")

        with EnvironContainer(update_environ=self.working_env):
            logger.info(f"HATOp Envs: {self.working_env}")
            # multi machine
            if num_machines > 1:
                cmd += _get_dist_url(self.hostfile, self.port, self.launcher)

                if self.launcher == "mpi":
                    _launch_by_mpi(
                        n=num_machines * len(device_ids),
                        ppn=len(device_ids),
                        command=cmd,
                        hostfile=self.hostfile,
                        dry_run=dry_run,
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported launcher: {self.launcher}"
                    )
            # use torchrun for single machine
            elif self.launcher == "torch":
                pre_cmd = [
                    "torchrun",
                    "--nproc_per_node",
                    "%d" % (len(device_ids)),
                ]
                cmd = pre_cmd + cmd[3:]
                cmd += ["--launcher", "torch"]
                _run_subprocess(command=cmd, dry_run=dry_run)
            else:
                _run_subprocess(command=cmd, dry_run=dry_run)

        self.resume_working_env(working_env)

    def update_working_env(self, working_env):
        if working_env is not None:
            self.old_env = {}
            for k, v in working_env.items():
                if k in self.working_env:
                    self.old_env[k] = working_env[k]
                self.working_env[k] = v

    def resume_working_env(self, working_env):
        if working_env is not None:
            for k in working_env.keys():
                self.working_env.pop(k)
            self.working_env.update(self.old_env)
            self.old_env = {}

    def update_entrance(self, working_path: Optional[str] = None):
        if working_path is not None and self.entrance.endswith(".py"):
            # update entrance path
            if not os.path.exists(self.entrance):
                self.entrance = os.path.join(working_path, self.entrance)
            assert os.path.exists(
                self.entrance
            ), f"Not exists: {self.entrance}"

    def update_working_path(self, working_path: Optional[str] = None):
        if working_path is not None:
            working_path = os.path.abspath(url_to_local_path(working_path))
            assert os.path.exists(working_path), f"Not exists: {working_path}"
            pythonpath = os.environ.get("PYTHONPATH", "")
            if working_path not in pythonpath:
                pythonpath = f"{working_path}:{pythonpath}"
                self.working_env["PYTHONPATH"] = pythonpath
            if working_path not in sys.path:
                sys.path.insert(0, working_path)
                # reload hat modules
                names = []
                for name in list(sys.modules.keys()):
                    if name == "hat" or name.startswith("hat."):
                        sys.modules.pop(name)
                        names.append(name)
                for name in names:
                    try:
                        module = importlib.import_module(name)
                        logger.info(
                            f"Reload Module {name} from {module.__file__}"
                        )
                    except Exception:
                        logger.warning(f"Error reload Module {name}")
                        continue

        return working_path


class HATTrainPipeline(HATOp):
    def __init__(
        self,
        entrance: Optional[str] = "hat.cli.train",
        enable_tracking: Optional[str] = False,
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__(
            entrance=entrance,
            enable_tracking=enable_tracking,
            hostfile=hostfile,
            port=port,
            launcher=launcher,
        )

    def forward(
        self,
        stages: Union[str, List[str]],
        config_path: str,
        device_ids: Optional[Union[int, List[int]]] = 0,
        num_machines: Optional[int] = 1,
        pipeline_test: Optional[bool] = False,
        working_path: Optional[str] = None,
        dry_run: Optional[bool] = False,
    ):
        """Hat train multi-stage op.

        Args:
            stages: List of stage.
            config_path: Config.
            device_ids: GPU id.
            num_machines: Num of machines.
            pipeline_test: Pipeline test.
            working_path: Add to PYTHONPATH.
            dry_run:
                Skip actual cmd run.

        Returns:
            None
        """
        for stage in _as_list(stages):
            super().forward(
                stage=stage,
                config_path=config_path,
                device_ids=device_ids,
                num_machines=num_machines,
                pipeline_test=pipeline_test,
                working_path=working_path,
                dry_run=dry_run,
            )


class HATPredictor(HATOp):
    def __init__(
        self,
        entrance: Optional[str] = "hat.cli.predict",
        enable_tracking: Optional[str] = False,
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__(
            entrance=entrance,
            enable_tracking=enable_tracking,
            hostfile=hostfile,
            port=port,
            launcher=launcher,
        )


def _get_dist_url(hostfile, port, launcher):
    with open(hostfile, "r") as fn:
        dis_url = fn.readlines()[0].strip()

    return [
        "--dist-url",
        f"tcp://{dis_url}:{port}",
        "--launcher",
        launcher,
    ]


def _launch_by_mpi(
    n: int,
    ppn: int,
    command: List[str],
    hostfile: Optional[str] = None,
    dry_run: Optional[bool] = False,
):
    temp_hostfile = tempfile.NamedTemporaryFile().name
    logger.info("temp_hostfile: {temp_hostfile}")
    try:
        mpi_cmd = [
            "mpirun",
            "-n",
            str(n),
            "-ppn",
            str(ppn),
        ]
        if hostfile is not None:
            url2ip(hostfile, temp_hostfile)
            mpi_cmd.extend(
                [
                    "--hostfile",
                    temp_hostfile,
                ]
            )
        _run_subprocess(command=mpi_cmd + command, dry_run=dry_run)
    finally:
        if os.path.exists(temp_hostfile):
            os.remove(temp_hostfile)


def _run_subprocess(
    command: List[str],
    shell: Optional[bool] = False,
    dry_run: Optional[bool] = False,
):
    try:
        command = list(map(str, _as_list(command)))
        logger.info(f"Subprocess command: {' '.join(command)}")
        if dry_run:
            logger.info("Dry_run=True, skip command run!")
        else:
            subprocess.check_call(command, shell=shell)
    except subprocess.CalledProcessError as e:
        logger.fatal(
            f"Subprocess({command}) failed({e.returncode})! {e.output}"
        )
        raise


class ModuleContainer:
    """Clean module added by import_module.

    Parameters
    ----------
    ignore_module_names : Optional[Union[str, List[str]]], optional
        ModuleName will not be cleaned, by default None
    """

    def __init__(
        self,
        ignore_module_names: Optional[Union[str, List[str]]] = None,
    ):
        if ignore_module_names is not None:
            self.ignore_module_names = set(_as_list(ignore_module_names))
        else:
            self.ignore_module_names = set()
        self.old_module_names = set()
        self.new_module_names = set()

    def __enter__(self):
        self.old_module_names = set(sys.modules.keys())

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module_name in list(sys.modules.keys()):
            if (
                module_name not in self.ignore_module_names
                and module_name not in self.old_module_names
            ):
                sys.modules.pop(module_name)


class EnvironContainer:
    def __init__(self, update_environ: Optional[Dict]):
        self.update_envrion = update_environ
        self.old_envrion = {}

    def __enter__(self):
        if self.update_envrion is None:
            return self

        # save old environ
        for k in self.update_envrion.keys():
            old_v = os.environ.get(k, None)
            if old_v is not None:
                self.old_envrion[k] = old_v

        # update environ
        os.environ.update(self.update_envrion)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.update_envrion is None:
            return True

        # delete environ new added
        for k in self.update_envrion.keys():
            del os.environ[k]

        # recover old envrion
        os.environ.update(self.old_envrion)
