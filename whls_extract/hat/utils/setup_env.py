import logging
import os

import torch

logger = logging.getLogger(__name__)

__all__ = ["setup_args_env", "setup_hat_env", "set_env_from_conf_file"]


def setup_args_env(args_env):
    args_env_infos = {}
    i = 0
    while i < len(args_env):
        if "=" in args_env[i]:
            args_env_info = args_env[i].split("=")
            args_env_info[0] = args_env_info[0].strip("-")
            args_env_info[0] = args_env_info[0].replace("-", "_").upper()
            args_env_infos.update({args_env_info[0]: args_env_info[1]})
            i += 1
        else:
            args_env_info = args_env[i].strip("-").replace("-", "_").upper()
            args_env_infos.update({args_env_info: args_env[i + 1]})
            i += 2
    for k, v in args_env_infos.items():
        os.environ[k] = v


def set_env_from_conf_file(conf_path: str):
    try:
        with open(conf_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                os.environ[key] = value
                logger.info(f"Set environment variable: {key}={value}")
    except FileNotFoundError:
        logger.warning(f"The file `{conf_path}` does not exist.")
    except Exception as e:
        logger.error(f"Failed to set env from `{conf_path}`: {e}")


def setup_hat_env(
    step,
    pipeline_test=False,
    enable_tracking=False,
    project_id=None,
    log_rank_zero_only=False,
):
    if torch.version.hip:
        os.environ["USE_DCU"] = "1"
    os.environ["HAT_TRAINING_STEP"] = step
    os.environ["HAT_INFERENCE_STEP"] = step
    os.environ["HAT_PIPELINE_TEST"] = str(int(pipeline_test))
    if project_id is not None:
        os.environ["PROJECT_ID"] = project_id
    os.environ["HAT_ENABLE_MODEL_TRACKING"] = str(int(enable_tracking))
    if bool(int(os.environ.get("HAT_USE_CUDAGRAPH", "0"))):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    else:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = os.environ.get(
            "NCCL_ASYNC_ERROR_HANDLING", "1"
        )

    # read rdma setting from nccl config
    set_env_from_conf_file(conf_path="/etc/nccl.conf")

    os.environ["TORCH_NUM_THREADS"] = os.environ.get("TORCH_NUM_THREADS", "12")
    os.environ["OPENCV_NUM_THREADS"] = os.environ.get(
        "OPENCV_NUM_THREADS", "12"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get(
        "OPENBLAS_NUM_THREADS", "12"
    )
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "12")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "12")
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")

    # extend bfloat16
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

    if int(os.environ.get("USE_DCU", 0)) == 1:
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_3"
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["HSA_USERPTR_FOR_PAGED_MEM"] = "0"
        os.environ["HIP_UPSAMPLE_OPTIMIZE"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
        os.environ[
            "UCX_IB_PCI_BW"
        ] = "mlx5_0:50Gbs,mlx5_1:50Gbs,mlx5_2:50Gbs,mlx5_3:50Gbs"
        os.environ["UCX_IB_RANGE_MAX_REGIONS"] = "1000"
        os.environ["NCCL_DEBUG"] = "INFO"
