"""align bpu validation tools, Only support int-infer."""
import argparse

import horizon_plugin_pytorch as horizon
import torch

from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        required=False,
        default=None,
        help="Reference device, only support single device.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="stage for int_infer or align_bpu.",
    )
    return parser.parse_args()


def _modify_config(cfg, device=None):
    cfg["device"] = device
    return cfg


if __name__ == "__main__":
    args = parse_args()
    if args.device_id is not None:
        torch.cuda.set_device(int(args.device_id))
    config = Config.fromfile(args.config)
    horizon.march.set_march(config.get("march"))
    init_logger(f".hat_logs/{config.task_name}_{args.stage}_validation")

    worker_cfg = config.get(f"{args.stage}_predictor")
    worker_cfg = _modify_config(
        cfg=worker_cfg,
        device=args.device_id,
    )
    predictor = build_from_registry(worker_cfg)

    predictor.fit()
