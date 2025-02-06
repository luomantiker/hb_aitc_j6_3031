import argparse
import logging
import os

import horizon_plugin_pytorch as horizon

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(args.config)

    if "quant_analysis_solver" not in cfg:
        raise RuntimeError("Please set quant_analysis_solver in config file.")
    quant_analysis_solver = cfg.quant_analysis_solver

    logger.info("=" * 50 + "BEGIN QUANT ANALYSIS" + "=" * 50)

    if "march" not in cfg:
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    if (
        "type" not in quant_analysis_solver
        or quant_analysis_solver["type"] not in "QuantAnalysis"
    ):
        raise ValueError(
            "quant_analysis_solver must have 'type=QuantAnalysis'."
        )

    if quant_analysis_solver.get("out_dir", None) is None:
        save_path = "."
        if "ckpt_dir" in cfg:
            save_path = cfg.ckpt_dir
        else:
            logger.warning(
                "No ckpt_dir found in config file. Analysis results will be "
                + "saved in ./quant_analysis"
            )
        save_path = os.path.join(save_path, "quant_analysis")
        os.makedirs(save_path, exist_ok=True)
        quant_analysis_solver["out_dir"] = save_path

    with RegistryContext():
        quant_analysis = build_from_registry(quant_analysis_solver)
        quant_analysis()

    logger.info("=" * 50 + "END QUANT ANALYSIS" + "=" * 50)
