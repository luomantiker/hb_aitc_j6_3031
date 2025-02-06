"""Pack argoverse2."""

import argparse
import os

from hat.data.datasets.argoverse2_dataset import Argoverse2Packer
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Pack argoverse2 dataset.")
    parser.add_argument(
        "-s",
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "-t",
        "--target-data-dir",
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="val", help="The mode to packed."
    )
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/argoverse2_packer")
    target_data_dir = os.path.join(
        args.target_data_dir, f"{args.split_name}_lmdb"
    )
    packer = Argoverse2Packer(
        args.src_data_dir,
        target_data_dir,
        split_name=args.split_name,
        pack_type=args.pack_type,
        num_workers=args.num_workers,
    )
    packer()
