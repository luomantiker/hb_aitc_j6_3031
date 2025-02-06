"""viz argoverse2."""

import argparse
import logging

from hat.data.datasets.argoverse2_dataset import Argoverse2PackedDataset
from hat.utils.logger import init_logger
from hat.visualize.argoverse2 import Argoverse2Viz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=5000,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".hat_logs/argoverse2_viz", level=logging.WARNING)
    dataset = Argoverse2PackedDataset(
        data_path=args.data_path,
        input_dim=2,
    )
    viz = Argoverse2Viz(
        num_historical_steps=50,
        is_plot=args.plot,
    )

    for i, data in enumerate(dataset):
        viz(data, None, None, save_path=f"tmp_vis/ar2_{i}")
        if i > int(args.viz_num):
            break
