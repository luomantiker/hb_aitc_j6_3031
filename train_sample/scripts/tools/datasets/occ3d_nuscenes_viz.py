"""viz Occ3dNuscenesDataset."""

import argparse
import logging
import pickle

from hat.data.datasets.occ3d_nuscenes_dataset import Occ3dNuscenesDataset
from hat.utils.logger import init_logger
from hat.visualize.occ import OccViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=2,
    )
    parser.add_argument(
        "--vis-bev2d",
        action="store_true",
    )
    parser.add_argument(
        "--vis-bev3d",
        action="store_true",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    init_logger(".hat_logs/occ3d_nuscenes_viz", level=logging.WARNING)
    dataset = Occ3dNuscenesDataset(
        data_path=args.data_path,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = OccViz(
        vis_bev_2d=args.vis_bev2d,
        vis_occ_3d=args.vis_bev3d,
    )

    for i, data in enumerate(dataset):

        viz(data["voxel_semantics"].numpy(), save_path=args.save_path)
        if i > int(args.viz_num):
            break
