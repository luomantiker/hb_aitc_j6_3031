"""viz nuscenes map."""

import argparse
import logging
import os
import os.path as osp
import pickle

from torchvision.transforms import functional as F

from hat.data.datasets.nuscenes_map_dataset import NuscenesMapDataset
from hat.utils.logger import init_logger
from hat.visualize.nuscenes_map import NuscenesMapViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--meta-path",
        required=True,
    )
    parser.add_argument(
        "--sd-map-path",
        default=None,
    )
    parser.add_argument(
        "--viz-num",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    parser.add_argument(
        "--use-lidar",
        action="store_true",
    )
    parser.add_argument("--fixed-num", default=20, type=int)
    parser.add_argument("--save-dir", default=None, type=str)
    parser.add_argument("--car-path", default=None)

    args = parser.parse_args()

    init_logger(".hat_logs/nuscenes_map_viz", level=logging.WARNING)
    if args.use_lidar:
        pc_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
        bev_size = [100, 50]
    else:
        pc_range = [-30.0, -15.0, -10.0, 30.0, 15.0, 10.0]
        bev_size = [50, 100]
    map_classes = ["divider", "ped_crossing", "boundary"]
    dataset = NuscenesMapDataset(
        data_path=args.data_path,
        map_path=args.meta_path,
        sd_map_path=args.sd_map_path,
        pc_range=pc_range,
        bev_size=bev_size,
        map_classes=map_classes,
        test_mode=True,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        padding_value=-10000,
        queue_length=1,
        use_lidar_gt=args.use_lidar,
        fixed_ptsnum_per_line=args.fixed_num,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = NuscenesMapViz(
        pc_range=pc_range,
        is_plot=args.plot,
        car_img_path=args.car_path,
        use_lidar=args.use_lidar,
    )

    for i, data in enumerate(dataset):
        data = data[0]
        imgs = []
        for img, name in zip(data["img"], data["cam_name"]):
            img = F.pil_to_tensor(img)
            imgs.append({"name": name, "img": img})

        img_name = data["img_name"][0]
        img_name = (
            img_name.split("/")[-1].split(".")[0].replace("_FRONT_LEFT", "")
        )
        if args.save_dir is not None:
            save_dir = osp.join(args.save_dir, img_name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None
        viz(imgs=imgs, gts=data, save_path=save_dir)
        if i >= args.viz_num - 1:
            break
