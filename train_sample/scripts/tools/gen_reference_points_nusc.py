import argparse
import os

from gen_camera_param_nusc import gen as param_gen
from homography_generator import gen as homo_gen
from reference_points_generator import gen as ref_gen
from reference_points_generator import gen_bevformer as ref_gen_bevformer
from reference_points_generator import init_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        default=None,
        help="sensor to ego translation",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=".",
        help="path to save generate npy",
    )
    parser.add_argument(
        "--save-by-city",
        action="store_true",
    )
    parser.add_argument(
        "--version", default="v1.0-mini", help="The version to packed."
    )
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
    dirs = param_gen(
        args.version, args.data_path, args.save_path, args.save_by_city
    )
    model, cfg = init_model(args.config)
    task_name = cfg.get("task_name", "model")
    for d in dirs:
        s2e_t_path = os.path.join(d, "sensor2ego_translation.npy")
        s2e_r_path = os.path.join(d, "sensor2ego_rotation.npy")
        cam_intrin_path = os.path.join(d, "camera_intrinsic.npy")
        homo_gen(s2e_t_path, s2e_r_path, cam_intrin_path, d)
        homo_path = os.path.join(d, "homography.npy")
        gen_ref_type = cfg.get("gen_ref_type")
        if gen_ref_type == "bevformer":
            ref_gen_bevformer(model, cfg, homo_path, d)
        else:
            ref_gen(model, cfg, homo_path, d)
