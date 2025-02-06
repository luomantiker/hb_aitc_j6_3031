import argparse
import os
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create flashocc data")
    parser.add_argument(
        "-s",
        "--src-data-path",
        type=str,
        required=True,
        help="src data path",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="output path",
    )
    return parser.parse_args()


map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}
classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not os.path.isfile(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def obtain_sensor2top(
    nusc,
    sensor_token,
    l2e_t,
    l2e_r_mat,
    e2g_t,
    e2g_r_mat,
    sensor_type="lidar",
    root_path="./",
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get(
        "calibrated_sensor", sd_rec["calibrated_sensor_token"]
    )
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    root_path = root_path.strip(os.sep)
    data_path = data_path.replace(root_path, "").strip(os.sep)

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def _fill_trainval_infos(
    nusc, train_scenes, val_scenes, test=False, max_sweeps=10, root_path="./"
):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in tqdm(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        if not os.path.isfile(lidar_path):
            raise FileNotFoundError(f"file {lidar_path} does not exist")

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc,
                cam_token,
                l2e_t,
                l2e_r_mat,
                e2g_t,
                e2g_r_mat,
                cam,
                root_path,
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc,
                    sd_rec["prev"],
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get("sample_annotation", token)
                for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array(
                [b.orientation.yaw_pitch_roll[0] for b in boxes]
            ).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = (
                    velo
                    @ np.linalg.inv(e2g_r_mat).T
                    @ np.linalg.inv(l2e_r_mat).T
                )
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in map_name_from_general_to_detection:
                    names[i] = map_name_from_general_to_detection[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]
            )
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]
            )
            info["valid_flag"] = valid_flag

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(
    root_path,
    info_prefix,
    version="v1.0-trainval",
    max_sweeps=10,
    out_dir="./",
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
        out_dir (str, optional): The output dir.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes)
    )
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(  # noqa
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(  # noqa
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in val_scenes
        ]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(
                len(train_scenes), len(val_scenes)
            )
        )
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc,
        train_scenes,
        val_scenes,
        test,
        max_sweeps=max_sweeps,
        root_path=root_path,
    )

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = os.path.join(
            out_dir, "{}_infos_test.pkl".format(info_prefix)
        )
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = os.path.join(
            out_dir, "{}_infos_train.pkl".format(info_prefix)
        )
        with open(info_path, "wb") as f:
            pickle.dump(data, f)
        data["infos"] = val_nusc_infos
        info_val_path = os.path.join(
            out_dir, "{}_infos_val.pkl".format(info_prefix)
        )
        with open(info_val_path, "wb") as f:
            pickle.dump(data, f)


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info["cams"]["CAM_FRONT"]["ego2global_rotation"]
    ego2global_translation = info["cams"]["CAM_FRONT"][
        "ego2global_translation"
    ]
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info["ann_infos"]:
        # Use ego coordinate.
        if (
            map_name_from_general_to_detection[ann_info["category_name"]]
            not in classes
            or ann_info["num_lidar_pts"] + ann_info["num_radar_pts"] <= 0
        ):
            continue
        box = Box(
            ann_info["translation"],
            ann_info["size"],
            Quaternion(ann_info["rotation"]),
            velocity=ann_info["velocity"],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info["category_name"]]
            )
        )
    return gt_boxes, gt_labels


def nuscenes_data_prep(
    root_path, info_prefix, version, max_sweeps=10, out_dir="./"
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    create_nuscenes_infos(
        root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        out_dir=out_dir,
    )


def add_ann_adj_info(extra_tag, dataroot, out_dir):
    nuscenes_version = "v1.0-trainval"
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ["train", "val"]:
        dataset = pickle.load(
            open("%s/%s_infos_%s.pkl" % (out_dir, extra_tag, set), "rb")
        )
        for id in range(len(dataset["infos"])):
            if id % 10 == 0:
                print("%d/%d" % (id, len(dataset["infos"])))
            info = dataset["infos"][id]
            # get sweep adjacent frame info
            sample = nuscenes.get("sample", info["token"])
            ann_infos = list()
            for ann in sample["anns"]:
                ann_info = nuscenes.get("sample_annotation", ann)
                velocity = nuscenes.box_velocity(ann_info["token"])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info["velocity"] = velocity
                ann_infos.append(ann_info)
            dataset["infos"][id]["ann_infos"] = ann_infos
            dataset["infos"][id]["ann_infos"] = get_gt(dataset["infos"][id])
            dataset["infos"][id]["scene_token"] = sample["scene_token"]

            scene = nuscenes.get("scene", sample["scene_token"])
            dataset["infos"][id]["scene_name"] = scene["name"]
            dataset["infos"][id]["occ_path"] = "occ3d/gts/%s/%s" % (
                scene["name"],
                info["token"],
            )
        with open(
            "%s/%s_infos_%s.pkl" % (out_dir, extra_tag, set), "wb"
        ) as fid:
            pickle.dump(dataset, fid)


if __name__ == "__main__":
    args = parse_args()
    dataset = "nuscenes"
    version = "v1.0"
    train_version = f"{version}-trainval"
    extra_tag = "nuscenes"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    nuscenes_data_prep(
        root_path=args.src_data_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0,
        out_dir=args.out_dir,
    )

    print("add_ann_infos")
    add_ann_adj_info(
        extra_tag, dataroot=args.src_data_path, out_dir=args.out_dir
    )
