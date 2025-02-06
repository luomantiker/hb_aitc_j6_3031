import copy
import os
import shutil
from functools import partial

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.collates import collate_lidar3d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion
from hat.visualize.lidar_det import lidar_det_visualize

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "pointpillars_kitti_car"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
base_data_dir = "./tmp_data/kitti3d/"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
convert_mode = "fx"

# PointPillars settings
norm_cfg = None

# Voxelization cfg
pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4.0]
max_points_in_voxel = 100
max_voxels_num = 12000

class_names = ["Car"]


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


# model settings
model = dict(
    type="PointPillarsDetector",
    feature_map_shape=get_feature_map_size(pc_range, voxel_size),
    pre_process=dict(
        type="PointPillarsPreProcess",
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        pool_size=(1, max_points_in_voxel),
        voxel_size=voxel_size,
        pc_range=pc_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
    ),
    backbone=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=64,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[64, 128, 256],
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=norm_cfg,
        quantize=True,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([128, 128, 128]),
        use_direction_classifier=True,
    ),
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    targets=dict(
        type="LidarTargetAssigner",
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        class_names=class_names,
        positive_fraction=-1,
    ),
    loss=dict(
        type="PointPillarsLoss",
        num_classes=len(class_names),
        loss_cls=dict(
            type="FocalLossV2",
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            reduction="none",
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="SmoothL1Loss",
            beta=1 / 9.0,
            reduction="none",
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            reduction="none",
            loss_weight=0.2,
        ),
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.4,
        post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
        max_per_img=100,
    ),
)

# model settings
deploy_model = dict(
    type="PointPillarsDetector",
    feature_map_shape=get_feature_map_size(pc_range, voxel_size),
    is_deploy=True,
    pre_process=dict(
        type="PointPillarsPreProcess",
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        pool_size=(1, max_points_in_voxel),
        voxel_size=voxel_size,
        pc_range=pc_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
    ),
    backbone=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=64,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[64, 128, 256],
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=norm_cfg,
        quantize=True,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([128, 128, 128]),
        use_direction_classifier=True,
    ),
)

deploy_inputs = dict(
    points=[
        torch.randn(150000, 4),
    ],
)

db_sampler = dict(
    type="DataBaseSampler",
    enable=True,
    root_path="./tmp_data/kitti3d/",
    db_info_path="./tmp_data/kitti3d/kitti3d_dbinfos_train.pkl",
    sample_groups=[dict(Car=15)],
    db_prep_steps=[
        dict(
            type="DBFilterByDifficulty",
            filter_by_difficulty=[-1],
        ),
        dict(
            type="DBFilterByMinNumPoint",
            filter_by_min_num_points=dict(
                Car=5,
            ),
        ),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path="./tmp_data/kitti3d/train_lmdb",
        transforms=[
            dict(
                type="ObjectSample",
                class_names=class_names,
                remove_points_after_sample=False,
                db_sampler=db_sampler,
            ),
            dict(
                type="ObjectNoise",
                gt_rotation_noise=[-0.15707963267, 0.15707963267],
                gt_loc_noise_std=[0.25, 0.25, 0.25],
                global_random_rot_range=[0, 0],
                num_try=100,
            ),
            dict(
                type="PointRandomFlip",
                probability=0.5,
            ),
            dict(
                type="PointGlobalRotation",
                rotation=[-0.78539816, 0.78539816],
            ),
            dict(
                type="PointGlobalScaling",
                min_scale=0.95,
                max_scale=1.05,
            ),
            dict(
                type="ShufflePoints",
                shuffle=True,
            ),
            dict(
                type="ObjectRangeFilter",
                point_cloud_range=pc_range,
            ),
            dict(type="LidarReformat"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path="./tmp_data/kitti3d/val_lmdb",
        transforms=[
            dict(type="LidarReformat"),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(0),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    preds = model_outs
    for metric in metrics:
        metric.update(preds, batch)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix=f"Validation {task_name}",
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix=f"loss_{task_name}",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=f"{training_step}-",
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP_3D_moderate",
)


val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    val_interval=1,
    log_interval=200,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        lr=2e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=160,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=50,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 50

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2Calibration", convert_mode=convert_mode),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1,
            ),
        ),
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=0.0)},
        lr=2e-4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=50,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=50,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
    opt="O2",
    transpose_dim=dict(
        outputs={
            "global": [0, 2, 3, 1],
        }
    ),
)


# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)


calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                verbose=True,
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_infer_model = dict(
    type="PointPillarsDetectorIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.4,
        post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
        max_per_img=100,
    ),
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


align_bpu_predictor = copy.deepcopy(int_infer_predictor)
align_bpu_predictor["log_interval"] = 1
align_bpu_predictor["data_loader"] = align_bpu_predictor["data_loader"][0]
align_bpu_predictor["batch_processor"] = dict(
    type="BasicBatchProcessor", need_grad_update=False
)


def process_inputs(infer_inputs, transforms=None):
    points = np.load(os.path.join(infer_inputs, "points.npy")).reshape((-1, 4))

    points = torch.from_numpy(points)
    model_input = {
        "points": [points],
    }

    if transforms is not None:
        model_input = transforms(model_input)

    return model_input, points


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs[0]
    viz_func(vis_inputs, preds)
    return None


single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    points_path = os.path.join(save_path, "points.npy")
    np.save(points_path, data["lidar"]["points"])


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=partial(lidar_det_visualize, score_thresh=0.4, is_plot=True),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
)
