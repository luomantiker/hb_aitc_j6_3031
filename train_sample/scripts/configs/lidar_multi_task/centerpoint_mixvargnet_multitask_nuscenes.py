import copy
import os
import shutil
from functools import partial

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.collates import collate_lidar3d
from hat.metrics.mean_iou import MeanIOU
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion
from hat.visualize.lidar_det import lidar_det_visualize

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "centerpoint_mixvargnet_multitask_nuscenes"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
# datadir settings
data_rootdir = "./tmp_data/nuscenes/lidar_seg/v1.0-trainval"
meta_rootdir = "./tmp_data/nuscenes/meta"
gt_data_root = "./tmp_nuscenes/lidar"
log_loss_show = 200

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
norm_cfg = None
bn_kwargs = dict(eps=2e-5, momentum=0.1)
enable_amp = False
convert_mode = "fx"

# Voxelization cfg
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

seg_classes_name = ["others", "driveable_surface"]
det_class_names = [
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
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
common_heads = dict(
    reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
)
with_velocity = "vel" in common_heads.keys()


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


# model settings
net_config = [
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=64,
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=64,
            out_channels=96,
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=96,
            out_channels=160,
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

model = dict(
    type="LidarMultiTask",
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    pre_process=dict(
        type="CenterPointPreProcess",
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
        norm_dims=[0, 1, 2, 3],
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=5,
        num_filters=(64,),
        with_distance=False,
        pool_size=(max_num_points, 1),
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
        hw_reverse=True,
    ),
    scatter=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        disable_quanti_input=True,
        input_channels=64,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[0, 1, 2, 3, 4],
    ),
    neck=dict(
        type="Unet",
        in_strides=(2, 4, 8, 16, 32),
        out_strides=(4,),
        stride2channels=dict(
            {
                2: 64,
                4: 64,
                8: 64,
                16: 96,
                32: 160,
            }
        ),
        out_stride2channels=dict(
            {
                2: 128,
                4: 128,
                8: 128,
                16: 128,
                32: 160,
            }
        ),
        factor=2,
        group_base=8,
        bn_kwargs=bn_kwargs,
    ),
    lidar_decoders=[
        dict(
            type="LidarSegDecoder",
            name="seg",
            task_weight=80.0,
            task_feat_index=0,
            head=dict(
                type="DepthwiseSeparableFCNHead",
                input_index=0,
                in_channels=128,
                feat_channels=64,
                num_classes=2,
                dropout_ratio=0.1,
                num_convs=2,
                bn_kwargs=bn_kwargs,
                int8_output=False,
            ),
            target=dict(
                type="FCNTarget",
            ),
            loss=dict(
                type="CrossEntropyLoss",
                loss_name="seg",
                reduction="mean",
                ignore_index=-1,
                use_sigmoid=False,
                class_weight=[1.0, 10.0],
            ),
            decoder=dict(
                type="FCNDecoder",
                upsample_output_scale=4,
                use_bce=False,
                bg_cls=-1,
            ),
        ),
        dict(
            type="LidarDetDecoder",
            name="det",
            task_weight=1.0,
            task_feat_index=0,
            head=dict(
                type="DepthwiseSeparableCenterPointHead",
                in_channels=128,
                tasks=tasks,
                share_conv_channels=64,
                share_conv_num=1,
                common_heads=common_heads,
                head_conv_channels=64,
                init_bias=-2.19,
                final_kernel=3,
            ),
            target=dict(
                type="CenterPointLidarTarget",
                grid_size=[512, 512, 1],
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                tasks=tasks,
                dense_reg=1,
                max_objs=500,
                gaussian_overlap=0.1,
                min_radius=2,
                out_size_factor=4,
                norm_bbox=True,
                with_velocity=with_velocity,
            ),
            loss=dict(
                type="CenterPointLoss",
                loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
                loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                ),
                with_velocity=with_velocity,
                code_weights=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.2,
                    0.2,
                ],
            ),
            decoder=dict(
                type="CenterPointPostProcess",
                tasks=tasks,
                norm_bbox=True,
                bbox_coder=dict(
                    type="CenterPointBBoxCoder",
                    pc_range=point_cloud_range[:2],
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    max_num=100,
                    score_threshold=0.1,
                    out_size_factor=4,
                    voxel_size=voxel_size[:2],
                ),
                # test_cfg
                max_pool_nms=False,
                score_threshold=0.1,
                post_center_limit_range=[
                    -61.2,
                    -61.2,
                    -10.0,
                    61.2,
                    61.2,
                    10.0,
                ],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                out_size_factor=4,
                nms_type="rotate",
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
                box_size=9,
            ),
        ),
    ],
)

# model settings
deploy_model = dict(
    type="LidarMultiTask",
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    is_deploy=True,
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=5,
        num_filters=(64,),
        with_distance=False,
        pool_size=(max_num_points, 1),
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
        hw_reverse=True,
    ),
    scatter=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        disable_quanti_input=True,
        input_channels=64,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[0, 1, 2, 3, 4],
    ),
    neck=dict(
        type="Unet",
        in_strides=(2, 4, 8, 16, 32),
        out_strides=(4,),
        stride2channels=dict(
            {
                2: 64,
                4: 64,
                8: 64,
                16: 96,
                32: 160,
                64: 160,
            }
        ),
        out_stride2channels=dict(
            {
                2: 128,
                4: 128,
                8: 128,
                16: 128,
                32: 160,
            }
        ),
        factor=2,
        group_base=8,
        bn_kwargs=bn_kwargs,
    ),
    lidar_decoders=[
        dict(
            type="LidarSegDecoder",
            name="seg",
            task_weight=80.0,
            task_feat_index=0,
            head=dict(
                type="DepthwiseSeparableFCNHead",
                input_index=0,
                in_channels=128,
                feat_channels=64,
                num_classes=2,
                dropout_ratio=0.1,
                num_convs=2,
                bn_kwargs=bn_kwargs,
                int8_output=False,
            ),
        ),
        dict(
            type="LidarDetDecoder",
            name="det",
            task_weight=1.0,
            task_feat_index=0,
            head=dict(
                type="DepthwiseSeparableCenterPointHead",
                in_channels=128,
                tasks=tasks,
                share_conv_channels=64,
                share_conv_num=1,
                common_heads=common_heads,
                head_conv_channels=64,
                init_bias=-2.19,
                final_kernel=3,
            ),
        ),
    ],
)

deploy_inputs = dict(
    features=torch.randn((1, 5, 20, 40000), dtype=torch.float32),
    coors=torch.zeros([40000, 4]).int(),
)


db_sampler = dict(
    type="DataBaseSampler",
    enable=True,
    root_path=gt_data_root,
    db_info_path=os.path.join(gt_data_root, "nuscenes_dbinfos_train.pkl"),
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            type="DBFilterByDifficulty",
            filter_by_difficulty=[-1],
        ),
        dict(
            type="DBFilterByMinNumPoint",
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            ),
        ),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

train_dataset = dict(
    type="NuscenesLidarWithSegDataset",
    num_sweeps=9,
    data_path=os.path.join(data_rootdir, "train_lmdb"),
    info_path=os.path.join(gt_data_root, "nuscenes_infos_train.pkl"),
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    pad_empty_sweeps=True,
    remove_close=True,
    use_valid_flag=True,
    classes=det_class_names,
    transforms=[
        dict(
            type="LidarMultiPreprocess",
            class_names=det_class_names,
            global_rot_noise=[-0.3925, 0.3925],
            global_scale_noise=[0.95, 1.05],
            db_sampler=db_sampler,
        ),
        dict(
            type="ObjectRangeFilter",
            point_cloud_range=point_cloud_range,
        ),
        dict(
            type="AssignSegLabel",
            bev_size=[512, 512],
            num_classes=2,
            class_names=[0, 1],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size[:2],
        ),
        dict(type="LidarReformat", with_gt=True),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(type="CBGSDataset", dataset=train_dataset),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_lidar3d,
)

val_dataset = dict(
    type="NuscenesLidarWithSegDataset",
    test_mode=True,
    num_sweeps=9,
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    pad_empty_sweeps=True,
    remove_close=True,
    classes=det_class_names,
    transforms=[
        dict(
            type="AssignSegLabel",
            bev_size=[512, 512],
            num_classes=2,
            class_names=[0, 1],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size[:2],
        ),
        dict(type="LidarReformat", with_gt=True),
    ],
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs[1].items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    meta_key="metadata",
    classes=det_class_names,
)
val_miou_metric = MeanIOU(seg_class=seg_classes_name, ignore_index=-1)


def update_val_metric(metrics, batch, model_outs):
    preds = model_outs[1]["det"]
    metrics[0].update(batch, preds)

    target = batch["gt_seg_labels"]
    preds = model_outs[1]["seg"]
    metrics[1].update(target, preds)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs[1])


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_nuscenes_metric, val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_loss_show,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    val_interval=100,
    log_interval=200,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_loss_show,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_interval=1,
    # mode="max",
    mode=None,
    best_refer_metric=val_nuscenes_metric,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
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
    num_epochs=20,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=200,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
)


calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_step = 100

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
                    ckpt_dir, "float-checkpoint-last.pth.tar"
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
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        weight_decay=0.0,
        lr=1e-4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=200,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr", "ddr"],
    opt="O2",
    output_layout="NHWC",
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
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric, val_miou_metric],
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_infer_model = dict(
    type="LidarMultiTaskIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    pre_process=dict(
        type="CenterPointPreProcess",
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
        norm_dims=[0, 1, 2, 3],
    ),
    lidar_decoders=[
        dict(
            type="FCNDecoder",
            upsample_output_scale=4,
            use_bce=False,
            bg_cls=-1,
        ),
        dict(
            type="CenterPointPostProcess",
            tasks=tasks,
            norm_bbox=True,
            bbox_coder=dict(
                type="CenterPointBBoxCoder",
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=100,
                score_threshold=0.1,
                out_size_factor=4,
                voxel_size=voxel_size[:2],
            ),
            # test_cfg
            max_pool_nms=False,
            score_threshold=0.1,
            post_center_limit_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            out_size_factor=4,
            nms_type="rotate",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            box_size=9,
        ),
    ],
)

pre_process = dict(
    type="CenterPointPreProcess",
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    max_voxels_num=(40000),
    max_points_in_voxel=max_num_points,
    norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
    norm_dims=[0, 1, 2, 3],
)

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)


align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=val_data_loader,
    metrics=[val_nuscenes_metric, val_miou_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms=None):
    points = np.load(os.path.join(infer_inputs, "points.npy")).reshape((-1, 5))

    points = torch.from_numpy(points)
    model_input = {
        "points": [points],
    }

    if transforms is not None:
        model_input = transforms(model_input)

    return model_input, points


def process_outputs(model_outs, viz_func, vis_inputs):
    # preds = model_outs[0]
    preds_det = model_outs[1]["det"][0]
    det_viz_func = viz_func[0]
    det_viz_func(
        points=vis_inputs,
        predictions=preds_det,
    )

    seg_viz_func = viz_func[1]

    preds_seg = model_outs[1]["seg"][0]

    seg_viz_func(None, preds_seg)

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
    viz_func=[
        partial(
            lidar_det_visualize, score_thresh=0.4, is_plot=True, reverse=True
        ),
        dict(
            type="SegViz",
            is_plot=True,
        ),
    ],
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
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
)
