import os

import torch
from horizon_plugin_pytorch.march import March

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "fcos3d_efficientnetb0_nuscenes_pretrain"
batch_size_per_gpu = 8
device_ids = [0, 1, 2, 3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
INF = 1e8

data_rootdir = "./tmp_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/nuscenes/meta"

model = dict(
    type="FCOS3D",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
        out_channels=64,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
    ),
    head=dict(
        type="FCOS3DHead",
        num_classes=10,
        in_channels=64,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo
        use_direction_classifier=True,
        pred_attrs=True,
        num_attrs=9,
        cls_branch=(256,),
        reg_branch=(
            (256,),  # offset
            (256,),  # depth
            (256,),  # size
            (256,),  # rot
            (),  # velo
        ),
        dir_branch=(256,),
        attr_branch=(256,),
        centerness_branch=(64,),
        centerness_on_reg=True,
        return_for_compiler=False,
        output_int32=True,
    ),
    targets=dict(
        type="FCOS3DTarget",
        num_classes=10,
        background_label=None,
        bbox_code_size=9,
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, INF)),
        strides=[8, 16, 32, 64, 128],
        pred_attrs=True,
        num_attrs=9,
        center_sampling=True,
        center_sample_radius=1.5,
        centerness_alpha=2.5,
        norm_on_bbox=True,
    ),
    post_process=dict(
        type="FCOS3DPostProcess",
        num_classes=10,
        use_direction_classifier=True,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),
        pred_attrs=True,
        num_attrs=9,
        attr_background_label=9,
        bbox_coder=dict(type="FCOS3DBBoxCoder", code_size=9),
        bbox_code_size=9,
        dir_offset=0.7854,
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=100,
            nms_thr=0.3,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=100,
        ),
    ),
    loss=dict(
        type="FCOS3DLoss",
        num_classes=10,
        pred_attrs=True,
        group_reg_dims=(2, 1, 3, 1, 2),
        num_attrs=9,
        pred_velo=True,
        use_direction_classifier=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        diff_rad_by_sin=True,
        loss_cls=dict(
            type="FocalLoss",
            loss_name=None,
            num_classes=11,
            gamma=2.0,
            alpha=0.25,
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
        loss_attr=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
        loss_centerness=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        train_cfg=dict(
            allowed_border=0,
            code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
            pos_weight=-1,
            debug=False,
        ),
    ),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMonoDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=[
            dict(type="Resize3D", img_scale=(896, 512), keep_ratio=True),
            dict(
                type="Pad",
                size=(512, 896),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMonoDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="Resize3D", img_scale=(896, 512), keep_ratio=True),
            dict(
                type="Pad",
                size=(512, 896),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=50000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="NDS",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_interval=4,
    val_on_train_end=True,
)

# float trainer
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/efficientnet_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.01)},
        lr=2e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=24,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="NuscenesMonoMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
)

# float predictor
float_predictor = dict(
    type="Predictor",
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="NuscenesMonoMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
