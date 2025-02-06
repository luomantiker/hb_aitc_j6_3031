import copy
import os
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.quantization.qconfig_template import (  # noqa F401
    calibration_8bit_weight_16bit_act_qconfig_setter,
    default_calibration_qconfig_setter,
    default_qat_fixed_act_qconfig_setter,
    sensitive_op_calibration_8bit_weight_16bit_act_qconfig_setter,
    sensitive_op_qat_8bit_weight_16bit_fixed_act_qconfig_setter,
)
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from hat.data.collates.nusc_collates import collate_nuscenes
from hat.data.datasets.nuscenes_dataset import CLASSES
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bev_sparse_henet_tinym_nuscenes"

num_classes = 10
batch_size_per_gpu = 4
dataloader_workers = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 1 node
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
convert_mode = "jit-strip"
qat_mode = "fuse_bn"

num_query = 900
query_align = 128

orig_shape = (3, 900, 1600)
resize_shape = (3, 396, 704)
data_shape = (3, 256, 704)
val_data_shape = (3, 256, 704)

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
vt_input_hw = (16, 44)

data_rootdir = "./tmp_data/nuscenes/track_dataset/"
meta_rootdir = "./tmp_data/nuscenes/meta"
anchor_file = "./tmp_data/nuscenes/nuscenes_kmeans900.npy"

num_epochs = 100
num_steps_per_epoch = int(28130 // (len(device_ids) * batch_size_per_gpu))
num_steps = num_steps_per_epoch * num_epochs

embed_dims = 256
num_groups = 8
num_levels = 1
num_classes = 10
drop_out = 0.1
num_single_frame_decoder = 0  # 1
num_decoder = 6
num_depth_layers = 3

model = dict(
    type="SparseBEVOE",
    compiler_model=False,
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=[4, 3, 8, 6],
        embed_dims=[64, 128, 192, 384],
        attention_block_num=[0, 0, 0, 0],
        mlp_ratios=[2, 2, 2, 3],
        mlp_ratio_attn=2,
        act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
        use_layer_scale=[True, True, True, True],
        layer_scale_init_value=1e-5,
        num_classes=1000,
        include_top=False,
        extra_act=[False, False, False, False],
        final_expand_channel=0,
        feature_mix_channel=1024,
        block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
        down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
        patch_embed="origin",
    ),
    neck=dict(
        type="MMFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        fix_out_channel=256,
        out_strides=[4, 8, 16, 32],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNetOE",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseBEVOEHead",
        enable_dn=True,
        level_index=[2],
        cls_threshold_to_reg=0.05,
        instance_bank=dict(
            type="MemoryBankOE",
            num_anchor=384,
            embed_dims=embed_dims,
            num_memory_instances=384,
            anchor=anchor_file,
            num_temp_instances=128,
            confidence_decay=0.6,
        ),
        anchor_encoder=dict(
            type="SparseBEVOEEncoder",
            pos_embed_dims=128,
            size_embed_dims=32,
            yaw_embed_dims=32,
            vel_embed_dims=64,
            vel_dims=3,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * num_single_frame_decoder
        + [
            "temp_interaction",
            "interaction",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * (num_decoder - num_single_frame_decoder),
        ffn=dict(
            type="AsymmetricFFNOE",
            in_channels=embed_dims * 2,
            pre_norm=True,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregationOE",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBEVOEKeyPointsGenerator",
                num_pts=8,
            ),
        ),
        refine_layer=dict(
            type="SparseBEVOERefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
        ),
        target=dict(
            type="SparseBEVOETarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                CLASSES.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        cls_allow_reverse=[CLASSES.index("barrier")],
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=num_classes + 1,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(type="L1Loss", loss_weight=0.25),
        loss_cns=dict(type="CrossEntropyLoss", use_sigmoid=True),
        loss_yns=dict(type="GaussianFocalLoss"),
        decoder=dict(type="SparseBEVOEDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

deploy_model = dict(
    type="SparseBEVOE",
    compiler_model=True,
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=[4, 3, 8, 6],
        embed_dims=[64, 128, 192, 384],
        attention_block_num=[0, 0, 0, 0],
        mlp_ratios=[2, 2, 2, 3],
        mlp_ratio_attn=2,
        act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
        use_layer_scale=[True, True, True, True],
        layer_scale_init_value=1e-5,
        num_classes=1000,
        include_top=False,
        extra_act=[False, False, False, False],
        final_expand_channel=0,
        feature_mix_channel=1024,
        block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
        down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
        patch_embed="origin",
    ),
    neck=dict(
        type="MMFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        fix_out_channel=256,
        out_strides=[4, 8, 16, 32],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNetOE",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseBEVOEHead",
        enable_dn=True,
        level_index=[2],
        cls_threshold_to_reg=0.05,
        instance_bank=dict(
            type="MemoryBankOE",
            num_anchor=384,
            embed_dims=embed_dims,
            num_memory_instances=384,
            anchor=anchor_file,
            num_temp_instances=128,
            confidence_decay=0.6,
        ),
        anchor_encoder=dict(
            type="SparseBEVOEEncoder",
            pos_embed_dims=128,
            size_embed_dims=32,
            yaw_embed_dims=32,
            vel_embed_dims=64,
            vel_dims=3,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * num_single_frame_decoder
        + [
            "temp_interaction",
            "interaction",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ]
        * (num_decoder - num_single_frame_decoder),
        ffn=dict(
            type="AsymmetricFFNOE",
            in_channels=embed_dims * 2,
            pre_norm=True,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregationOE",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.0,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBEVOEKeyPointsGenerator",
                num_pts=8,
            ),
        ),
        refine_layer=dict(
            type="SparseBEVOERefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
        ),
    ),
)


def get_deploy_input():
    inputs = {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, 384, 11)),
        "cached_feature": torch.randn((1, 384, 256)),
    }
    return inputs


deploy_inputs = get_deploy_input()


def get_eval_trace_input():
    inputs = {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, 384, 11)),
        "cached_feature": torch.randn((1, 384, 256)),
        "cached_confidence": torch.randn((1, 384)),
        "mask": torch.ones((1)).bool(),
        "timestamp": torch.randn((1)),
        "lidar2global": torch.randn((1, 4, 4)),
        "lidar2img": torch.randn((6, 4, 4)),
    }
    return inputs


eval_trace_inputs = get_eval_trace_input()


def get_train_trace_input():
    inputs = {
        "img": torch.randn((6, 3, 256, 704)),
        "timestamp": torch.randn((1)),
        "lidar2global": torch.randn((1, 4, 4)),
        "lidar2img": torch.randn((6, 4, 4)),
        "lidar_bboxes_labels": torch.randn((1, 20, 10)),
        "instance_ids": torch.randn((1, 20)),
        "camera_intrinsic": torch.randn((6, 3, 3)),
        "points": torch.rand((1, 20000, 3)),
    }
    return inputs


train_trace_inputs = get_train_trace_input()

train_dataset = dict(
    type="NuscenesBevDataset",
    data_path=os.path.join(data_rootdir, "train_lmdb"),
    transforms=[
        dict(type="MultiViewsImgResize", scales=(0.40, 0.47)),
        dict(type="MultiViewsImgCrop", size=(256, 704), random=False),
        dict(type="MultiViewsImgFlip"),
        dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
        dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
        dict(type="MultiViewsPhotoMetricDistortion"),
        dict(
            type="MultiViewsGridMask",
            use_h=True,
            use_w=True,
            rotate=1,
            offset=False,
            ratio=0.5,
            mode=1,
            prob=0.7,
        ),
        dict(
            type="MultiViewsImgTransformWrapper",
            transforms=[
                dict(type="PILToTensor"),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128, std=128),
            ],
        ),
    ],
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    with_lidar_bboxes=True,
    need_lidar=True,
    num_split=2,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    batch_sampler=dict(
        type="DistStreamBatchSampler",
        batch_size=batch_size_per_gpu,
        dataset=train_dataset,
        keep_consistent_seq_aug=True,
        skip_prob=0.0,
        sequence_flip_prob=0.0,
    ),
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

val_dataset = dict(
    type="NuscenesBevDataset",
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    transforms=[
        dict(type="MultiViewsImgResize", size=(396, 704)),
        dict(type="MultiViewsImgCrop", size=(256, 704)),
        dict(
            type="MultiViewsImgTransformWrapper",
            transforms=[
                dict(type="PILToTensor"),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128, std=128),
            ],
        ),
    ],
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    with_lidar_bboxes=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
    batch_size=1,
    shuffle=False,
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
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    grad_scaler=torch.cuda.amp.GradScaler(init_scale=32.0),
    enable_amp=True,
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
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
    batch_size=batch_size_per_gpu,
)

grad_callback = dict(
    type="GradClip",
    max_norm=25,
    norm_type=2,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    interval_by="step",
    save_interval=num_steps_per_epoch * 5,
    strict_match=False,
    mode="max",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=num_steps_per_epoch * 5,
    interval_by="step",
    val_on_train_end=True,
    log_interval=500,
)

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    use_lidar=True,
    trans_lidar_dim=True,
    trans_lidar_rot=False,
    use_ddp=False,
    lidar_key="sensor2ego",
    version="v1.0-trainval",
    save_prefix="./WORKSPACE/results" + task_name,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    "./tmp_pretrained_models/henet_tinym_imagenet/float-checkpoint-best.pth.tar",  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr=3e-4),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=3e-4,
        weight_decay=0.001,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps,
    stop_by="step",
    # num_epochs=100,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage"),
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    sync_bn=True,
    val_metrics=[val_nuscenes_metric],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 12

cali_qconfig_setter = (default_calibration_qconfig_setter,)
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    skip_step=2,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_inputs=eval_trace_inputs,
                qconfig_setter=cali_qconfig_setter,
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=[val_nuscenes_metric],
)

qat_qconfig_setter = (default_qat_fixed_act_qconfig_setter,)
qat_model = copy.deepcopy(model)
qat_model["head"]["enable_dn"] = False
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
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
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=train_trace_inputs,
                state="train",
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        params={
            "backbone": dict(lr=3e-5),
        },
        lr=3e-5,
        weight_decay=0.001,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps * 0.1,
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage", base_steps=50000),
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[int(num_steps * 0.1 * 0.6)],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_nuscenes_metric],
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name + "_model",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    debug=True,
    input_source="pyramid, ddr, ddr, ddr, ddr, ddr",
    opt="O2",
    split_dim=dict(
        inputs={
            "0": [0, 6],
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
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_inputs=eval_trace_inputs,
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=eval_trace_inputs,
                state="val",
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)
first_frame_input = {
    "cached_anchor": torch.zeros((1, 384, 11)),
    "cached_feature": torch.zeros((1, 384, 256)),
    "cached_confidence": torch.zeros((1, 384)),
    "mask": torch.zeros((1)).bool(),
}

hbir_infer_model = dict(
    type="SparseBEVOEIrInfer",
    first_frame_input=first_frame_input,
    projection_mat_key="lidar2img",
    gobel_mat_key="lidar2global",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    decoder=dict(type="SparseBEVOEDecoder"),
    use_memory_bank=True,
    confidence_decay=0.6,
    num_temp_instances=128,
    num_memory_instances=384,
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

infer_transforms = [
    dict(type="MultiViewsImgResize", size=(396, 704)),
    dict(type="MultiViewsImgCrop", size=(256, 704)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
        ],
    ),
]
align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesFromImage",
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=infer_transforms,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        with_lidar_bboxes=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=copy.deepcopy(model),
    device_id=2,
    dataloader=copy.deepcopy(calibration_data_loader),
    num_steps=100,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=calibration_predictor[
        "model_convert_pipeline"
    ],
    analysis_model_type="fake_quant",
)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo


def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]

    return resized_img, orig_shape


def prepare_inputs(infer_inputs):
    dir_list = os.listdir(infer_inputs)
    dir_list.sort()
    input_datas = []
    for _, frame in enumerate(dir_list):
        data = {}
        frame_path = os.path.join(infer_inputs, frame)
        file_list = list(os.listdir(frame_path))
        image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))
        image_dir_list.sort()

        data["imgs"] = [
            os.path.join(frame_path, tmpdir) for tmpdir in image_dir_list
        ]
        timestamp_path = os.path.join(frame_path, "timestamp.npy")
        data["timestamp"] = np.load(timestamp_path).reshape(1)
        lidar2global_path = os.path.join(frame_path, "lidar2global.npy")
        data["lidar2global"] = np.load(lidar2global_path).reshape(1, 4, 4)
        lidar2img_path = os.path.join(frame_path, "lidar2img.npy")
        data["lidar2img"] = np.load(lidar2img_path)
        input_datas.append(data)
    return input_datas


def process_inputs(data, transforms=None):
    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]

    orig_imgs = []
    for i, img_path in enumerate(data["imgs"]):
        img, orig_shape = process_img(img_path, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = data["lidar2img"]

    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))

    model_input = {
        "img": input_imgs,
        "lidar2img": torch.tensor(homo),
        "lidar2global": torch.tensor(data["lidar2global"]),
        "timestamp": torch.tensor(data["timestamp"]),
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["meta"] = {"lidar2img": homo}

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    outs = torch.cat(
        [
            model_outs[0]["bboxes"][..., :9].view(1, -1, 9),
            model_outs[0]["scores"].view(1, -1, 1),
            model_outs[0]["labels"].view(1, -1, 1),
        ],
        dim=-1,
    )
    outs[..., 3], outs[..., 4] = outs[..., 4], outs[..., 3]
    preds = {"lidar_det": outs}
    viz_func(vis_inputs["img"], preds, vis_inputs["meta"])
    return None


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for idx_, sample_data in enumerate(data):
        save_dir = os.path.join(save_path, f"frame{idx_}")
        os.makedirs(save_dir, exist_ok=True)
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            save_name = f"img{image_idx}_{os.path.basename(img_name)}"
            img_data.save(os.path.join(save_dir, save_name), "JPEG")

        lidar2global_path = os.path.join(save_dir, "lidar2global.npy")
        np.save(lidar2global_path, np.array(sample_data["lidar2global"]))
        lidar2img_path = os.path.join(save_dir, "lidar2img.npy")
        np.save(lidar2img_path, np.array(sample_data["lidar2img"]))
        timestamp_path = os.path.join(save_dir, "timestamp.npy")
        np.save(timestamp_path, np.array(sample_data["timestamp"]))


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0, 3],
        inputs_save_func=inputs_save_func,
    ),
    prepare_inputs=prepare_inputs,
    process_inputs=process_inputs,
    viz_func=dict(type="NuscenesViz", is_plot=True),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    inputs=eval_trace_inputs,
    stage="qat",
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=eval_trace_inputs,
                state="val",
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
)

calops_cfg = dict(method="hook")
