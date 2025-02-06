import copy
import os
import re
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.quantization.fake_quantize import FakeQuantize
from horizon_plugin_pytorch.quantization.observer_v2 import FixedScaleObserver
from horizon_plugin_pytorch.quantization.qconfig import QConfig
from horizon_plugin_pytorch.quantization.qconfig_template import (  # noqa F401
    ModuleNameQconfigSetter,
    default_calibration_qconfig_setter,
    default_qat_fixed_act_qconfig_setter,
)
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from hat.data.collates.nusc_collates import collate_nuscenes
from hat.data.transforms.functional_img import image_pad
from hat.models.base_modules.attention import MultiheadAttention
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = False

task_name = "maptroe_sparse_henet_tinym_nuscenes"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
num_machines = 2
log_freq = 200
ckpt_dir = "./tmp_models/%s" % task_name
redirect_config_logging_path = (
    f"/job_log/hat_config_output_{training_step}.log"  # noqa
)
local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [1]
    log_freq = 1
    batch_size_per_gpu = 4
    redirect_config_logging_path = (
        f"./.hat_logs/{task_name}_config_output_{training_step}.log"  # noqa
    )

float_lr = 3e-4
float_backbone_lr = 1e-3
qat_lr = 1e-5
num_epochs = 100
num_steps_per_epoch = int(
    28130 // (len(device_ids) * batch_size_per_gpu * num_machines)
)
num_steps = num_steps_per_epoch * num_epochs
enable_amp = True
qat_enable_amp = False
enable_amp_dtype = torch.float16
convert_mode = "jit-strip"
qat_mode = "with_bn"
cudnn_benchmark = True
seed = None
filter_warning = False
log_rank_zero_only = True
march = March.NASH_M

data_rootdir = "./tmp_data/nuscenes/track_dataset/"
meta_rootdir = "./tmp_data/nuscenes/meta"
anchor_file = "./tmp_orig_data/nuscenes/kmeans_map_100.npy"
map_classes = ["divider", "ped_crossing", "boundary"]
fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
num_anchor = 100
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=False,
    pv_seg=True,
    dense_depth=True,
    seg_classes=1,
    feat_down_sample=[(1, 8), (2, 16), (3, 32)],
    pv_thickness=1,
)

bn_kwargs = {}
_dim_ = 256
num_groups = 8
drop_out = 0.1
num_single_frame_decoder = 1
num_decoder = 6
feat_indices = [2]  # strides = [4, 8, 16, 32]
num_levels = len(feat_indices)
num_depth_layers = 3

use_lidar_gt = True
use_lidar2img = use_lidar_gt
if use_lidar_gt:
    point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
    bev_h_ = 100
    bev_w_ = 50
    post_center_range = [-20, -35, -20, -35, 20, 35, 20, 35]
else:
    point_cloud_range = [-30.0, -15.0, -10.0, 30.0, 15.0, 10.0]
    bev_h_ = 50
    bev_w_ = 100
    post_center_range = [-35, -20, -35, -20, 35, 20, 35, 20]

# henet-tinym
depth = [4, 3, 8, 6]
block_cls = ["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"]
width = [64, 128, 192, 384]
attention_block_num = [0, 0, 0, 0]
mlp_ratios, mlp_ratio_attn = [2, 2, 2, 3], 2
act_layer = ["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"]
use_layer_scale = [True, True, True, True]
extra_act = [False, False, False, False]
final_expand_channel, feature_mix_channel = 0, 1024
down_cls = ["S2DDown", "S2DDown", "S2DDown", "None"]
patch_embed = "origin"
pretrained_model_dir = "./tmp_pretrained_models/henet_tinym_imagenet/float-checkpoint-best.pth.tar"

head = dict(
    type="SparseOMOEHead",
    feat_indices=feat_indices,
    num_anchor=num_anchor,
    num_pts_per_vec=fixed_ptsnum_per_pred_line,
    num_views=6,
    projection_mat_key="lidar2img",
    anchor_encoder=dict(
        type="SparseOEPoint3DEncoder",
        embed_dims=_dim_,
        input_dim=fixed_ptsnum_per_pred_line * 2,
    ),
    instance_bank=dict(
        type="InstanceBankOE",
        num_anchor=num_anchor,
        embed_dims=_dim_,
        anchor=anchor_file,
        num_temp_instances=0,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
    ),
    instance_interaction=dict(
        type=MultiheadAttention,
        embed_dim=_dim_,
        num_heads=num_groups,
        batch_first=True,
        attn_drop=drop_out,
        proj_drop=drop_out,
    ),
    norm_layer=dict(
        type=torch.nn.LayerNorm,
        normalized_shape=_dim_,
    ),
    ffn=dict(
        type="AsymmetricFFNOE",
        in_channels=_dim_ * 2,
        pre_norm=True,
        embed_dims=_dim_,
        num_fcs=2,
        ffn_drop=0.1,
        feedforward_channels=_dim_ * 4,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregationOEv2",
        embed_dims=_dim_,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        residual_mode="cat",
        use_camera_embed=True,
        grid_align_num=4,
        kps_generator=dict(
            type="SparsePoint3DKeyPointsGenerator",
            embed_dims=_dim_,
            num_sample=fixed_ptsnum_per_pred_line,
            num_learnable_pts=2,
            fix_height=(0, 0.5, -0.5),
            ground_height=-1.84023,  # ground height in lidar frame
        ),
    ),
    refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=_dim_,
        num_sample=fixed_ptsnum_per_pred_line,
        coords_dim=2,
        num_cls=num_map_classes,
        with_cls_branch=True,
    ),
    num_decoder=num_decoder,
    num_single_frame_decoder=num_single_frame_decoder,
    operation_order=[
        "interaction",
        "norm",
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * num_single_frame_decoder
    + [
        "interaction",
        "norm",
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * (num_decoder - num_single_frame_decoder),
)
model = dict(
    type="MapTROE",
    out_indices=(1, 2, 3, 4),
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=depth,
        embed_dims=width,
        attention_block_num=attention_block_num,
        mlp_ratios=mlp_ratios,
        mlp_ratio_attn=mlp_ratio_attn,
        act_layer=act_layer,
        use_layer_scale=use_layer_scale,
        layer_scale_init_value=1e-5,
        num_classes=1000,
        include_top=False,
        extra_act=extra_act,
        final_expand_channel=final_expand_channel,
        feature_mix_channel=feature_mix_channel,
        block_cls=block_cls,
        down_cls=down_cls,
        patch_embed=patch_embed,
        stage_out_norm=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[4, 8, 16, 32],
        in_channels=[64, 128, 192, 384],
        out_strides=[4, 8, 16, 32],
        out_channels=[_dim_, _dim_, _dim_, _dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    bev_decoders=[
        dict(
            type="SparseMapPerceptionDecoder",
            embed_dims=_dim_,
            num_cam=6,
            num_vec_one2one=num_anchor,
            num_vec_one2many=0,
            k_one2many=0,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            transform_method="minmax",
            is_deploy=False,
            decoder=head,
            aux_seg=aux_seg_cfg,
            depth_branch=dict(  # for auxiliary supervision only
                type="DenseDepthNetOE",
                embed_dims=_dim_,
                num_depth_layers=num_depth_layers,
                loss_weight=0.2,
            ),
            criterion=dict(
                type="MapTRCriterion",
                dir_interval=1,
                num_classes=num_map_classes,
                code_weights=[1.0, 1.0, 1.0, 1.0],
                sync_cls_avg_factor=True,
                pc_range=point_cloud_range,
                num_pts_per_vec=fixed_ptsnum_per_pred_line,  # one bbox
                num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
                gt_shift_pts_pattern="sparsedrive",
                aux_seg=aux_seg_cfg,
                pred_absolute_points=True,
                assigner=dict(
                    type="MapTRAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=1.0),
                    pts_cost=dict(
                        type="OrderedPtsL1Cost", weight=0.5, beta=0.01
                    ),
                    pc_range=point_cloud_range,
                    pred_absolute_points=True,
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_map_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    reduction="mean",
                ),
                loss_pts=dict(type="PtsL1Loss", loss_weight=0.5, beta=0.01),
                loss_dir=dict(type="PtsDirCosLoss", loss_weight=0),
                loss_pv_seg=dict(
                    type="SimpleLoss", pos_weight=1.0, loss_weight=2.0
                ),
            ),
            post_process=dict(
                type="MapTRPostProcess",
                pc_range=point_cloud_range,
                post_center_range=post_center_range,
                max_num=num_anchor,
                num_classes=num_map_classes,
                pred_absolute_points=True,
            ),
        ),
    ],
)

test_model = copy.deepcopy(model)
test_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    dense_depth=False,
)
test_model["bev_decoders"][0].pop("criterion")

deploy_model = copy.deepcopy(model)
deploy_model["bev_decoders"][0]["is_deploy"] = True
deploy_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    dense_depth=False,
)
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesSparseMapDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_path=meta_rootdir,
        pc_range=point_cloud_range,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        padding_value=-10000,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        need_lidar=True,
        use_lidar_gt=use_lidar_gt,
        transforms=[
            dict(type="MultiViewsImgResize", scales=(0.40, 0.47)),
            dict(type="MultiViewsImgCrop", size=(256, 704), random=False),
            dict(type="MultiViewsImgFlip"),
            dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
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
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesSparseMapDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_path=meta_rootdir,
        pc_range=point_cloud_range,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        padding_value=-10000,
        map_classes=map_classes,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        use_lidar_gt=use_lidar_gt,
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
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)


def loss_collector(outputs: dict):
    losses = []
    for outs in outputs:
        for _, loss in outs.items():
            losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
    enable_amp_dtype=enable_amp_dtype,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss(metrics, batch, model_outs):
    for model_out in model_outs:
        for metric in metrics:
            metric.update(model_out)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)


def update_val_metric(metrics, batch, model_outs):
    preds = model_outs
    for metric in metrics:
        metric.update(batch, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


val_callback = dict(
    type="Validation",
    val_interval=20,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=test_model,
    val_on_train_end=True,
    init_with_train_model=True,
    log_interval=200,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="MAP",
)
grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)


val_map_metric = dict(
    type="NuscenesMapMetric",
    classes=map_classes,
    save_prefix="./WORKSPACE/results" + task_name,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    pc_range=point_cloud_range,
    metric="chamfer",
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrained_model_dir,
                allow_miss=True,
                ignore_extra=True,
                verbose=False,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr=float_backbone_lr),
        },
        lr=float_lr,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    device=None,
    num_steps=num_steps,
    stop_by="step",
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=log_freq,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        val_map_metric,
    ],
)


QINT16_MAX = 32767.5
fixed_int16_qconfig = dict()
for layer in range(2, 36, 6):
    fixed_int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.point_cat"
    ] = QConfig(
        output=FakeQuantize.with_args(
            observer=FixedScaleObserver,
            dtype=qint16,
            scale=60 / QINT16_MAX,
        )
    )
    fixed_int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.reciprocal_op"
    ] = QConfig(
        output=FakeQuantize.with_args(
            observer=FixedScaleObserver,
            dtype=qint16,
            scale=10 / QINT16_MAX,
        )
    )
    fixed_int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.point_sum"
    ] = QConfig(
        output=FakeQuantize.with_args(
            observer=FixedScaleObserver,
            dtype=qint16,
            scale=60 / QINT16_MAX,
        )
    )
    fixed_int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.point_mul"
    ] = QConfig(
        output=FakeQuantize.with_args(
            observer=FixedScaleObserver,
            dtype=qint16,
            scale=1.1 / QINT16_MAX,
        )
    )

cali_qconfig_setter = (
    ModuleNameQconfigSetter(fixed_int16_qconfig),
    default_calibration_qconfig_setter,
)
qat_qconfig_setter = (
    ModuleNameQconfigSetter(fixed_int16_qconfig),
    default_qat_fixed_act_qconfig_setter,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False
val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)
calibration_step = 50
calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1
pre_step_float_ckpt = os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pre_step_float_ckpt,
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
                check_hash=False,
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_data_loader=calibration_example_data_loader,
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(
                type="FixWeightQScale",
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_ckpt_callback,
        calibration_val_callback,
    ],
    val_metrics=[
        val_map_metric,
    ],
    log_interval=calibration_step / 10,
)

qat_val_callback = copy.deepcopy(calibration_val_callback)
qat_ckpt_callback = copy.deepcopy(ckpt_callback)
qat_ckpt_callback["save_interval"] = 1
qat_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=qat_enable_amp,
    enable_amp_dtype=enable_amp_dtype,
)
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_data_loader=copy.deepcopy(
                    calibration_example_data_loader
                ),
                qconfig_setter=qat_qconfig_setter,
                state="train",
            ),
            dict(
                type="FixWeightQScale",
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
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
        lr=qat_lr,
        weight_decay=0.001,
    ),
    batch_processor=qat_batch_processor,
    device=None,
    num_epochs=10,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        qat_val_callback,
        qat_ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        val_map_metric,
    ],
)


float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        val_map_metric,
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

calibration_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_data_loader=copy.deepcopy(val_example_data_loader),
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        val_map_metric,
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_data_loader=copy.deepcopy(val_example_data_loader),
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                verbose=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        val_map_metric,
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)


deploy_inputs = {
    "img": torch.randn((6, 3, 256, 704)),
    "projection_mat": torch.randn((6, 4, 4)),
}
deploy_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
            check_hash=False,
        ),
    ],
)
hbir_exporter = dict(  # noqa: C408
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=deploy_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    input_names=list(deploy_inputs.keys()),
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    debug=True,
    input_source=["pyramid", "ddr"],
    opt="O2",
    split_dim=dict(
        inputs={
            "0": [0, 6],
        }
    ),
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=deploy_convert_pipeline,
)

calops_cfg = dict(method="hook")
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

# -------------------------- int validation config --------------------------
hbir_infer_model = dict(
    type="SparseMapIrInfer",
    test_model=copy.deepcopy(test_model),
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    projection_mat_key="lidar2img",
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["num_workers"] = 0
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        val_map_metric,
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=10,
)
align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        val_map_metric,
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


# -------------------------- model infer config --------

single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["test_model"]["bev_decoders"][0]["post_process"][
    "score_threshold"
] = 0.2

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
        lidar2img_path = os.path.join(save_dir, "lidar2img.npy")
        np.save(lidar2img_path, np.array(sample_data["lidar2img"]))


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
        lidar2img_path = os.path.join(frame_path, "lidar2img.npy")
        data["lidar2img"] = np.load(lidar2img_path)
        input_datas.append(data)
    return input_datas


resize_shape = (3, 396, 704)
val_data_shape = (3, 256, 704)


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
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    viz_func(vis_inputs["img"], preds=preds)
    return None


car_img_path = (
    "tmp_orig_data/nuscenes/infer_imgs/single_frame_lidarego/cars/lidar_car.png"
    if use_lidar_gt
    else "tmp_orig_data/nuscenes/infer_imgs/single_frame_lidarego/cars/car.png"
)
infer_cfg = dict(
    model=single_hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0, 1],
        inputs_save_func=inputs_save_func,
    ),
    prepare_inputs=prepare_inputs,
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesMapViz",
        is_plot=True,
        pc_range=point_cloud_range,
        car_img_path=car_img_path,
        use_lidar=use_lidar_gt,
    ),
    process_outputs=process_outputs,
)
