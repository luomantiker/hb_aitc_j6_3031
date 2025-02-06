import copy
import os
import re
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.quantization.qconfig_template import (  # noqa F401
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

from hat.data.collates.nusc_collates import collate_nuscenes_sequencev2
from hat.data.transforms.functional_img import image_pad
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = False

task_name = "maptroe_henet_tinym_bevformer_nuscenes"
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
log_freq = 200

redirect_config_logging_path = (
    f"/job_log/hat_config_output_{training_step}.log"  # noqa
)
local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [1]
    log_freq = 1
    batch_size_per_gpu = 1
    redirect_config_logging_path = (
        f"./.hat_logs/{task_name}_config_output_{training_step}.log"  # noqa
    )
ckpt_dir = "./tmp_models/%s" % task_name

float_lr = 2.5e-4
qat_lr = 1e-9
enable_amp = False
qat_enable_amp = False
enable_amp_dtype = torch.float16
convert_mode = "jit-strip"
qat_mode = "with_bn"

cudnn_benchmark = True
seed = None
filter_warning = False
log_rank_zero_only = True
march = March.NASH_M
data_rootdir = "./tmp_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/nuscenes/meta"
sd_map_path = "./tmp_data/nuscenes/osm"
map_classes = ["divider", "ped_crossing", "boundary"]

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=[(-1, 32)],
    pv_thickness=1,
)

bn_kwargs = {}
_num_levels_ = 1
hidden_dim = 64
bev_embed_dims = 256
bev_sparse_rate = 0.4
max_camoverlap_num = 2
queue_length = 1  # each sequence contains `queue_length` frames.
test_queue_length = 1
single_bev = True

head_embed_dims = 512
num_vec_one2one = 50
num_vec_one2many = 300
k_one2many = 6
fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

use_lidar_gt = False
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

model = dict(
    type="MapTROE",
    out_indices=(-1,),
    sd_map_fusion=True,
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
        stage_out_norm=True,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[384],
        out_strides=[32],
        out_channels=[bev_embed_dims],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="SingleBevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=bev_embed_dims,
        queue_length=queue_length,
        in_indices=(-1,),
        single_bev=single_bev,
        use_lidar2img=use_lidar2img,
        max_camoverlap_num=max_camoverlap_num,
        virtual_bev_h=int(bev_sparse_rate * bev_h_),
        virtual_bev_w=bev_w_,
        positional_encoding=dict(
            type="PositionEmbeddingLearned",
            num_pos_feats=bev_embed_dims // 2,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="SingleBEVFormerEncoder",
            num_layers=1,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=bev_embed_dims,
            encoder_layer=dict(
                type="SingleBEVFormerEncoderLayer",
                embed_dims=bev_embed_dims,
                selfattention=dict(
                    type="HorizonMultiScaleDeformableAttention",
                    embed_dims=bev_embed_dims,
                    num_levels=1,
                    grid_align_num=10,
                    batch_first=True,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    max_camoverlap_num=max_camoverlap_num,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    deformable_attention=dict(
                        type="HorizonMultiScaleDeformableAttention3D",
                        embed_dims=bev_embed_dims,
                        num_points=8,
                        num_levels=_num_levels_,
                        grid_align_num=20,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=bev_embed_dims,
                ),
                dropout=0.1,
            ),
        ),
    ),
    osm_encoder=dict(
        type="ConvDown",
        in_dim=1,
        mid_dim=hidden_dim // 2,
        out_dim=hidden_dim,
        quant_input=True,
    ),
    bev_fusion=dict(
        type="MapFusion",
        input_dim=bev_embed_dims,
        embed_dims=hidden_dim,
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_down=dict(
            type="ConvDown",
            in_dim=bev_embed_dims,
            mid_dim=bev_embed_dims,
            out_dim=bev_embed_dims,
            quant_input=False,
        ),
        fusion_up=dict(
            type="ConvUp",
            in_dim=2 * hidden_dim,
            mid_dim=bev_embed_dims,
            out_dim=bev_embed_dims,
        ),
    ),
    bev_decoders=[
        dict(
            type="MapInstanceDetectorHead",
            in_channels=bev_embed_dims,
            num_cam=6,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=head_embed_dims,
            num_vec_one2one=num_vec_one2one,
            num_vec_one2many=num_vec_one2many,
            k_one2many=6,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            transform_method="minmax",
            gt_shift_pts_pattern="v2",
            code_size=2,
            num_classes=num_map_classes,
            aux_seg=aux_seg_cfg,
            decoder=dict(
                type="MapInstanceDecoder",
                num_layers=6,
                return_intermediate=True,
                decoder_layer=dict(
                    type="DetrTransformerDecoderLayer",
                    embed_dims=head_embed_dims,
                    crossattention=dict(
                        type="HorizonMultiPointDeformableAttention",
                        embed_dims=head_embed_dims,
                        num_levels=1,
                        grid_align_num=2,
                        num_points=fixed_ptsnum_per_pred_line,
                        feats_size=[[bev_w_, bev_h_]],
                    ),
                    dropout=0.1,
                ),
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
                gt_shift_pts_pattern="v2",
                aux_seg=aux_seg_cfg,
                assigner=dict(
                    type="MapTRAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=4.0),
                    pts_cost=dict(
                        type="OrderedPtsL1Cost", weight=2.5, beta=0.01
                    ),
                    pc_range=point_cloud_range,
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_map_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=4.0,
                    reduction="mean",
                ),
                loss_pts=dict(type="PtsL1Loss", loss_weight=2.5, beta=0.01),
                loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
                loss_seg=dict(
                    type="SimpleLoss", pos_weight=4.0, loss_weight=1.0
                ),
                loss_pv_seg=dict(
                    type="SimpleLoss", pos_weight=1.0, loss_weight=2.0
                ),
            ),
            post_process=dict(
                type="MapTRPostProcess",
                post_center_range=post_center_range,
                pc_range=point_cloud_range,
                max_num=50,
                num_classes=num_map_classes,
            ),
        ),
    ],
)

calib_model = copy.deepcopy(model)
calib_model["bev_decoders"][0]["num_vec"] = num_vec_one2one + num_vec_one2many

test_model = copy.deepcopy(model)
test_model["view_transformer"]["queue_length"] = test_queue_length
test_model["bev_decoders"][0]["queue_length"] = test_queue_length
test_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
)
test_model["bev_decoders"][0].pop("criterion")

deploy_model = copy.deepcopy(model)
deploy_model["view_transformer"]["queue_length"] = 1
deploy_model["view_transformer"]["is_compile"] = True
deploy_model["bev_decoders"][0]["queue_length"] = 1
deploy_model["bev_decoders"][0]["is_deploy"] = True
deploy_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
)
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMapDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_path=meta_rootdir,
        sd_map_path=sd_map_path,
        pc_range=point_cloud_range,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        aux_seg=aux_seg_cfg,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        use_lidar_gt=use_lidar_gt,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(
                        type="TorchVisionAdapter",
                        interface="ColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    ),
                    dict(type="PILToNumpy"),
                    dict(
                        type="GridMask",
                        use_h=True,
                        use_w=True,
                        rotate=1,
                        offset=False,
                        ratio=0.5,
                        mode=1,
                        prob=0.7,
                    ),
                    dict(type="ToTensor", to_yuv=False),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    batch_size=batch_size_per_gpu,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMapDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_path=meta_rootdir,
        sd_map_path=sd_map_path,
        pc_range=point_cloud_range,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=test_queue_length,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        use_lidar_gt=use_lidar_gt,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
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
    val_interval=30,
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
bn_callback = dict(
    type="FreezeBNStatistics",
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
                checkpoint_path=(pretrained_model_dir),
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
            "backbone": dict(lr_mult=0.1),
        },
        lr=float_lr,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=30,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            warmup_lr_begin2ratio=True,
            step_log_interval=500,
            stop_lr=3e-3 * float_lr,
        ),
        # bn_callback,
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

cali_qconfig_setter = (default_calibration_qconfig_setter,)
qat_qconfig_setter = (default_qat_fixed_act_qconfig_setter,)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
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
            type="RepModel2Deploy",
        ),
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

calibration_trainer = dict(
    type="Calibrator",
    model=calib_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                verbose=True,
                check_hash=False,
            ),
            dict(
                type="RepModel2Deploy",
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
        calibration_val_callback,
        calibration_ckpt_callback,
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
                type="RepModel2Deploy",
            ),
            dict(
                type="SyncBnConvert",
            ),
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_data_loader=copy.deepcopy(
                    calibration_example_data_loader
                ),
                qconfig_setter=qat_qconfig_setter,
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
        params={
            "backbone": dict(lr_mult=0.1),
        },
        lr=qat_lr,
        weight_decay=0.1,
    ),
    batch_processor=qat_batch_processor,
    device=None,
    num_epochs=3,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        qat_val_callback,
        qat_ckpt_callback,
    ],
    sync_bn=False,
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
                type="RepModel2Deploy",
            ),
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
                type="RepModel2Deploy",
            ),
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


deploy_inputs = dict(
    img=torch.randn((6, 3, 480, 800)),
    osm_mask=torch.randn((1, 1, bev_h_, bev_w_)),
    queries_rebatch_grid=torch.randn(
        (6, int(bev_sparse_rate * bev_h_), bev_w_, 2)
    ),
    restore_bev_grid=torch.randn((1, max_camoverlap_num * bev_h_, bev_w_, 2)),
    reference_points_rebatch=torch.randn(
        (6, int(bev_sparse_rate * bev_h_ * bev_w_), 4, 2)
    ),
    bev_pillar_counts=torch.randn((1, bev_h_ * bev_w_, 1)),
)

deploy_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="RepModel2Deploy",
        ),
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
    input_source=["pyramid", "ddr", "ddr"],
    opt="O2",
    split_dim=dict(
        inputs={
            "0": [0, 6],
        }
    ),
    debug=True,
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
    type="MapTROEIrInfer",
    sd_map_fusion=True,
    test_model=copy.deepcopy(test_model),
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False
int_infer_data_loader["num_workers"] = 0

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

# -------------------------- align bpu predictor config --------------------------
infer_transforms = [
    dict(type="MultiViewsImgResize", size=(450, 800)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="Pad", divisor=32),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMapFromImageSequence",
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=infer_transforms,
        with_bev_bboxes=False,
        with_ego_bboxes=False,
        with_bev_mask=False,
        num_seq=1,
        map_path=meta_rootdir,
        sd_map_path=sd_map_path,
        map_classes=map_classes,
        pc_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        padding_value=-10000,
        use_lidar_gt=use_lidar_gt,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
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

single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    ego2imgs = []
    lidar2imgs = []
    osm_masks = []
    for seq_idx, sample_data in enumerate(data):
        img_idx_start = seq_idx * len(sample_data["img_name"])
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            image_idx_save = image_idx + img_idx_start
            save_name = f"img{image_idx_save}_{os.path.basename(img_name)}"
            img_data.save(os.path.join(save_path, save_name), "JPEG")
        ego2imgs.append(sample_data["ego2img"])
        lidar2imgs.append(sample_data["ego2global"])
        osm_masks.append(sample_data["osm_mask"].numpy())
    ego2img_path = os.path.join(save_path, "ego2img.npy")
    ego2imgs_np = np.array(ego2imgs)
    ego2imgs_np = ego2imgs_np.reshape(
        -1, ego2imgs_np.shape[-2], ego2imgs_np.shape[-1]
    )
    np.save(ego2img_path, ego2imgs_np)

    lidar2imgs_path = os.path.join(save_path, "lidar2img.npy")
    lidar2imgs_np = np.array(lidar2imgs)[None]
    np.save(lidar2imgs_path, lidar2imgs_np)

    osm_mask_path = os.path.join(save_path, "osm_mask.npy")
    osm_masks_np = np.array(osm_masks)
    np.save(osm_mask_path, osm_masks_np)


resize_shape = (3, 450, 800)
val_data_shape = (3, 480, 800)
orig_shape = (3, 900, 1600)


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)
    ego2img = np.load(os.path.join(infer_inputs, "ego2img.npy"))
    lidar2img = np.load(os.path.join(infer_inputs, "lidar2img.npy"))
    osm_mask = np.load(os.path.join(infer_inputs, "osm_mask.npy"))
    frames_inputs = []
    num_cam = 6
    num_frame = len(image_dir_list) // num_cam
    for i in range(num_frame):
        frame_inputs = {}
        img_paths = image_dir_list[i * num_cam : (i + 1) * num_cam]

        img_paths = [
            os.path.join(infer_inputs, img_path) for img_path in img_paths
        ]
        frame_inputs["img_paths"] = img_paths
        frame_inputs["ego2img"] = ego2img[i * num_cam : (i + 1) * num_cam]
        frame_inputs["lidar2img"] = lidar2img[i * num_cam : (i + 1) * num_cam]
        frame_inputs["osm_mask"] = osm_mask[i]
        frames_inputs.append(frame_inputs)
    frames_inputs.reverse()
    return frames_inputs


def process_img(img_path, resize_size, pad_divisor):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw)
    resized_img = image_pad(
        resized_img, "chw", None, pad_divisor, 0
    ).unsqueeze(0)
    return resized_img, orig_shape


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def process_inputs(infer_inputs, transforms=None):
    pad_divisor = 32
    resize_size = resize_shape[1:]
    orig_imgs = []
    for i, img_path in enumerate(infer_inputs["img_paths"]):
        img, orig_shape = process_img(img_path, resize_size, pad_divisor)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    ego2img = infer_inputs["ego2img"]
    lidar2img = infer_inputs["lidar2img"]
    osm_mask = infer_inputs["osm_mask"]
    osm_mask = torch.tensor(osm_mask).unsqueeze(0)

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    ego2img = resize_homo(ego2img, scale)
    lidar2img = resize_homo(lidar2img, scale)

    model_input = {
        "img": input_imgs,
        "osm_mask": osm_mask,
        "seq_meta": [
            {
                "meta": [
                    {
                        "scene": "test_infer",
                    }
                ],
                "ego2img": [ego2img],
                "lidar2img": [lidar2img],
            }
        ],
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
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesMapViz",
        is_plot=True,
        pc_range=point_cloud_range,
        car_img_path=car_img_path,
        use_lidar=use_lidar_gt,
    ),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
)
gen_ref_type = "bevformer"
