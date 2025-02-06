import copy
import os
import re
import shutil
from functools import partial

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March
from horizon_plugin_pytorch.quantization.qconfig_template import (
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
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.checkpoint import update_state_dict_by_add_prefix
from hat.utils.config import ConfigVersion
from hat.visualize.lidar_det import lidar_det_visualize

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bevfusion_pointpillar_henet_multisensor_multitask_nuscenes"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"

lidar_task_name = "centerpoint_pointpillar_nuscenes"
lidar_ckpt_dir = "./tmp_pretrained_models/centerpoint_pointpillar_nuscenes"

camera_task_name = "bevformer_henet_camera_multitask_nuscenes"
camera_ckpt_dir = f"./tmp_models/{camera_task_name}"
# datadir settings

data_rootdir = "tmp_data/occ3d_nuscenes/bev_occ_new/v1.0-trainval"


meta_rootdir = "./tmp_data/nuscenes/meta"
log_loss_show = 200

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
norm_cfg = None
bn_kwargs = dict(eps=2e-5, momentum=0.1)
qat_mode = "fuse_bn"
convert_mode = "fx"

# Voxelization cfg
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 0.8)
bev_size_occ = (40, 40)

voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128
input_size = (512, 960)
map_size = (15, 30, 0.15)
use_lidar2img = True
with_ego_occ = False

max_camoverlap_num = 2
bev_sparse_rate = 0.4

lidar_input = True
camera_input = True

use_occ_head = True
use_bev_head = True

class_names = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

num_classes = 10
num_classes_occ = 18

occ3d_seg_class = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


lidar_network = dict(
    type="CenterPointDetector",
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
        up_layer_strides=[0.5, 1, 2],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=norm_cfg,
        quantize=True,
        use_relu6=False,
    ),
)
camera_network = dict(
    type="BevFormer",
    out_indices=(-1,),
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
        type="FPN",
        in_strides=[32],
        in_channels=[384],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="SingleBevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=2,
        embed_dims=_dim_,
        queue_length=1,
        in_indices=(-1,),
        single_bev=True,
        use_lidar2img=use_lidar2img,
        max_camoverlap_num=max_camoverlap_num,
        virtual_bev_h=64,
        virtual_bev_w=80,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="SingleBEVFormerEncoder",
            num_layers=1,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="SingleBEVFormerEncoderLayer",
                embed_dims=_dim_,
                selfattention=dict(
                    type="HorizonMultiScaleDeformableAttention",
                    embed_dims=_dim_,
                    num_points=2,
                    num_levels=1,
                    grid_align_num=4,
                    batch_first=True,
                    # reduce_align_num=8,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    max_camoverlap_num=max_camoverlap_num,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    deformable_attention=dict(
                        type="HorizonMultiScaleDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=2,
                        num_levels=_num_levels_,
                        grid_align_num=64,
                        feats_size=[[30, 16]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
)
bev_head = dict(
    type="BEVFormerDetDecoder",
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=900,
    embed_dims=_dim_,
    pc_range=point_cloud_range,
    decoder=dict(
        type="DetectionTransformerDecoder",
        num_layers=6,
        return_intermediate=True,
        decoder_layer=dict(
            type="DetrTransformerDecoderLayer",
            crossattention=dict(
                type="HorizonMultiScaleDeformableAttention",
                embed_dims=_dim_,
                num_levels=1,
                grid_align_num=4,
                feats_size=[[bev_w_, bev_h_]],
            ),
            dropout=0.1,
        ),
    ),
    criterion=dict(
        type="BevFormerCriterion",
        assigner=dict(
            type="BevFormerHungarianAssigner3D",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction="mean",
        ),
        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.25,
        ),
        pc_range=point_cloud_range,
        bbox_key="lidar_bboxes_labels",
    ),
    post_process=dict(
        type="BevFormerProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        num_classes=10,
    ),
)
occ_head = dict(
    type="BevformerOccDetDecoder",
    use_mask=True,
    lidar_input=lidar_input,
    camera_input=camera_input,
    num_classes=num_classes_occ,
    roi_resizer=dict(
        type="RoiResize",
        in_strides=[1],
        roi_resize_cfgs=[
            dict(
                in_stride=1,
                roi_box=(14, 14, 114, 114),
            )
        ],
    ),
    occ_head=dict(
        type="BEVOCCHead2D",
        in_dim=256,
        out_dim=128,
        Dz=16,
        num_classes=num_classes_occ,
        use_predicter=True,
        use_upsample=True,
    ),
    loss_occ=dict(
        type="CrossEntropyLoss",
        use_sigmoid=False,
        ignore_index=255,
        loss_weight=6.0,
    ),
)
bev_decoders = []
if use_bev_head:
    bev_decoders.append(bev_head)
if use_occ_head:
    bev_decoders.append(occ_head)
# model settings
model = dict(
    type="BevFusion",
    lidar_network=lidar_network if lidar_input else None,
    camera_network=camera_network if camera_input else None,
    bev_decoders=bev_decoders,
    fuse_module=dict(type="BevFuseModule", input_c=384 + _dim_, fuse_c=_dim_),
    bev_h=bev_h_,
    bev_w=bev_w_,
)

# model settings
deploy_model = copy.deepcopy(model)
deploy_model["lidar_network"].pop("pre_process")
deploy_model["camera_network"]["view_transformer"]["is_compile"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")
deploy_model["bev_decoders"][1]["is_compile"] = True
deploy_model["bev_decoders"][1].pop("loss_occ")
deploy_inputs = dict(
    features=torch.randn((1, 5, 20, 40000), dtype=torch.float32),
    coors=torch.zeros([40000, 4]).int(),
    img=torch.randn((6, 3, 512, 960)),
    queries_rebatch_grid=torch.randn((6, 64, 80, 2)),
    restore_bev_grid=torch.randn((1, 256, 128, 2)),
    reference_points_rebatch=torch.randn((6, 5120, 2, 2)),
    bev_pillar_counts=torch.randn((1, 16384, 1)),
)

train_dataset = dict(
    type="NuscenesBevSequenceDataset",
    data_path=os.path.join(data_rootdir, "train_lmdb"),
    map_size=map_size,
    map_path=meta_rootdir,
    with_lidar_bboxes=use_lidar2img,
    with_bev_bboxes=False,
    with_ego_bboxes=True,
    bev_range=point_cloud_range,
    need_lidar=True,
    num_sweeps=9,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    num_seq=1,
    with_ego_occ=False,
    with_lidar_occ=use_lidar2img,
    transforms=[
        dict(type="MultiViewsImgResize", size=input_size),
        dict(type="MultiViewsImgFlip"),
        dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
        dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
        dict(type="MultiViewsPhotoMetricDistortion"),
        dict(
            type="MultiViewsImgTransformWrapper",
            transforms=[
                dict(type="PILToTensor"),
                dict(type="BgrToYuv444", rgb_input=True),
                dict(type="Normalize", mean=128, std=128),
            ],
        ),
    ],
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_nuscenes_sequencev2,
)


val_dataset = dict(
    type="NuscenesBevSequenceDataset",
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    map_size=map_size,
    map_path=meta_rootdir,
    with_lidar_bboxes=True,
    with_bev_bboxes=False,
    with_ego_bboxes=True,
    bev_range=point_cloud_range,
    with_lidar_occ=use_lidar2img,
    need_lidar=True,
    num_sweeps=9,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    num_seq=1,
    transforms=[
        dict(type="MultiViewsImgResize", size=input_size),
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
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    enable_amp=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    idx = 0
    if use_bev_head:
        metric_gt = {}
        metric_gt["meta"] = batch["seq_meta"][0]["meta"]
        metrics[idx].update(metric_gt, model_outs[idx])
        idx += 1
    if use_occ_head:
        # occ metric
        gt_semantics = batch["seq_meta"][0]["gt_occ_info"]["voxel_semantics"][
            0
        ].squeeze()  # (Dx, Dy, Dz)
        semantics_pred = model_outs[idx][0].squeeze()  # (Dx, Dy, Dz)

        if lidar_input and camera_input:
            lidar_mask = batch["seq_meta"][0]["gt_occ_info"]["mask_lidar"][
                0
            ].squeeze()
            camera_mask = batch["seq_meta"][0]["gt_occ_info"]["mask_camera"][
                0
            ].squeeze()
            mask = lidar_mask | camera_mask
        elif lidar_input:
            mask = batch["seq_meta"][0]["gt_occ_info"]["mask_lidar"][
                0
            ].squeeze()  # (Dx, Dy, Dz)
        elif camera_input:
            mask = batch["seq_meta"][0]["gt_occ_info"]["mask_camera"][
                0
            ].squeeze()
        masked_semantics_gt = gt_semantics[mask]
        masked_semantics_pred = semantics_pred[mask]
        results = {
            "label": masked_semantics_gt.reshape(-1),
            "preds": masked_semantics_pred.reshape(-1),
        }
        metrics[idx].update(**results)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_loss_show,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
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
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
    val_interval=1,
    log_interval=200,
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

bev_val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    classes=class_names,
    save_prefix="./WORKSPACE/results" + task_name,
    lidar_key="sensor2ego",
    trans_lidar_dim=True,
    trans_lidar_rot=False,
)
occ_val_nuscenes_metric = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)
val_metrics = []
if use_bev_head:
    val_metrics.append(bev_val_nuscenes_metric)
if use_occ_head:
    val_metrics.append(occ_val_nuscenes_metric)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    camera_ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    lidar_ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=partial(
                    update_state_dict_by_add_prefix, prefix="lidar_net."
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        params={
            "camera_net.backbone": dict(lr_mult=0.1),
        },
        lr=4e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=24,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=val_metrics,
)


calibration_data_loader = copy.deepcopy(val_data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0
calibration_batch_processor = copy.deepcopy(val_batch_processor)

cali_qconfig_setter = (default_calibration_qconfig_setter,)
qat_qconfig_setter = (default_qat_fixed_act_qconfig_setter,)
print("NOT Load sensitive table!")


float2calibration = dict(
    type="Float2Calibration",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=cali_qconfig_setter,
)

float2qat = dict(
    type="Float2QAT",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=qat_qconfig_setter,
)


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
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="RepModel2Deploy",
            ),
            float2calibration,
            dict(
                type="FixWeightQScale",
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=100,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=val_metrics,
    log_interval=20,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="RepModel2Deploy",
            ),
            float2qat,
            dict(
                type="FixWeightQScale",
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
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
        lr=2e-5,
        weight_decay=1e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=16,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[8],
            step_log_interval=500,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=val_metrics,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=[
        "ddr",
    ],
    opt="O2",
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
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
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
            dict(
                type="RepModel2Deploy",
            ),
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
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
            dict(
                type="RepModel2Deploy",
            ),
            float2qat,
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
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_deploy_model = copy.deepcopy(model)

hbir_infer_model = dict(
    type="BevFusionHbirInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    deploy_model=hbir_deploy_model,
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
infer_transforms = [
    dict(type="MultiViewsImgResize", size=(512, 960)),
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
        type="NuscenesFromImageSequence",
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=infer_transforms,
        num_seq=1,
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        with_lidar_bboxes=True,
        bev_range=point_cloud_range,
        need_lidar=True,
        num_sweeps=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        with_lidar_occ=True,
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
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="RepModel2Deploy",
        ),
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)


hbir_exporter = dict(
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    march=march,
    input_names=list(deploy_inputs.keys()),
)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


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


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)

    image_dir_list.sort()
    ego2img = np.load(os.path.join(infer_inputs, "ego2img.npy"))
    lidar2img = np.load(os.path.join(infer_inputs, "lidar2img.npy"))
    frame_inputs = {}
    num_cam = 6

    img_paths = [os.path.join(infer_inputs, p) for p in image_dir_list]
    frame_inputs["img_paths"] = img_paths
    frame_inputs["ego2img"] = ego2img[:num_cam]
    frame_inputs["lidar2img"] = lidar2img[:num_cam]
    frame_inputs["points"] = os.path.join(infer_inputs, "lidar.npy")

    return [frame_inputs]


resize_shape = (3, 512, 960)
val_data_shape = (3, 512, 960)
orig_shape = (3, 900, 1600)


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

    lidar2img = infer_inputs["lidar2img"]
    ego2img = infer_inputs["ego2img"]
    point_path = infer_inputs["points"]
    points = [torch.tensor(np.load(point_path))]

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    lidar2img = resize_homo(lidar2img, scale)
    ego2img = resize_homo(ego2img, scale)
    model_input = {
        "img": input_imgs,
        "points": points,
        "seq_meta": [
            {
                "meta": [
                    {
                        "scene": "test_infer",
                    }
                ],
                "lidar2img": [lidar2img],
                "ego2img": [ego2img],
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["points"] = points
    vis_inputs["meta"] = {
        "ego2img": ego2img,
        "lidar2img": copy.deepcopy(lidar2img),
    }

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    semantics_pred = model_outs[1][0].squeeze().numpy().astype(np.uint8)
    preds = {"bev_det": model_outs[0], "occ_det": semantics_pred}
    viz_func(
        vis_inputs["img"], vis_inputs["points"], preds, vis_inputs["meta"]
    )
    return None


single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["deploy_model"]["bev_decoders"][0]["post_process"][
    "score_threshold"
] = 0.1


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, (img_name, img_data) in enumerate(
        zip(data[0]["img_name"], data[0]["img"])
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")

    ego2img_path = os.path.join(save_path, "ego2img.npy")
    np.save(ego2img_path, np.array(data[0]["ego2img"]))
    np.save(os.path.join(save_path, "lidar2img.npy"), data[0]["lidar2img"])
    np.save(os.path.join(save_path, "lidar.npy"), data[0]["points"])


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
        type="NuscenesMultitaskViz",
        occ_viz=dict(
            type="OccViz",
            vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            vis_bev_2d=True,
        ),
        score_thresh=0.3,
        is_plot=True,
        bev_size=bev_size,
        use_bce=False,
    ),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
)
calops_cfg = dict(method="hook")
onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
)
