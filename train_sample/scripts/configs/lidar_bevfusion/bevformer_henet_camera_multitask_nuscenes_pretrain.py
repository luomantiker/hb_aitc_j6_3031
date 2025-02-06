import copy
import os
import re
import shutil
from functools import partial

import cv2
import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

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

task_name = "bevformer_henet_camera_multitask_nuscenes"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"

data_rootdir = (
    "tmp_data/pack_data/pack_data/occ3d_nuscenes/bev_occ_new/v1.0-trainval"
)


meta_rootdir = "./tmp_data/nuscenes/meta"
log_loss_show = 200

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
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

lidar_input = False
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
    bev_h=bev_h_,
    bev_w=bev_w_,
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
        metrci_gt = {}
        metrci_gt["meta"] = batch["seq_meta"][0]["meta"]
        metrics[idx].update(metrci_gt, model_outs[idx])
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
                checkpoint_path="./tmp_pretrained_models/henet_tinym_imagenet/float-checkpoint-best.pth.tar",
                state_dict_update_func=partial(
                    update_state_dict_by_add_prefix, prefix="camera_net."
                ),
                allow_miss=True,
                ignore_extra=True,
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
