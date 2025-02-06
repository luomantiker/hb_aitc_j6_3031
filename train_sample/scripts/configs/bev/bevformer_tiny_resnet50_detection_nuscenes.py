import copy
import os
import re
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
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

enable_model_tracking = True

task_name = "bevformer_tiny_resnet50_detection_nuscenes"
num_classes = 1000
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
lossshow = 50
convert_mode = "fx"
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
data_rootdir = "./tmp_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/nuscenes/meta"
map_size = (15, 30, 0.15)
CLASSES = (
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
bn_kwargs = {}
_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3  # each sequence contains `queue_length` frames.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 2.048)
use_bce = False
num_classes = 10
model = dict(
    type="BevFormer",
    out_indices=(-1,),
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[2048],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="BevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=_dim_,
        queue_length=3,
        in_indices=(-1,),
        max_camoverlap_num=2,
        virtual_bev_h=20,
        virtual_bev_w=32,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="BEVFormerEncoder",
            num_layers=3,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="BEVFormerEncoderLayer",
                selfattention=dict(
                    type="HorizonTemporalSelfAttention",
                    embed_dims=_dim_,
                    num_levels=1,
                    grid_align_num=100,
                    reduce_align_num=8,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    max_camoverlap_num=2,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    deformable_attention=dict(
                        type="HorizonMultiScaleDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=8,
                        num_levels=_num_levels_,
                        grid_align_num=20,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
    bev_decoders=[
        dict(
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
                        batch_first=False,
                        grid_align_num=10,
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
            ),
            post_process=dict(
                type="BevFormerProcess",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10,
            ),
        ),
    ],
)
test_model = copy.deepcopy(model)

test_model["view_transformer"]["queue_length"] = 1

deploy_model = copy.deepcopy(model)

deploy_model["view_transformer"]["queue_length"] = 1
deploy_model["view_transformer"]["is_compile"] = True
# deploy_model["is_compile"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        bev_range=point_cloud_range,
        num_seq=3,
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
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        bev_range=point_cloud_range,
        num_seq=1,
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
    sampler=None,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


def loss_collector(outputs: dict):
    losses = []
    for output in outputs:
        for _, loss in output.items():
            losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=True,
    enable_amp_dtype=torch.float16,
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
    step_log_freq=lossshow,
    epoch_log_freq=1,
    log_prefix="",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to inde
    preds = model_outs
    metrci_gt = {}
    metrci_gt["meta"] = batch["seq_meta"][0]["meta"]
    metrics[0].update(metrci_gt, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    # metrics=[val_nuscenes_metric],
    metric_update_func=update_val_metric,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_callback = dict(
    type="Validation",
    val_interval=4,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=test_model,
    val_on_train_end=True,
    init_with_train_model=True,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=4,
    strict_match=True,
    mode="max",
    # best_refer_metric=val_nuscenes_metric,
    monitor_metric_key="NDS",
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
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/resnet50_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr_mult=0.1),
        },
        lr=4e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=24,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False
calibration_step = 10
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="RepModel2Deploy",
        ),
        dict(type="Float2Calibration", convert_mode=convert_mode),
    ],
)
calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

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
                ignore_extra=True,
                verbose=True,
                allow_miss=True,
            ),
            dict(
                type="RepModel2Deploy",
            ),
            dict(type="Float2Calibration", convert_mode=convert_mode),
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
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    log_interval=calibration_step / 10,
)
qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1
qat_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="RepModel2Deploy",
        ),
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)
qat_ckpt_callback = copy.deepcopy(ckpt_callback)
qat_ckpt_callback["save_interval"] = 1
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="RepModel2Deploy",
            ),
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="FixWeightQScale",
            ),
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
        type=torch.optim.AdamW,
        lr=2e-5,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=10,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[5],
            step_log_interval=500,
        ),
        qat_val_callback,
        qat_ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
)

int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
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
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="RepModel2Deploy",
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

calibration_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="RepModel2Deploy",
            ),
            dict(type="Float2Calibration", convert_mode=convert_mode),
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
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


qat_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="RepModel2Deploy",
            ),
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
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
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
hbir_deploy_model = copy.deepcopy(test_model)

hbir_infer_model = dict(
    type="BevFormerIrInfer",
    deploy_model=hbir_deploy_model,
    model_convert_pipeline=qat_predictor["model_convert_pipeline"],
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
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
        bev_range=point_cloud_range,
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
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

deploy_inputs = dict(
    img=torch.randn((6, 3, 480, 800)),
    prev_bev=torch.randn((1, 2500, 256)),
    prev_bev_ref=torch.randn((1, 50, 50, 2)),
    queries_rebatch_grid=torch.randn((6, 20, 32, 2)),
    restore_bev_grid=torch.randn((1, 100, 50, 2)),
    reference_points_rebatch=torch.randn((6, 640, 4, 2)),
    bev_pillar_counts=torch.randn((1, 2500, 1)),
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
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
    image_dir_list = list(filter(lambda x: x.endswith(".png"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)

    homo = np.load(os.path.join(infer_inputs, "ego2img.npy"))
    temporal_homo = np.load(os.path.join(infer_inputs, "ego2global.npy"))
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
        frame_inputs["homo"] = homo[i * num_cam : (i + 1) * num_cam]
        frame_inputs["temporal_homo"] = temporal_homo[:, i : i + 1]
        frames_inputs.append(frame_inputs)
    frames_inputs.reverse()
    return frames_inputs


resize_shape = (3, 450, 800)
val_data_shape = (3, 480, 800)
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

    homo = infer_inputs["homo"]

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)

    temporal_homo = infer_inputs["temporal_homo"][0]

    model_input = {
        "img": input_imgs,
        "seq_meta": [
            {
                "meta": [
                    {
                        "scene": "test_infer",
                    }
                ],
                "ego2img": [homo],
                "ego2global": temporal_homo,
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["meta"] = {"ego2img": homo}

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = {"bev_det": model_outs}
    viz_func(vis_inputs["img"], preds, vis_inputs["meta"])
    return None


single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["deploy_model"]["bev_decoders"][0]["post_process"][
    "score_threshold"
] = 0.1

single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None
single_infer_dataset["num_seq"] = 3


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    ego2imgs = []
    ego2globals = []
    for seq_idx, sample_data in enumerate(data):
        img_idx_start = seq_idx * len(sample_data["img_name"])
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            image_idx_save = image_idx + img_idx_start
            save_name = (
                f"img{image_idx_save}_{os.path.basename(img_name)}".replace(
                    ".jpg", ".png"
                )
            )
            img_data.save(os.path.join(save_path, save_name), "PNG")
        ego2imgs.append(sample_data["ego2img"])
        ego2globals.append(sample_data["ego2global"])
    ego2img_path = os.path.join(save_path, "ego2img.npy")
    ego2imgs_np = np.array(ego2imgs)
    ego2imgs_np = ego2imgs_np.reshape(
        -1, ego2imgs_np.shape[-2], ego2imgs_np.shape[-1]
    )
    np.save(ego2img_path, ego2imgs_np)

    ego2global_path = os.path.join(save_path, "ego2global.npy")
    ego2imgs_np = np.array(ego2globals)[None]
    np.save(ego2global_path, ego2imgs_np)


infer_cfg = dict(
    model=single_hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[2],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesViz", is_plot=True, bev_size=bev_size, use_bce=use_bce
    ),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid", "ddr", "ddr", "ddr"],
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
    model_convert_pipeline=qat_predictor["model_convert_pipeline"],
)

gen_ref_type = "bevformer"
