import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "fcos_efficientnetb2_mscoco"
num_classes = 80
batch_size_per_gpu = 8
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
convert_mode = "fx"

model = dict(
    type="FCOS",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b2",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 24, 4: 32, 8: 48, 16: 120, 32: 352}),
        out_channels=112,
        num_outs=5,
        stack=5,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 112, 16: 112, 32: 112, 64: 112, 128: 112}),
        upscale_bbox_pred=False,
        feat_channels=112,
        stacked_convs=4,
        int8_output=False,
        dequant_output=True,
        bbox_relu=False,
    ),
    targets=dict(
        type="DynamicFcosTarget",
        strides=[8, 16, 32, 64, 128],
        cls_out_channels=80,
        background_label=80,
        topK=10,
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=80 + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_reg=dict(
            type="GIoULoss", loss_name="reg", loss_weight=3.0, reduction="none"
        ),
        bbox_relu=True,
    ),
    post_process=dict(
        type="FCOSDecoder",
        num_classes=80,
        strides=[8, 16, 32, 64, 128],
        nms_use_centerness=True,
        nms_sqrt=True,
        transforms=[dict(type="Resize")],
        inverse_transform_key=["scale_factor"],
        test_cfg=dict(
            score_thr=0.05,
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
        ),
        upscale_bbox_pred=True,
        bbox_relu=True,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=80 + 1,
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
    ),
    loss_centerness=dict(
        type="CrossEntropyLoss", loss_name="centerness", use_sigmoid=True
    ),
    loss_reg=dict(
        type="GIoULoss",
        loss_name="reg",
        loss_weight=1.0,
    ),
)

deploy_model = dict(
    type="FCOS",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b2",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 24, 4: 32, 8: 48, 16: 120, 32: 352}),
        out_channels=112,
        num_outs=5,
        stack=5,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 112, 16: 112, 32: 112, 64: 112, 128: 112}),
        upscale_bbox_pred=False,
        feat_channels=112,
        stacked_convs=4,
        int8_output=False,
        nhwc_output=False,
        dequant_output=True,
        bbox_relu=False,
    ),
)

transform_step1 = [
    dict(type="Resize", img_scale=(768, 768), keep_ratio=True),
    dict(
        type="Mosaic",
        image_size=768,
        mixup=True,
        degrees=0.0,
        translate=0.5,
        scale=0.5,
        shear=0,
        perspective=0.0,
    ),
    dict(
        type="RandomFlip",
        px=0.5,
        py=0,
    ),
    dict(type="AugmentHSV", hgain=0.015, sgain=0.7, vgain=0.4),
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
]

transform_step2 = [
    dict(
        type="Resize",
        img_scale=(768, 768),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(768, 768),
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
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="BatchTransformDataset",
        dataset=dict(
            type="Coco",
            data_path="./tmp_data/mscoco/train_lmdb/",
        ),
        transforms_cfgs=[
            transform_step1,
            transform_step2,
        ],
        epoch_steps=[285],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(768, 768),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(768, 768),
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
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(768, 768),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(768, 768),
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
    batch_size=batch_size_per_gpu,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=1,
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
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=5000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

deploy_inputs = dict(img=torch.randn((1, 3, 768, 768)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP",
    save_hash=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=1,
    val_on_train_end=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.07,
        momentum=0.937,
        nesterov=True,
    ),
    batch_processor=batch_processor,
    num_epochs=300,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage"),
        dict(
            type="CosLrUpdater",
            warmup_len=2,
            warmup_by="epoch",
            step_log_interval=100,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    sync_bn=True,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(qat_data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="min_max",
        ),
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
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
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
    data_loader=qat_data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[3, 6],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
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
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
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
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_transforms = [
    dict(
        type="Resize",
        img_scale=(768, 768),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(768, 768),
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
]

hbir_infer_model = dict(
    type="FCOSIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="FCOSDecoder",
        num_classes=80,
        strides=[8, 16, 32, 64, 128],
        nms_use_centerness=True,
        nms_sqrt=True,
        transforms=[dict(type="Resize")],
        inverse_transform_key=["scale_factor"],
        test_cfg=dict(
            score_thr=0.05,
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
        ),
        upscale_bbox_pred=True,
        bbox_relu=True,
    ),
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
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CocoFromImage",
        root="./tmp_orig_data/mscoco/val2017",
        annFile="./tmp_data/mscoco/instances_val2017.json",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_img = Image.open(os.path.join(infer_inputs, "img.jpg"))
    ori_img.convert("RGB")
    image = np.array(ori_img)
    model_input = {
        "img": image,
        "ori_img": image,
        "img_name": [os.path.basename(infer_inputs)],
        "img_id": [0],
        "layout": "hwc",
        "color_space": "rgb",
        "img_shape": image.shape[0:2],
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["ori_img"] = [model_input["ori_img"]]
    model_input["layout"] = [model_input["layout"]]
    model_input["color_space"] = [model_input["color_space"]]
    model_input["img_shape"] = [model_input["img_shape"]]
    model_input["pad_shape"] = [model_input["pad_shape"]]
    return model_input, image


def process_outputs(model_outs, viz_func, vis_inputs):
    dets = model_outs["pred_bboxes"][0]
    viz_func(vis_inputs, dets)
    return None


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    save_name = "img.jpg"
    img_data = Image.fromarray(data["ori_img"], mode="RGB")
    img_data.save(os.path.join(save_path, save_name), "JPEG")


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    transforms=infer_transforms,
    viz_func=dict(type="DetViz", is_plot=True),
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
            ),
        ],
    ),
)
