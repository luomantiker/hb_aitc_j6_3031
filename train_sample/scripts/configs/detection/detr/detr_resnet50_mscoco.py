import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms import Compose

from hat.data.collates.collates import collate_2d_pad
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "detr_resnet50_mscoco"
num_classes = 80
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
qat_mode = "fuse_bn"
img_shape = (800, 1332)
convert_mode = "fx"

model = dict(
    type="Detr",
    backbone=dict(
        type="ResNet50",
        num_classes=num_classes,
        bn_kwargs={},
        include_top=False,
        stride_change=True,
    ),
    head=dict(
        type="DetrHead",
        transformer=dict(
            type="Transformer",
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            normalize_before=False,
            return_intermediate_dec=True,
        ),
        pos_embed=dict(
            type="PositionEmbeddingSine",
            num_pos_feats=128,
            normalize=True,
        ),
        num_classes=num_classes,
        in_channels=2048,
        max_per_img=100,
        input_shape=img_shape,
    ),
    criterion=dict(
        type="DetrCriterion",
        num_classes=num_classes,
        dec_layers=6,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        loss_ce=1.0,
        loss_bbox=5.0,
        loss_giou=2.0,
        eos_coef=0.1,
        aux_loss=True,
    ),
    post_process=dict(
        type="DetrPostProcess",
    ),
)

deploy_model = dict(
    type="Detr",
    backbone=dict(
        type="ResNet50",
        num_classes=num_classes,
        bn_kwargs={},
        include_top=False,
        stride_change=True,
    ),
    head=dict(
        type="DetrHead",
        transformer=dict(
            type="Transformer",
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            normalize_before=False,
            return_intermediate_dec=False,
        ),
        pos_embed=dict(
            type="PositionEmbeddingSine",
            num_pos_feats=128,
            normalize=True,
        ),
        num_classes=num_classes,
        in_channels=2048,
        max_per_img=100,
        input_shape=img_shape,
    ),
)
deploy_inputs = dict(
    img=torch.randn((1, 3, img_shape[0], img_shape[1])),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomSelectOne",
                transforms=[
                    dict(
                        type="Resize",
                        img_scale=[
                            (480, 1333),
                            (512, 1333),
                            (544, 1333),
                            (576, 1333),
                            (608, 1333),
                            (640, 1333),
                            (672, 1333),
                            (704, 1333),
                            (736, 1333),
                            (768, 1333),
                            (800, 1333),
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                    ),
                    dict(
                        type=Compose,
                        transforms=[
                            dict(
                                type="Resize",
                                img_scale=[
                                    (400, 9999999),
                                    (500, 9999999),
                                    (600, 9999999),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                            ),
                            dict(
                                type="RandomSizeCrop",
                                min_size=384,
                                max_size=600,
                                filter_area=False,
                            ),
                            dict(
                                type="Resize",
                                img_scale=[
                                    (480, 1333),
                                    (512, 1333),
                                    (544, 1333),
                                    (576, 1333),
                                    (608, 1333),
                                    (640, 1333),
                                    (672, 1333),
                                    (704, 1333),
                                    (736, 1333),
                                    (768, 1333),
                                    (800, 1333),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                            ),
                        ],
                    ),
                ],
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
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d_pad,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1333)],
                multiscale_mode="value",
                keep_ratio=False,
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
    collate_fn=collate_2d_pad,
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
    step_log_freq=500,
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
    log_freq=500,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

bn_callback = dict(
    type="FreezeBNStatistics",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=0.1,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=False,
    mode="max",
    monitor_metric_key="mAP",
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
    log_interval=500,
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
                checkpoint_path=(
                    "./tmp_pretrained_models/resnet50_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
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
            "backbone": dict(lr=1e-5),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=0.0001,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=150,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[100],
            step_log_interval=500,
        ),
        bn_callback,
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                verbose=True,
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
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=1e-6,
        weight_decay=4e-5,
    ),
    batch_processor=batch_processor,
    num_epochs=5,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
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

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="detr_test_model",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt="O2",
    transpose_dim=dict(
        outputs={
            "global": [0, 3, 1, 2],
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
                ignore_extra=True,
                verbose=True,
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
        img_scale=[(800, 1333)],
        multiscale_mode="value",
        keep_ratio=False,
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
    type="DetrIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="DetrPostProcess",
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
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
)
