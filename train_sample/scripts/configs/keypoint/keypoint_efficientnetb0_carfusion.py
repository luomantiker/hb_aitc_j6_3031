import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "keypoint_efficientnetb0_carfusion"

batch_size_per_gpu = 32
device_ids = [0, 1]

ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
convert_mode = "fx"
image_size = (128, 128)
NUM_LDMK = 12

data_root = "./tmp_data/carfusion"
pretrain_model_path = "./tmp_pretrained_models/efficientnet_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
# Train Model
model = dict(
    type="HeatmapKeypointModel",
    backbone=dict(
        type="efficientnet",
        model_type="b0",
        num_classes=1,
        bn_kwargs={},
        activation="relu",
        use_se_block=False,
        include_top=False,
    ),
    decode_head=dict(
        type="DeconvDecoder",
        in_channels=320,
        out_channels=NUM_LDMK,
        input_index=4,
        num_conv_layers=3,
        num_deconv_filters=[128, 128, 128],
        num_deconv_kernels=[4, 4, 4],
        final_conv_kernel=3,
    ),
    loss=dict(type="MSELoss", reduction="mean"),
    post_process=dict(
        type="HeatmapDecoder",
        scale=4,
        mode="averaged",
    ),
)
# Deploy Model
deploy_model = dict(
    type="HeatmapKeypointModel",
    backbone=dict(
        type="efficientnet",
        model_type="b0",
        num_classes=1,
        bn_kwargs={},
        activation="relu",
        use_se_block=False,
        include_top=False,
    ),
    decode_head=dict(
        type="DeconvDecoder",
        in_channels=320,
        out_channels=NUM_LDMK,
        input_index=4,
        num_conv_layers=3,
        num_deconv_filters=[128, 128, 128],
        num_deconv_kernels=[4, 4, 4],
        final_conv_kernel=3,
    ),
    deploy=True,
)


deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))

deploy_model_convert_pipeline = dict(  # noqa: C408
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),  # noqa: C408
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CarfusionPackData",
        data_path=f"{data_root}/train_lmdb",
        transforms=[
            dict(type="RandomFlip", px=0.5),
            dict(
                type="Resize",
                img_scale=image_size,
                keep_ratio=True,
            ),
            dict(
                type="RandomPadLdmkData",
                size=image_size,
            ),
            dict(
                type="AddGaussianNoise",
                prob=0.2,
                mean=0,
                sigma=2,
            ),
            dict(
                type="GenerateHeatmapTarget",
                num_ldmk=NUM_LDMK,
                feat_stride=4,
                heatmap_shape=(32, 32),
                sigma=1.0,
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
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CarfusionPackData",
        data_path=f"{data_root}/test_lmdb",
        transforms=[
            dict(
                type="Resize",
                img_scale=image_size,
                keep_ratio=True,
            ),
            dict(
                type="RandomPadLdmkData",
                size=image_size,
                random=False,
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
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_2d,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_loss(metrics, batch, model_outs):
    preds, losses = model_outs
    for metric in metrics:
        metric.update(losses)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)


def val_update_metric(metrics, batch, model_outs):
    data = {
        "gt_ldmk": batch["gt_ldmk"],
        "pr_ldmk": model_outs[1],
        "gt_ldmk_attr": batch["gt_ldmk_attr"],
    }
    for metric in metrics:
        metric.update(data)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=5000,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
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
    strict_match=True,
    mode="max",
    monitor_metric_key="PCK@0.1",
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
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_model_path,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(type=torch.optim.AdamW, lr=0.001, weight_decay=5e-2),
    batch_processor=batch_processor,
    num_epochs=50,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosLrUpdater",
            warmup_len=0,
            warmup_by="epoch",
            step_log_interval=100,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(
            type="LossShow",
        ),
    ],
    sync_bn=True,
    val_metrics=[
        dict(type="PCKMetric", alpha=0.1, feat_stride=4, img_shape=image_size),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
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
        qconfig_params=dict(
            activation_calibration_observer="percentile",
            activation_calibration_qkwargs=dict(
                percentile=99.975,
            ),
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
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
    val_metrics=[
        dict(type="PCKMetric", alpha=0.1, feat_stride=4, img_shape=image_size),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
    log_interval=calibration_step / 10,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="efficientb0-heatmap-keypoint-model",
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
                checkpoint_path=(
                    os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="PCKMetric",
            alpha=0.1,
            feat_stride=4,
            img_shape=image_size,
        ),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
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
            dict(type="Float2Calibration", convert_mode=convert_mode),
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
    metrics=[
        dict(type="PCKMetric", alpha=0.1, feat_stride=4, img_shape=image_size),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_infer_model = dict(
    type="HeatmapKeypointIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="HeatmapDecoder",
        scale=4,
        mode="averaged",
    ),
)
int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="PCKMetric", alpha=0.1, feat_stride=4, img_shape=image_size),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


infer_transforms = [
    dict(
        type="Resize",
        img_scale=image_size,
        keep_ratio=True,
    ),
    dict(
        type="RandomPadLdmkData",
        size=image_size,
        random=False,
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

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CarfusionPackData",
        data_path=f"{data_root}/test_lmdb",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_2d,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[
        dict(type="PCKMetric", alpha=0.1, feat_stride=4, img_shape=image_size),
        dict(
            type="MeanKeypointDist",
            feat_stride=4,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_img = Image.open(os.path.join(infer_inputs, "img.jpg")).convert("RGB")

    model_input = {
        "img": np.array(ori_img),
        "ori_img": np.array(ori_img),
        "layout": "hwc",
        "color_space": "rgb",
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["layout"] = ["hwc"]
    vis_inputs = {}
    vis_inputs["img"] = np.array(ori_img)
    vis_inputs["scale"] = image_size
    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs[1][0]
    viz_func(vis_inputs["img"], preds, vis_inputs["scale"])
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
    viz_func=dict(type="KeypointsViz", threshold=0.5, is_plot=True),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    stage="qat",
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
)
