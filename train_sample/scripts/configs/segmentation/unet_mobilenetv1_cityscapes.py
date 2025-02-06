import copy
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.march import March
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from hat.data.collates.collates import collate_2d
from hat.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS
from hat.metrics.mean_iou import MeanIOU
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "unet_mobilenetv1_cityscapes"
num_classes = 19
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
dataloader_workers = batch_size_per_gpu  # per gpu
pretrained_model_dir = "./tmp_pretrained_models/%s" % task_name
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
enable_amp = False

data_shape = (3, 1024, 2048)
train_scales = (4, 8, 16, 32, 64)
alpha = 0.25
bn_kwargs = dict(eps=2e-5, momentum=0.1)

output_with_bn = True

weight_decay = 5e-3
start_lr = 0.01
train_epochs = 300

qat_lr = 2e-4
qat_train_epochs = 30

data_rootdir = os.path.join(".", "tmp_data", "cityscapes")
tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")

# model
model = dict(
    type="Segmentor",
    backbone=dict(
        type="MobileNetV1",
        num_classes=-1,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        dw_with_relu=True,
        include_top=False,
        flat_output=False,
    ),
    neck=dict(
        type="DwUnet",
        base_channels=int(32 * alpha),
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU,
        use_deconv=False,
        dw_with_act=True,
        output_scales=train_scales,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=train_scales,
        out_strides=train_scales,
        stride2channels={
            stride: int(stride * 32 * alpha) for stride in train_scales
        },
        feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
        stacked_convs=0,
        argmax_output=False,
        dequant_output=True,
        int8_output=True,
        upscale=False,
        output_with_bn=output_with_bn,
        bn_kwargs=bn_kwargs,
    ),
    losses=dict(
        type="SoftmaxFocalLoss",
        loss_name="Focal",
        num_classes=num_classes,
        weight=tuple(np.array((256, 128, 64, 32, 16)) / 19),
        reduction="mean",
    ),
)

deploy_model = dict(
    type="Segmentor",
    backbone=dict(
        type="MobileNetV1",
        num_classes=-1,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        dw_with_relu=True,
        include_top=False,
        flat_output=False,
    ),
    neck=dict(
        type="DwUnet",
        base_channels=int(32 * alpha),
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU,
        use_deconv=False,
        dw_with_act=True,
        output_scales=train_scales,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=train_scales,
        out_strides=train_scales,
        stride2channels={
            stride: int(stride * 32 * alpha) for stride in train_scales
        },
        feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
        stacked_convs=0,
        argmax_output=True,
        dequant_output=False,
        int8_output=True,
        upscale=False,
        output_with_bn=output_with_bn,
        bn_kwargs=bn_kwargs,
        only_export_first=True,
    ),
    losses=None,
)

deploy_inputs = {"img": torch.randn((1,) + data_shape)}

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
    ],
)


# data
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)


def loss_collector(model_outs):
    losses = model_outs[1]["Focal"]
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
        dict(type="SegOneHot", num_classes=num_classes),
        dict(type="SegResize", size=data_shape[1:]),
        dict(
            type="SegRandomAffine",
            degrees=0,
            scale=(0.5, 2.0),
            interpolation=InterpolationMode.BILINEAR,
            label_fill_value=0,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
        dict(
            type="Scale",
            scales=tuple(1 / np.array(train_scales)),
            mode="bilinear",
        ),
    ],
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
        dict(type="SegOneHot", num_classes=num_classes),
        dict(type="SegResize", size=data_shape[1:]),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)

miou_metric = MeanIOU(seg_class=[str(i) for i in range(num_classes)])


def update_metric(metrics, batch, model_outs):
    # Convert one hot to index
    target: Tensor = batch["gt_seg"][0]
    ignore_points = target.sum(dim=1) == 0
    target = target.argmax(dim=1)
    target[ignore_points] = 255

    preds = model_outs[0][0]
    preds = torch.argmax(preds, dim=1, keepdim=False)

    for metric in metrics:
        metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[miou_metric],
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def tb_update_func(writer, epoch_id, **kwargs):
    name, value = miou_metric.get()
    if isinstance(name, str):
        writer.add_scalar(name, value, global_step=epoch_id)
    else:
        for k, v in zip(name, value):
            writer.add_scalar(k, v, global_step=epoch_id)


# tb data saved to aidi platform is not permanent, save a copy to model dir
tb_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "train"),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[tb_update_func],
)

tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "loss"),
    loss_name_reg="^.*Focal.*",
    update_freq=100,
    update_by="step",
)

# job on aidi platform will have TENSORBOARD_LOG_PATH env,
# where the saved tb can be shown through aidi web UI
aidi_tb_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "train",
    ),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[tb_update_func],
)

aidi_tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "loss",
    ),
    loss_name_reg="^.*Focal.*",
    update_freq=100,
    update_by="step",
)

val_miou_metric = MeanIOU(seg_class=[str(i) for i in range(num_classes)])


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to index
    target: Tensor = batch["gt_seg"]
    ignore_points = target.sum(dim=1) == 0
    target = target.argmax(dim=1)
    target[ignore_points] = 255

    preds = model_outs[0]
    preds = (
        F.interpolate(preds.float(), scale_factor=4, mode="nearest")
        .to(dtype=torch.uint8)
        .squeeze(1)
    )
    for metric in metrics:
        metric.update(target, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


def val_tb_update_func(writer, epoch_id, **kwargs):
    name, value = val_miou_metric.get()
    if isinstance(name, str):
        writer.add_scalar(name, value, global_step=epoch_id)
    else:
        for k, v in zip(name, value):
            writer.add_scalar(k, v, global_step=epoch_id)


val_tb_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "val"),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[val_tb_update_func],
)

val_aidi_tb_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "val",
    ),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[val_tb_update_func],
)

float_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, val_tb_callback, val_aidi_tb_callback],
    val_model=deploy_model,
)

calibration_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, val_tb_callback, val_aidi_tb_callback],
    val_model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
        ],
    ),
)

qat_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, val_tb_callback, val_aidi_tb_callback],
    val_model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
        ],
    ),
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    best_refer_metric=val_miou_metric,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=start_lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[200, 240, 280],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        tb_callback,
        aidi_tb_callback,
        tb_loss_callback,
        aidi_tb_loss_callback,
        float_val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=data_loader,
    batch_processor=val_batch_processor,
    num_steps=1000 // batch_size_per_gpu,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        ckpt_callback,
    ],
    val_metrics=miou_metric,
    log_interval=1,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=0.0,
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
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
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=qat_lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=qat_train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[20],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        tb_callback,
        aidi_tb_callback,
        qat_val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
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
    model=deploy_model,
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
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=deploy_model,
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
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
hbir_infer_model = dict(
    type="SegmentorIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
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
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


def update_align_bpu_metric(metrics, batch, model_outs):
    # Convert one hot to index
    target = batch["gt_seg"]
    ignore_points = target.sum(dim=1) == 0
    target = target.argmax(dim=1)
    target[ignore_points] = 255
    preds = model_outs[0]

    preds = (
        F.interpolate(preds.float(), size=target.shape[1:], mode="nearest")
        .to(dtype=torch.uint8)
        .squeeze(1)
    )
    for metric in metrics:
        metric.update(target, preds)


align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="PILToTensor"),
            dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
            dict(type="SegOneHot", num_classes=num_classes),
            dict(type="SegResize", size=data_shape[1:]),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=None,
)

align_bpu_val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_align_bpu_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    callbacks=[
        align_bpu_val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    image = Image.open(os.path.join(infer_inputs, "img.jpg")).convert("RGB")
    ori_img = np.ascontiguousarray(np.array(image).transpose((2, 0, 1)))

    model_input = {
        "img": image,
        "ori_img": ori_img,
        "layout": "chw",
        "color_space": "rgb",
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)

    return model_input, model_input["ori_img"]


def process_outputs(model_outs, viz_func, vis_inputs):
    img = np.transpose(vis_inputs, (1, 2, 0))
    preds = model_outs[0][0]
    # preds = torch.argmax(preds[0], dim=0)
    viz_func(img, preds)
    return None


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    save_name = "img.jpg"
    img_data = data["img"]
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
    transforms=[
        dict(type="PILToTensor"),
        dict(type="SegResize", size=data_shape[1:]),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    viz_func=dict(type="SegViz", is_plot=True),
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
            dict(type="Float2QAT", convert_mode="eager"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
