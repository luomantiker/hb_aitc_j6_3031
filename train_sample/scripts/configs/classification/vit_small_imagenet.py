import copy
import os
import shutil

import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")


model_type = "vit_small_patch16_224"
task_name = "vit_small_imagenet"

num_classes = 1000
batch_size_per_gpu = 256
gpu_num = 8
device_ids = [i for i in range(gpu_num)]  # noqa [C416]
batch_size = batch_size_per_gpu * gpu_num
base_lr = 1e-4
lr = batch_size / 256.0 * base_lr

ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = 1
log_rank_zero_only = True
march = March.NASH_E
convert_mode = "fx"
size = int(model_type.split("_")[-1])
img_shape = (size, size)
qat_mode = "fuse_bn"

model = dict(
    type="Classifier",
    backbone=dict(
        type="ViT",
        model_type=model_type,
        drop_path_rate=0.1,
        num_classes=1000,
    ),
    losses=dict(
        type="SoftTargetCrossEntropy",
    ),
)
qat_model = dict(
    type="Classifier",
    backbone=dict(
        type="ViT",
        model_type=model_type,
        drop_path_rate=0.0,
        num_classes=1000,
    ),
    losses=dict(
        type="SoftTargetCrossEntropy",
    ),
)


deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="ViT",
        model_type=model_type,
        drop_path_rate=0.0,
        num_classes=1000,
    ),
    losses=None,
)

deploy_inputs = dict(img=torch.randn((1, 3, img_shape[0], img_shape[1])))
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
        type="ImageNet",
        data_path="./tmp_data/imagenet/train_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TimmTransforms",
                input_size=img_shape,
                is_training=True,
                no_aug=False,
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                scale=[0.08, 1.0],
                ratio=[3.0 / 4.0, 4.0 / 3.0],
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation="bicubic",
                mean=[0, 0, 0],
                std=[1.0, 1.0, 1.0],
            ),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=0.0,
                std=1 / 255.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/val_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="Resize",
                size=int(size / 0.875),
            ),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=size),
            dict(type="PILToTensor"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
        dict(
            type="TimmMixup",
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=num_classes,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs[1])


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def val_update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        metric.update(target, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)
ema_callback = dict(type="ExponentialMovingAverage", decay=0.9999)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=lr, betas=(0.9, 0.95), weight_decay=0.3),
        custom_param_mapper=dict(
            norm_types={"weight_decay": 0.0},
            dist_token={"weight_decay": 0.0},
            cls_token={"weight_decay": 0.0},
        ),
    ),
    batch_processor=batch_processor,
    num_epochs=300,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=20,
            step_log_interval=1000,
        ),
        ema_callback,
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 1

calibration_trainer = dict(
    type="Calibrator",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                verbose=True,
                ignore_extra=True,
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
            ),
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
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    log_interval=calibration_step / 10,
)

qat_data_loader = copy.deepcopy(data_loader)
qat_data_loader["batch_size"] = batch_size_per_gpu // 2

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
                verbose=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=qat_data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=4e-5),
        custom_param_mapper=dict(
            norm_types={"weight_decay": 0.0},
            dist_token={"weight_decay": 0.0},
            cls_token={"weight_decay": 0.0},
        ),
    ),
    batch_processor=batch_processor,
    num_epochs=50,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=1,
            step_log_interval=1000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=qat_model,
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
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
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
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=qat_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_transforms = [
    dict(type="TorchVisionAdapter", interface="Resize", size=256),
    dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
    dict(type="TorchVisionAdapter", interface="PILToTensor"),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

hbir_infer_model = dict(
    type="ClassifierIrInfer",
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
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNetFromImage",
        root="./tmp_orig_data/imagenet/val",
        split="val",
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
        dict(type="Accuracy"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms=None):
    ori_img = Image.open(os.path.join(infer_inputs, "img.jpg")).convert("RGB")
    model_input = {
        "img": ori_img,
    }
    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    return model_input, ori_img


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    preds = viz_func(vis_inputs, preds)
    return f"The result is: {int(preds)}"


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    save_name = "img.jpg"
    img_data = data["ori_img"]
    img_data.save(os.path.join(save_path, save_name), "JPEG")


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[500],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(type="ClsViz", is_plot=True),
    process_outputs=process_outputs,
    transforms=infer_transforms,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
