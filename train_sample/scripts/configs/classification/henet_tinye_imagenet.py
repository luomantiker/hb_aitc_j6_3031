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

num_classes = 1000
batch_size_per_gpu = 128
image_shape = (3, 224, 224)
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
convert_mode = "fx"
auto_augment = True
mixup = True
float_train_epochs = 300
task_name = "henet_tinye_imagenet"
ckpt_dir = "./tmp_models/%s" % task_name
ImageNet_path = "./tmp_data/imagenet/"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# ---------------------- TinyE ----------------------
depth = [3, 3, 8, 6]
block_cls = ["DWCB", "GroupDWCB", "AltDWCB", "DWCB"]
width = [48, 96, 192, 384]
attention_block_num = [0, 0, 0, 0]
mlp_ratios, mlp_ratio_attn = [2, 2, 2, 3], 2
act_layer = ["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"]
use_layer_scale = [True, True, True, True]
extra_act = [False, False, False, False]
final_expand_channel, feature_mix_channel = 0, 1024
down_cls = ["S2DDown", "S2DDown", "S2DDown", "None"]
patch_embed = "origin"
stage_out_norm = False
lr = 2e-3
float_train_weight_decay = 0.06


model = dict(
    type="Classifier",
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
        include_top=True,
        extra_act=extra_act,
        final_expand_channel=final_expand_channel,
        feature_mix_channel=feature_mix_channel,
        block_cls=block_cls,
        down_cls=down_cls,
        patch_embed=patch_embed,
        stage_out_norm=stage_out_norm,
    ),
    losses=(
        dict(type="CEWithLabelSmooth")
        if not mixup
        else dict(type="SoftTargetCrossEntropy")
    ),
)
deploy_model = dict(
    type="Classifier",
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
        include_top=True,
        extra_act=extra_act,
        final_expand_channel=final_expand_channel,
        feature_mix_channel=feature_mix_channel,
        block_cls=block_cls,
        down_cls=down_cls,
        patch_embed=patch_embed,
        stage_out_norm=stage_out_norm,
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 224, 224)))

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
        data_path=ImageNet_path + "train_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TimmTransforms",
                input_size=image_shape[-1],
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m9-mstd0.5-inc1" if auto_augment else None,
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=[0, 0, 0],
                std=[1.0, 1.0, 1.0],
                interpolation="bicubic",
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=ImageNet_path + "train_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TimmTransforms",
                input_size=image_shape[-1],
                is_training=True,
                auto_augment="rand-m9-mstd0.5-inc1",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=[0, 0, 0],
                std=[1.0, 1.0, 1.0],
                interpolation="bicubic",
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=ImageNet_path + "val_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="Resize",
                size=256,
                interpolation=3,
            ),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
            dict(type="TorchVisionAdapter", interface="ToTensor"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=0.0,
            std=1 / 255.0,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)
if mixup:
    batch_processor["batch_transforms"].append(
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
        )
    )

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=0.0,
            std=1 / 255.0,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
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
    log_freq=1000,
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

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=lr,
        weight_decay=float_train_weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=float_train_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=5,
            warmup_begin_lr=1e-6,
            step_log_interval=200,
            stop_lr=1e-5,
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

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100  # default

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
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    log_interval=calibration_step / 10,
)

qat_lr, qat_epochs, lr_decay_id, lr_decay_factor = (
    2e-5,
    50,
    [15, 30, 40],
    0.1,
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
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
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
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=4e-5)},
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=qat_lr,
        weight_decay=0.0,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_epochs,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=200,
            lr_decay_id=lr_decay_id,
            lr_decay_factor=lr_decay_factor,
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


# just for saving int_infer pth and pt
int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
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
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
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
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
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
    data_loader=int_infer_data_loader,
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
