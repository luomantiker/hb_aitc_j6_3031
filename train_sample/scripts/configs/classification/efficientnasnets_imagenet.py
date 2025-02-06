import copy
import os
import shutil

import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms import InterpolationMode

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "efficientnasnets_imagenet"
num_classes = 1000
batch_size_per_gpu = 32
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
convert_mode = "fx"
resize_shape = 300
data_shape = 280

SEG_BLOCKS_ARGS = [
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=32,
        out_filters=16,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=2,
        in_filters=16,
        out_filters=24,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=3,
        in_filters=24,
        out_filters=48,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=5,
        num_repeat=3,
        in_filters=48,
        out_filters=88,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=3,
        in_filters=88,
        out_filters=128,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=3,
        in_filters=128,
        out_filters=216,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25,
    ),
    dict(
        kernel_size=3,
        num_repeat=1,
        in_filters=216,
        out_filters=352,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25,
    ),
]

model = dict(
    type="Classifier",
    backbone=dict(
        type="efficientnet",
        bn_kwargs={},
        model_type="b0",
        coefficient_params=(1.0, 1.0, 280, 0.2),
        num_classes=1000,
        include_top=True,
        activation="relu",
        use_se_block=False,
        blocks_args=SEG_BLOCKS_ARGS,
        drop_connect_rate=0.2,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)

deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="efficientnet",
        bn_kwargs={},
        model_type="b0",
        coefficient_params=(1.0, 1.0, 280, 0.2),
        num_classes=1000,
        include_top=True,
        activation="relu",
        flat_output=False,
        use_se_block=False,
        blocks_args=SEG_BLOCKS_ARGS,
        drop_connect_rate=0.2,
    ),
    losses=None,
)

deploy_inputs = dict(img=torch.randn((1, 3, 280, 280)))

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path="./tmp_data/imagenet/train_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TimmTransforms",
                input_size=280,
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m6-mstd0.5-inc1",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                interpolation="bicubic",
            ),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=0.0,
                std=1 / 225.0,
            ),
            dict(type="BgrToYuv444", rgb_input=True),
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
                size=300,
                interpolation=InterpolationMode.BICUBIC,
            ),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=280),
            dict(type="TorchVisionAdapter", interface="ToTensor"),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=0.0,
                std=1 / 225.0,
            ),
            dict(type="BgrToYuv444", rgb_input=True),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    sampler=dict(type=torch.utils.data.DistributedSampler),
)


def loss_collector(outputs):
    return [outputs[1]]


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=loss_collector,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
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
    target = batch["labels"]
    pred = model_outs[0]
    for m in metrics:
        m.update(target, pred)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_loss(metrics, batch, model_outs):
    target = batch["labels"]
    metrics[0].update(model_outs[1])
    metrics[1].update(target, model_outs[0])


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
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
    log_interval=100,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=1e-5)},
        lr=0.5,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=120,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=5,
            step_log_interval=1000,
        ),
        loss_show_update,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="Accuracy"),
    ],
    val_metrics=[dict(type="Accuracy"), dict(type="TopKAccuracy", top_k=5)],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

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

infer_transforms = [
    dict(
        type="TorchVisionAdapter",
        interface="Resize",
        size=resize_shape,
        interpolation=InterpolationMode.BICUBIC,
    ),
    dict(type="TorchVisionAdapter", interface="CenterCrop", size=data_shape),
    dict(type="TorchVisionAdapter", interface="ToTensor"),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=0.0,
        std=1 / 225.0,
    ),
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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
