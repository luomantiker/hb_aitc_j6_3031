import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torch import Tensor

from hat.data.collates.collates import collate_2d
from hat.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS
from hat.metrics.mean_iou import MeanIOU
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "fastscnn_efficientnetb0tiny_cityscapes"
num_classes = 19
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3]
dataloader_workers = batch_size_per_gpu  # per gpu
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_E
enable_amp = False
convert_mode = "fx"

data_shape = (3, 512, 1024)
val_data_shape = (3, 1024, 2048)
bn_kwargs = dict(eps=1e-5, momentum=0.01)

weight_decay = 4e-5
start_lr = 0.12
train_epochs = 900

qat_lr = 2e-4
qat_train_epochs = 30

data_rootdir = os.path.join(".", "tmp_data", "cityscapes")

# model
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0tiny",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="FastSCNNNeck",
        in_channels=[24, 160],
        feat_channels=[128, 128],
        indexes=[-3, -1],
        bn_kwargs=bn_kwargs,
        split_pooling=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableFCNHead",
        in_channels=128,
        feat_channels=128,
        num_classes=19,
        input_index=-1,
        bn_kwargs=bn_kwargs,
        argmax_output=False,
        dequant_output=True,
        int8_output=False,
        dropout_ratio=0.1,
        num_convs=1,
        upsample_output_scale=None,
    ),
    target=dict(
        type="FCNTarget",
        num_classes=19,
    ),
    loss=dict(
        type="CrossEntropyLoss",
        loss_name="decode",
        use_sigmoid=True,
        reduction="mean",
        ignore_index=255,
        loss_weight=1.0,
    ),
    decode=dict(
        type="FCNDecoder",
        upsample_output_scale=8,
    ),
    auxiliary_heads=[
        dict(
            head=dict(
                type="FCNHead",
                input_index=1,
                in_channels=128,
                feat_channels=32,
                num_classes=19,
                dropout_ratio=0.1,
                num_convs=1,
                bn_kwargs=bn_kwargs,
            ),
            target=dict(
                type="FCNTarget",
                num_classes=19,
            ),
            loss=dict(
                type="CrossEntropyLoss",
                loss_name="aux1",
                ignore_index=255,
                use_sigmoid=True,
                reduction="mean",
                loss_weight=0.4,
            ),
        ),
        dict(
            head=dict(
                type="FCNHead",
                input_index=0,
                in_channels=24,
                feat_channels=32,
                num_classes=19,
                dropout_ratio=0.1,
                num_convs=1,
                bn_kwargs=bn_kwargs,
            ),
            target=dict(
                type="FCNTarget",
                num_classes=19,
            ),
            loss=dict(
                type="CrossEntropyLoss",
                loss_name="aux2",
                ignore_index=255,
                use_sigmoid=True,
                reduction="mean",
                loss_weight=0.4,
            ),
        ),
    ],
)

deploy_model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0tiny",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="FastSCNNNeck",
        in_channels=[24, 160],
        feat_channels=[128, 128],
        indexes=[-3, -1],
        bn_kwargs=bn_kwargs,
        split_pooling=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableFCNHead",
        in_channels=128,
        feat_channels=128,
        num_classes=19,
        input_index=-1,
        bn_kwargs=bn_kwargs,
        argmax_output=False,
        dequant_output=True,
        int8_output=False,
        dropout_ratio=0.1,
        num_convs=1,
    ),
)

deploy_inputs = {"img": torch.randn((1,) + val_data_shape)}

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
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
            dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
            dict(type="TensorToNumpy"),
            dict(
                type="Resize",
                img_scale=(1024, 2048),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(
                type="SegRandomCrop",
                size=(512, 1024),
                cat_max_ratio=0.75,
                ignore_index=255,
            ),
            dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
            dict(type="SegOneHot", num_classes=19),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
        color_space="rgb",
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
            dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
            dict(type="TensorToNumpy"),
            dict(
                type="Resize",
                img_scale=(1024, 2048),
                keep_ratio=True,
            ),
            dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
        color_space="rgb",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[],
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=[],
)

val_miou_metric = MeanIOU(
    seg_class=[str(i) for i in range(num_classes)], ignore_index=255
)


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to index
    target: Tensor = batch["gt_seg"]

    preds = model_outs
    for metric in metrics:
        metric.update(target, preds)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
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
        loss_show_update,
        dict(
            type="PolyLrUpdater",
            max_update=160000,
            warmup_len=0,
            warmup_by="epoch",
            step_log_interval=50,
            power=0.9,
            final_lr=1e-4,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
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
    log_interval=calibration_step / 10,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name + "_model",
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
        qat_mode="fuse_bn",
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
                verbose=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_infer_model = dict(
    type="EncoderDecoderIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="FCNDecoder",
        upsample_output_scale=8,
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
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="PILToTensor"),
            dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
            dict(type="TensorToNumpy"),
            dict(
                type="Resize",
                img_scale=(1024, 2048),
                keep_ratio=True,
            ),
            dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
        color_space="rgb",
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
    callbacks=[
        val_metric_updater,
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
    preds = model_outs[0]
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
        dict(type="TensorToNumpy"),
        dict(
            type="Resize",
            img_scale=(1024, 2048),
            keep_ratio=True,
        ),
        dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
        dict(type="Normalize", mean=128.0, std=128.0),
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
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
)
