import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_disp_cat
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "stereonetplus_mixvargenet_sceneflow"
data_num_workers = 2
march = March.NASH_E
ckpt_dir = "./tmp_models/%s" % task_name

train_batch_size_per_gpu = 8
test_batch_size_per_gpu = 8

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
convert_mode = "fx"

loss_weights = [1 / 3, 2 / 3, 1.0]

maxdisp = 192
bias = False
bn_kwargs = {}
refine_levels = 3
base_lr = 0.002
num_epochs = 200

model = dict(
    type="StereoNetPlus",
    backbone=dict(
        type="MixVarGENet",
        net_config=[
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f2",
                    stack_ops=[],
                    stride=1,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=64,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=96,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=96,
                    out_channels=160,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
        ],
        disable_quanti_input=False,
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[0, 1, 2, 3, 4],
    ),
    neck=dict(
        type="FPN",
        in_strides=[8, 16, 32],
        in_channels=[64, 96, 160],
        out_strides=[8, 16, 32],
        out_channels=[16, 16, 16],
    ),
    head=dict(
        type="StereoNetHeadPlus",
        maxdisp=maxdisp,
        bn_kwargs=bn_kwargs,
        refine_levels=refine_levels,
    ),
    post_process=dict(
        type="StereoNetPostProcessPlus",
        maxdisp=maxdisp,
    ),
    loss=dict(type="SmoothL1Loss"),
    loss_weights=loss_weights,
)

deploy_model = dict(
    type="StereoNetPlus",
    backbone=dict(
        type="MixVarGENet",
        net_config=[
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f2",
                    stack_ops=[],
                    stride=1,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=64,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=96,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=96,
                    out_channels=160,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
        ],
        disable_quanti_input=False,
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[0, 1, 2, 3, 4],
    ),
    neck=dict(
        type="FPN",
        in_strides=[8, 16, 32],
        in_channels=[64, 96, 160],
        out_strides=[8, 16, 32],
        out_channels=[16, 16, 16],
    ),
    head=dict(
        type="StereoNetHeadPlus",
        maxdisp=maxdisp,
        bn_kwargs=bn_kwargs,
        refine_levels=refine_levels,
    ),
)

deploy_inputs = dict(
    img=torch.randn((2, 3, 544, 960)),
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlow",
        data_path="./tmp_data/SceneFlow/train_lmdb",
        transforms=[
            dict(
                type="RandomCrop",
                size=(256, 512),
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
                use_yuv_v2=False,
            ),
            dict(
                type="MultiViewsSpiltImgTransformWrapper",
                transforms=[
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(
                        type="TorchVisionAdapter",
                        interface="Normalize",
                        mean=128.0,
                        std=128.0,
                    ),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlow",
        data_path="./tmp_data/SceneFlow/test_lmdb",
        transforms=[
            dict(
                type="Pad",
                divisor=32,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
                use_yuv_v2=False,
            ),
            dict(
                type="MultiViewsSpiltImgTransformWrapper",
                transforms=[
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(
                        type="TorchVisionAdapter",
                        interface="Normalize",
                        mean=128.0,
                        std=128.0,
                    ),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_disp_cat,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def loss_collector(outputs: dict):
    return outputs["losses"]


train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss_metric(metrics, batch, model_outs):
    loss = sum(model_outs["losses"])
    metrics[0].update(loss)
    labels = batch["gt_disp"]
    preds = model_outs["pred_disps"]
    masks = (labels > 0) & (labels < maxdisp)
    metrics[1].update(labels, preds, masks)


loss_show_callback = dict(
    type="MetricUpdater",
    metric_update_func=update_loss_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="train_" + task_name,
)


def update_metric(metrics, batch, model_outs):
    labels = batch["gt_disp"]
    preds = model_outs
    masks = (labels > 0) & (labels < maxdisp)
    metrics[0].update(labels, preds, masks)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1,
    epoch_log_freq=1,
    log_prefix="Validation_" + task_name,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="min",
    monitor_metric_key="EPE",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=base_lr,
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="./tmp_pretrained_models/mixvargenet_imagenet/float-checkpoint-best.pth.tar",
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    batch_processor=train_batch_processor,
    stop_by="epoch",
    num_epochs=num_epochs,
    device=None,
    sync_bn=True,
    callbacks=[
        stat_callback,
        loss_show_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=10,
            step_log_interval=1000,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="percentile",
            activation_calibration_qkwargs=dict(
                percentile=99.985,
                bins=8192,
            ),
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    log_interval=calibration_step / 10,
)
qat_data = copy.deepcopy(data_loader)
qat_data["dataset"]["transforms"] = val_data_loader["dataset"]["transforms"]
qat_data["batch_size"] = 16


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=qat_data,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.00005,
    ),
    batch_processor=train_batch_processor,
    num_epochs=40,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_callback,
        dict(
            type="CosLrUpdater",
            step_log_interval=1000,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    split_dim=dict(
        inputs={
            "0": [0, 2],
        }
    ),
)

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
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
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
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
            ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=1,
)

hbir_infer_model = dict(
    type="StereoNetIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="StereoNetPostProcessPlus",
        maxdisp=maxdisp,
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=1,
)

infer_transforms = [
    dict(
        type="Pad",
        divisor=32,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
        use_yuv_v2=False,
    ),
    dict(
        type="MultiViewsSpiltImgTransformWrapper",
        transforms=[
            dict(type="BgrToYuv444", rgb_input=True),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
]


align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlowFromImage",
        data_path="./tmp_orig_data/SceneFlow/",
        data_list="./tmp_orig_data/SceneFlow/SceneFlow_finalpass_test.txt",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_disp_cat,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_left_img = Image.open(
        os.path.join(infer_inputs, "img_left.jpg")
    ).convert("RGB")
    img_l = np.array(ori_left_img)
    ori_right_img = Image.open(
        os.path.join(infer_inputs, "img_right.jpg")
    ).convert("RGB")
    img_r = np.array(ori_right_img)
    img = np.concatenate((img_l, img_r), axis=2)

    model_input = {
        "img": img,
        "ori_img": copy.deepcopy(img),
        "layout": "hwc",
        "color_space": "rgb",
        "img_shape": img_l.shape,
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"]

    vis_inputs = {}
    vis_inputs["f"] = 1050
    vis_inputs["baseline"] = 0.54
    vis_inputs["img"] = copy.deepcopy(img)

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs.squeeze(0).cpu().numpy()
    f = float(vis_inputs["f"])
    baseline = float(vis_inputs["baseline"])
    img = vis_inputs["img"]
    depth = baseline * f / preds
    preds = viz_func(img, preds, depth)
    return None


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    img_data = Image.fromarray(data["ori_img"][..., :3], mode="RGB")
    img_data.save(os.path.join(save_path, "img_left.jpg"), "JPEG")
    img_data = Image.fromarray(data["ori_img"][..., 3:], mode="RGB")
    img_data.save(os.path.join(save_path, "img_right.jpg"), "JPEG")


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
    viz_func=dict(type="DispViz", is_plot=True),
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
