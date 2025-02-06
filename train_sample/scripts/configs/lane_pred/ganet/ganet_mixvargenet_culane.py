import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2


training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "ganet_mixvargenet_culane"
data_num_workers = 4
march = March.NASH_E
convert_mode = "fx"
ckpt_dir = "./tmp_models/%s" % task_name

batch_size_per_gpu = 64
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

train_data_path = "./tmp_data/CULane/train_lmdb"
val_data_path = "./tmp_data/CULane/test_lmdb"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True


base_lr = 0.01
num_epochs = 240
bn_kwargs = {}
radius = 2
hid_dim = 32
attn_ratio = 4

model = dict(
    type="GaNet",
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
        output_list=[2, 3, 4],
    ),
    neck=dict(
        type="GaNetNeck",
        fpn_module=dict(
            type="FPN",
            in_strides=[8, 16, 32],
            in_channels=[64, 96, hid_dim],
            out_strides=[8, 16, 32],
            out_channels=[hid_dim, hid_dim, hid_dim],
        ),
        attn_in_channels=[160],
        attn_out_channels=[hid_dim],
        attn_ratios=[attn_ratio],
        pos_shape=(1, 10, 25),
    ),
    head=dict(
        type="GaNetHead",
        in_channel=hid_dim,
    ),
    targets=dict(
        type="GaNetTarget",
        hm_down_scale=8,
        radius=radius,
    ),
    post_process=dict(
        type="GaNetDecoder",
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
    ),
    losses=dict(
        type="GaNetLoss",
        loss_kpts_cls=dict(
            type="LaneFastFocalLoss",
            loss_weight=1.0,
        ),
        loss_pts_offset_reg=dict(
            type="L1Loss",
            loss_weight=0.5,
        ),
        loss_int_offset_reg=dict(
            type="L1Loss",
            loss_weight=1.0,
        ),
    ),
)

# deploy model
deploy_model = copy.deepcopy(model)
deploy_model["targets"] = None
deploy_model["losses"] = None
deploy_model["post_process"] = None

deploy_inputs = dict(img=torch.randn((1, 3, 320, 800)))


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path=train_data_path,
        to_rgb=True,
        transforms=[
            dict(
                type="FixedCrop",
                size=(0, 270, 1640, 320),
            ),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0.0,
            ),
            dict(
                type="Resize",
                img_scale=(320, 800),
                multiscale_mode="value",
                keep_ratio=False,
            ),
            dict(
                type="RandomSelectOne",
                transforms=[
                    dict(
                        type="RGBShift",
                        r_shift_limit=(-10, 10),
                        g_shift_limit=(-10, 10),
                        b_shift_limit=(-10, 10),
                        p=1.0,
                    ),
                    dict(
                        type="HueSaturationValue",
                        hue_range=(-10, 10),
                        sat_range=(-15, 15),
                        val_range=(-10, 10),
                        p=1.0,
                    ),
                ],
                p=0.7,
            ),
            dict(
                type="JPEGCompress",
                p=0.2,
                max_quality=85,
                min_quality=95,
            ),
            dict(
                type="RandomSelectOne",
                transforms=[
                    dict(
                        type="MeanBlur",
                        ksize=3,
                        p=1.0,
                    ),
                    dict(
                        type="MedianBlur",
                        ksize=3,
                        p=1.0,
                    ),
                ],
                p=0.2,
            ),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.0, 0.0),
                p=0.5,
            ),
            dict(
                type="ShiftScaleRotate",
                shift_limit=(-0.1, 0.1),
                scale_limit=(0.8, 1.2),
                rotate_limit=(-10, 10),
                interpolation=1,
                border_mode=0,
                p=0.6,
            ),
            dict(
                type="RandomResizedCrop",
                height=320,
                width=800,
                scale=(0.8, 1.2),
                ratio=(1.7, 2.7),
                p=0.6,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    pin_memory=True,
    shuffle=True,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path=val_data_path,
        to_rgb=True,
        transforms=[
            dict(
                type="FixedCrop",
                size=(0, 270, 1640, 320),
            ),
            dict(
                type="Resize",
                img_scale=(320, 800),
                multiscale_mode="value",
                keep_ratio=False,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)
stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=20,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)


def update_metric(metrics, batch, model_outs):
    target = batch["ori_gt_lines"]
    for metric in metrics:
        metric.update(target, model_outs)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=20,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
    monitor_metric_key="CulaneF1Score",
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
    callbacks=[metric_updater],
    val_on_train_end=False,
)

qat_val_callback = copy.deepcopy(val_callback)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=base_lr,
    ),
    batch_processor=train_batch_processor,
    stop_by="epoch",
    num_epochs=num_epochs,
    device=None,
    sync_bn=True,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosLrUpdater",
            warmup_len=1,
            warmup_by="epoch",
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=[
        dict(type="CulaneF1Score"),
    ],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
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
        qat_val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="CulaneF1Score"),
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
    opt="O2",
)

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
    log_interval=50,
    callbacks=[
        stat_callback,
        metric_updater,
    ],
    metrics=[
        dict(type="CulaneF1Score"),
    ],
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
    metrics=[
        dict(type="CulaneF1Score"),
    ],
    callbacks=[
        stat_callback,
        metric_updater,
    ],
    log_interval=50,
)

hbir_infer_model = dict(
    type="GaNetIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="GaNetDecoder",
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
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
    callbacks=[
        stat_callback,
        metric_updater,
    ],
    metrics=[
        dict(type="CulaneF1Score"),
    ],
    log_interval=1,
)

infer_transforms = [
    dict(
        type="FixedCrop",
        size=(0, 270, 1640, 320),
    ),
    dict(
        type="Resize",
        img_scale=(320, 800),
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]


align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path="./tmp_data/CULane/test_lmdb/",
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
    callbacks=[
        metric_updater,
    ],
    metrics=[
        dict(type="CulaneF1Score"),
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_img = Image.open(os.path.join(infer_inputs, "img.jpg"))

    model_input = {
        "img": np.array(ori_img),
        "ori_img": np.array(ori_img),
        "layout": "hwc",
        "color_space": "rgb",
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["ori_img"] = [model_input["ori_img"]]
    model_input["crop_offset"] = [model_input["crop_offset"]]
    model_input["scale_factor"] = [torch.tensor(model_input["scale_factor"])]

    return model_input, np.array(ori_img)


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs[0]
    viz_func(vis_inputs, preds)
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
    viz_func=dict(type="LanelineViz", is_plot=True),
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
            ),
        ],
    ),
)
