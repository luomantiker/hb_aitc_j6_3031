import copy
import os
import re
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms import Compose

from hat.data.collates.collates import collate_mot_seq
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "motr_efficientnetb3_mot17"
num_classes = 1

ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
qat_mode = "fuse_bn"
convert_mode = "eager"

img_shape = (800, 1422)

train_lmdb = "./tmp_data/mot17/train_lmdb"
val_lmdb = "./tmp_data/mot17/test_lmdb"
val_gt = "./tmp_data/mot17/test_gt"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

train_batch_size_per_gpu = 1
test_batch_size_per_gpu = 1

num_queries = 256
model = dict(
    type="Motr",
    backbone=dict(
        type="efficientnet",
        bn_kwargs={},
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="MotrHead",
        transformer=dict(
            type="MotrDeformableTransformer",
            pos_embed=dict(
                type="PositionEmbeddingSine",
                num_pos_feats=128,
                normalize=True,
                temperature=20,
            ),
            d_model=256,
            num_queries=num_queries,
            dim_feedforward=1024,
            dropout=0.0,
            return_intermediate_dec=True,
            extra_track_attn=True,
            enc_n_points=1,
            dec_n_points=1,
        ),
        num_classes=num_classes,
        in_channels=[384],
        max_per_img=num_queries,
    ),
    criterion=dict(
        type="MotrCriterion",
        num_classes=num_classes,
    ),
    post_process=dict(
        type="MotrPostProcess",
    ),
    track_embed=dict(
        type="QueryInteractionModule",
        dim_in=256,
        hidden_dim=1024,
    ),
    batch_size=train_batch_size_per_gpu,
)
test_model = copy.deepcopy(model)

deploy_model = dict(
    type="Motr",
    backbone=dict(
        type="efficientnet",
        bn_kwargs={},
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="MotrHead",
        transformer=dict(
            type="MotrDeformableTransformer",
            pos_embed=dict(
                type="PositionEmbeddingSine",
                num_pos_feats=128,
                normalize=True,
                temperature=20,
            ),
            d_model=256,
            num_queries=num_queries,
            dim_feedforward=1024,
            dropout=0.1,
            return_intermediate_dec=False,
            extra_track_attn=True,
            enc_n_points=1,
            dec_n_points=1,
        ),
        num_classes=num_classes,
        in_channels=[384],
        max_per_img=num_queries,
    ),
    track_embed=dict(
        type="QueryInteractionModule",
        dim_in=256,
        hidden_dim=1024,
    ),
    compile_motr=True,
)

deploy_inputs = dict(
    img=torch.randn((1, 3, img_shape[0], img_shape[1])),
    query_pos=torch.randn((1, 256, 2, 128), dtype=torch.float),
    mask_query=torch.ones((1, 1, 1, num_queries), dtype=torch.float),
    ref_pts=torch.randn((1, 4, 2, 128), dtype=torch.float),
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Mot17Dataset",
        data_path=train_lmdb,
        sampler_lengths=[2, 3, 4, 5],
        sample_mode="random_interval",
        sample_interval=10,
        sampler_steps=[100, 180, 240],
        transforms=[
            dict(
                type="SeqRandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomSelectOne",
                transforms=[
                    dict(
                        type="SeqResize",
                        img_scale=[
                            (608, 1536),
                            (640, 1536),
                            (672, 1536),
                            (704, 1536),
                            (736, 1536),
                            (768, 1536),
                            (800, 1536),
                            (832, 1536),
                            (864, 1536),
                            (896, 1536),
                            (928, 1536),
                            (960, 1536),
                            (992, 1536),
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                        rm_neg_coords=False,
                        divisor=2,
                    ),
                    dict(
                        type=Compose,
                        transforms=[
                            dict(
                                type="SeqResize",
                                img_scale=[
                                    (400, 9999999),
                                    (500, 9999999),
                                    (600, 9999999),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                                rm_neg_coords=False,
                                divisor=2,
                            ),
                            dict(
                                type="SeqRandomSizeCrop",
                                min_size=384,
                                max_size=600,
                                filter_area=False,
                                rm_neg_coords=False,
                            ),
                            dict(
                                type="SeqResize",
                                img_scale=[
                                    (608, 1536),
                                    (640, 1536),
                                    (672, 1536),
                                    (704, 1536),
                                    (736, 1536),
                                    (768, 1536),
                                    (800, 1536),
                                    (832, 1536),
                                    (864, 1536),
                                    (896, 1536),
                                    (928, 1536),
                                    (960, 1536),
                                    (992, 1536),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                                rm_neg_coords=False,
                                divisor=2,
                            ),
                        ],
                    ),
                ],
                p=1,
            ),
            dict(
                type="SeqToTensor",
                to_yuv=False,
            ),
            dict(type="SeqBgrToYuv444", rgb_input=True),
            dict(
                type="SeqNormalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type="DistSetEpochDatasetSampler"),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_mot_seq,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Mot17Dataset",
        data_path=val_lmdb,
        sampler_lengths=[1],
        sample_mode="fixed_interval",
        sample_interval=10,
        transforms=[
            dict(
                type="SeqResize",
                img_scale=(800, 1422),
                keep_ratio=False,
            ),
            dict(
                type="SeqToTensor",
                to_yuv=False,
            ),
            dict(type="SeqBgrToYuv444", rgb_input=True),
            dict(
                type="SeqNormalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_mot_seq,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        mean_loss = sum(loss) / len(loss)
        losses.append(mean_loss)
    return losses


def update_loss(metrics, batch, model_outs):
    mean_losses = {}
    for loss_name, loss in model_outs.items():
        mean_loss = sum(loss) / len(loss)
        mean_losses[loss_name] = mean_loss
    for metric in metrics:
        metric.update(mean_losses)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=20,
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
    log_freq=20,
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
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    init_with_train_model=True,
    val_interval=1,
    val_on_train_end=True,
    log_interval=200,
    val_model=test_model,
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
                    "./tmp_pretrained_models/fcos_efficientnetb3_mscoco/float-checkpoint-best.pth.tar"  # noqa
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
            "backbone": dict(lr=2e-5),
            "sampling_offsets": dict(lr=2e-5),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=2e-4,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=400,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[200],
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
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/float_train"),
    ),
    sync_bn=True,
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["sampler_steps"] = [3, 5, 7]
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = train_batch_size_per_gpu * 2
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
        ),
    ],
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
                allow_miss=True,
                verbose=True,
            ),
            dict(type="Float2Calibration"),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        calibration_val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_train"),
    ),
)

qat_train_dataloader = copy.deepcopy(data_loader)
qat_train_dataloader["dataset"]["sampler_steps"] = [3, 5, 7]
qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1
qat_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
        ),
    ],
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
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
            dict(
                type="Float2QAT",
            ),
        ],
    ),
    data_loader=qat_train_dataloader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=1e-4,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[5],
            step_log_interval=500,
        ),
        qat_val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_train"),
    ),
)


int_infer_trainer = dict(
    type="Trainer",
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
            dict(
                type="ExportHbir",
                example_input=deploy_inputs,
                save_path=os.path.join(ckpt_dir, "qat.mlir"),
            ),
            dict(
                type="ConvertHbir",
                march=march,
                save_path=os.path.join(ckpt_dir, "quantized.mlir"),
            ),
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

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="motr",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid", "ddr", "ddr", "ddr"],
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
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/float_predict"),
    ),
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
            dict(type="Float2Calibration"),
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
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_predict"),
    ),
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
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_predict"),
    ),
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)

hbir_infer_model = dict(
    type="MotrIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    qim_ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "qim", "quantized.bc"),
    ),
    post_process=dict(
        type="MotrPostProcess",
    ),
    LoadCheckpoint=dict(
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
        ),
    ),
    num_classes=num_classes,
)
int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False
int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/int_predict"),
    ),
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)
infer_transforms = [
    dict(
        type="SeqResize",
        img_scale=(800, 1422),
        keep_ratio=False,
    ),
    dict(
        type="SeqToTensor",
        to_yuv=False,
    ),
    dict(type="SeqBgrToYuv444", rgb_input=True),
    dict(
        type="SeqNormalize",
        mean=128.0,
        std=128.0,
    ),
]

infer_ckpt = int_infer_trainer["model_convert_pipeline"]["converters"][1][
    "checkpoint_path"
]

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Mot17FromImage",
        data_path="./tmp_orig_data/mot17/split_data/test",
        sampler_lengths=[1],
        sample_mode="fixed_interval",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_mot_seq,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/align_bpu"),
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    image = Image.open(infer_inputs).convert("RGB")
    ori_img = np.array(image)

    model_input = {
        "frame_data_list": [
            {
                "img": ori_img,
                "ori_img": ori_img,
                "layout": "hwc",
                "color_space": "rgb",
                "img_name": [os.path.basename(infer_inputs)],
                "seq_name": ["demo"],
            },
        ]
    }

    model_input = transforms(model_input)
    model_input["frame_data_list"][0]["img"] = [
        model_input["frame_data_list"][0]["img"].unsqueeze(0)
    ]

    return model_input, model_input["frame_data_list"][0]["ori_img"]


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    viz_func(vis_inputs, preds)
    return None


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)

    image_dir_list = [
        os.path.join(infer_inputs, img_path) for img_path in image_dir_list
    ]

    return image_dir_list


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None
single_infer_dataset["sampler_lengths"] = [2]


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for img_idx, frame_data in enumerate(data["frame_data_list"]):
        save_name = f"{img_idx}.jpg"
        img_data = Image.fromarray(frame_data["img"], mode="RGB")
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
    viz_func=dict(type="TrackViz", is_plot=True),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
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
            ),
        ],
    ),
)
