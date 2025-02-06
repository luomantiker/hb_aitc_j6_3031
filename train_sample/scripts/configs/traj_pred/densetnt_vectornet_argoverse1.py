import copy
import os
import shutil

import numpy as np
import torch
from horizon_plugin_pytorch.march import March

from hat.data.collates.collates import collate_argoverse
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "densetnt_vectornet_argoverse1"
batch_size_per_gpu = 64
dataloader_workers = 1
device_ids = [0, 1, 2, 3]  # 1 node
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
convert_mode = "fx"
qat_mode = "fuse_bn"

data_rootdir = "./tmp_data/argoverse-1"
map_path = "./tmp_data/argoverse-1/map_files"

model = dict(
    type="MotionForecasting",
    encoder=dict(
        type="Vectornet",
        depth=3,
        traj_in_channels=9,
        traj_num_vec=9,
        lane_in_channels=11,
        lane_num_vec=19,
        hidden_size=128,
    ),
    decoder=dict(
        type="Densetnt",
        in_channels=128,
        hidden_size=128,
        num_traj=32,
        target_graph_depth=2,
        pred_steps=30,
        top_k=150,
    ),
    target=dict(
        type="DensetntTarget",
    ),
    loss=dict(
        type="DensetntLoss",
    ),
    postprocess=dict(
        type="DensetntPostprocess", threshold=2.0, pred_steps=30, mode_num=6
    ),
)

deploy_model = dict(
    type="MotionForecasting",
    encoder=dict(
        type="Vectornet",
        depth=3,
        traj_in_channels=9,
        traj_num_vec=9,
        lane_in_channels=11,
        lane_num_vec=19,
        hidden_size=128,
    ),
    decoder=dict(
        type="Densetnt",
        in_channels=128,
        hidden_size=128,
        num_traj=32,
        target_graph_depth=2,
        pred_steps=30,
        top_k=150,
    ),
)


def get_deploy_input():
    inputs = {
        "traj_feat": torch.randn((30, 9, 19, 32)),
        "lane_feat": torch.randn((30, 11, 9, 64)),
        "instance_mask": torch.randn((30, 1, 1, 96)),
        "goals_2d": torch.randn((30, 2, 1, 2048)),
        "goals_2d_mask": torch.randn((30, 1, 1, 2048)),
    }

    return inputs


deploy_inputs = get_deploy_input()


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse1Dataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_path=map_path,
        pred_step=20,
        max_distance=50.0,
        max_lane_num=64,
        max_traj_num=32,
        max_goals_num=2048,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_argoverse,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse1Dataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_path=map_path,
        pred_step=20,
        max_distance=50.0,
        max_lane_num=64,
        max_traj_num=32,
        max_goals_num=2048,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_argoverse,
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
        metric.update(batch, model_outs)


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

val_metric = dict(
    type="ArgoverseMetric",
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=False,
    mode="min",
    best_refer_metric=val_metric,
)


val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=1,
    val_on_train_end=True,
    log_interval=200,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=1e-3,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosLrUpdater",
            warmup_len=1,
            warmup_by="epoch",
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    sync_bn=True,
    val_metrics=[val_metric],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
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
        val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=[val_metric],
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
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=0.001,
        weight_decay=0.05,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[6],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_metric],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="densetnt_test_model",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr", "ddr", "ddr", "ddr", "ddr"],
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
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_metric],
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
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_metric],
    callbacks=[
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
    metrics=[val_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

hbir_infer_model = dict(
    type="MotionForecastingIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    postprocess=dict(
        type="DensetntPostprocess", threshold=2.0, pred_steps=30, mode_num=6
    ),
    pad_batch=30,
)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 30
int_infer_data_loader["shuffle"] = False

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse1Dataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_path=map_path,
        pred_step=20,
        max_distance=50.0,
        max_lane_num=64,
        max_traj_num=32,
        max_goals_num=2048,
    ),
    batch_size=30,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_argoverse,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[val_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms=None):
    traj_feat = np.load(os.path.join(infer_inputs, "traj_feat.npy"))
    traj_mask = np.load(os.path.join(infer_inputs, "traj_mask.npy"))
    lane_feat = np.load(os.path.join(infer_inputs, "lane_feat.npy"))
    lane_mask = np.load(os.path.join(infer_inputs, "lane_mask.npy"))
    instance_mask = np.load(os.path.join(infer_inputs, "instance_mask.npy"))
    goals_2d = np.load(os.path.join(infer_inputs, "goals_2d.npy"))
    goals_2d_mask = np.load(os.path.join(infer_inputs, "goals_2d_mask.npy"))
    labels = np.load(os.path.join(infer_inputs, "traj_labels.npy"))
    traj_feat_mask = np.load(os.path.join(infer_inputs, "feat_mask.npy"))

    model_inputs = {
        "traj_feat": torch.tensor(traj_feat),
        "lane_feat": torch.tensor(lane_feat),
        "instance_mask": torch.tensor(instance_mask),
        "goals_2d": torch.tensor(goals_2d),
        "goals_2d_mask": torch.tensor(goals_2d_mask),
    }

    vis_inputs = {
        "traj_feat_mask": traj_feat_mask,
        "traj_feat": traj_feat,
        "lane_feat": lane_feat,
        "traj_mask": traj_mask,
        "lane_mask": lane_mask,
        "labels": labels,
    }
    return model_inputs, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds, scores = model_outs
    preds = preds[0][0].cpu().numpy()
    traj_feat_mask = vis_inputs["traj_feat_mask"][0]
    traj_mask = vis_inputs["traj_mask"][0]
    lane_mask = vis_inputs["lane_mask"][0]
    labels = vis_inputs["labels"][0]
    traj_feat = vis_inputs["traj_feat"][0].transpose((2, 1, 0))
    lane_feat = vis_inputs["lane_feat"][0].transpose((2, 1, 0))

    viz_func(
        traj_feat_mask,
        traj_feat,
        traj_mask,
        lane_feat,
        lane_mask,
        labels,
        preds,
    )
    return None


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    inputs_keys_list = [
        "traj_feat",
        "traj_mask",
        "lane_feat",
        "lane_mask",
        "instance_mask",
        "goals_2d",
        "goals_2d_mask",
        "traj_labels",
        "feat_mask",
    ]
    inputs_dict = {}
    for sample_data in data:
        for key_name in inputs_keys_list:
            if key_name in inputs_dict:
                inputs_dict[key_name].append(sample_data[key_name])
            else:
                inputs_dict[key_name] = [sample_data[key_name]]

    for key_name in inputs_keys_list:
        save_path_np = os.path.join(save_path, f"{key_name}.npy")
        np.save(save_path_np, np.array(inputs_dict[key_name]))


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0, 30],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="ArgoverseViz",
    ),
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)
