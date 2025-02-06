import copy
import os
import pickle
import shutil
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.quantization.qconfig_template import (
    default_calibration_qconfig_setter,
    default_qat_fixed_act_qconfig_setter,
    sensitive_op_calibration_8bit_weight_16bit_act_qconfig_setter,
    sensitive_op_qat_8bit_weight_16bit_fixed_act_qconfig_setter,
)

from hat.data.collates.qc_collate import (
    collate_qc_argoverse2,  # Reduce the data sampling frequency
)
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "qcnet_oe_argoverse2"

batch_size_per_gpu = 2 * 4
dataloader_workers = 4
device_ids = [i for i in range(4)]  # noqa
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = 2
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
qat_mode = "fuse_bn"

data_rootdir = "tmp_data/argoverse-2/qc_data"
input_dim = 2
hidden_dim = 128
ori_historical_sec = (
    5  # Length of the original historical trajectory in seconds）
)
ori_future_sec = 6  # Length of the predicted future trajectory in seconds
ori_sample_fre = (
    10  # Original sampling frequency, which is 10 steps per second
)
sample_fre = 2  # Frequency used for downsampling
num_historical_steps = ori_historical_sec * sample_fre
num_future_steps = ori_future_sec * sample_fre
num_freq_bands = 32
num_heads = 8
head_dim = 16
output_dim = 2
time_span = 2
dropout = 0.1
num_map_layers = 1
num_agent_layers = 1
num_modes = 6
num_recurrent_steps = 1
num_t2m_steps = 30 // ori_sample_fre * sample_fre  # times steps for decode

num_pl2a = 32  # pl2a gather
num_a2a = 36  # a2a gather

num_dec_layers = 1
split_rec_modules = True
reuse_agent_rembs = True  # Flag to reuse agent rembs for the decoder.

# quant_infer_cold_start: Indicates the compilation mode for inference.
# True for cold start streaming inference (requires one model compilation).
# False for hot start streaming inference (requires two model compilations).
# Applicable only for prediction or integer inference.
quant_infer_cold_start = True  # Set to False when training

model = dict(
    type="QCNetOE",
    encoder=dict(
        type="QCNetOEEncoder",
        map_encoder=dict(
            type="QCNetOEMapEncoder",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        ),
        agent_encoder=dict(
            type="QCNetOEAgentEncoderStream",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_t2m_steps=num_t2m_steps,
            num_pl2a=num_pl2a,
            num_a2a=num_a2a,
            dropout=dropout,
            save_memory=True,
            stream_infer=True,
            reuse_agent_rembs=reuse_agent_rembs,
            deploy=False,
            quant_infer_cold_start=quant_infer_cold_start,
        ),
    ),
    decoder=dict(
        type="QCNetOEDecoder",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
        num_modes=num_modes,
        num_recurrent_steps=num_recurrent_steps,
        num_t2m_steps=num_t2m_steps,
        num_freq_bands=num_freq_bands,
        num_layers=num_dec_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
        split_rec_modules=split_rec_modules,
        reuse_agent_rembs=reuse_agent_rembs,
        deploy=False,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    loss=dict(
        type="QCNetOELoss",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
    ),
    preprocess=dict(
        type="QCNetOEPreprocess",
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        num_pl2a=num_pl2a,
        num_a2a=num_a2a,
        stream=True,
        save_memory=True,
        deploy=False,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    postprocess=dict(
        type="QCNetOEPostprocess",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
    ),
    quant_infer_cold_start=quant_infer_cold_start,
)


cali_model = copy.deepcopy(model)
cali_model["encoder"]["agent_encoder"]["save_memory"] = False
cali_model["encoder"]["agent_encoder"]["stream_infer"] = True
cali_model["preprocess"]["save_memory"] = False
cali_model["preprocess"]["stream"] = True

deploy_model = dict(
    type="QCNetOE",
    encoder=dict(
        type="QCNetOEEncoder",
        map_encoder=dict(
            type="QCNetOEMapEncoder",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        ),
        agent_encoder=dict(
            type="QCNetOEAgentEncoderStream",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_t2m_steps=num_t2m_steps,
            dropout=dropout,
            save_memory=False,
            stream_infer=True,
            reuse_agent_rembs=reuse_agent_rembs,
            deploy=True,
            quant_infer_cold_start=quant_infer_cold_start,
        ),
    ),
    decoder=dict(
        type="QCNetOEDecoder",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
        num_modes=num_modes,
        num_recurrent_steps=num_recurrent_steps,
        num_t2m_steps=num_t2m_steps,
        num_freq_bands=num_freq_bands,
        num_layers=num_dec_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
        split_rec_modules=split_rec_modules,
        reuse_agent_rembs=reuse_agent_rembs,
        deploy=True,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    quant_infer_cold_start=quant_infer_cold_start,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        split="train",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="train",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=None,
        pt_N_downsample_nums=50,
    ),
)


qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        split="train",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="train",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=None,
        pt_N_downsample_nums=50,
    ),
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        split="val",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=50,
    ),
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
    step_log_freq=50,
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
        metric.update(**model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)

val_metric = dict(type="HitRate")
val_metrics = [
    dict(type="BrierMetric"),
    dict(type="MinADE"),
    dict(type="MinFDE"),
    dict(type="MissRate"),
    dict(type="HitRate"),
]
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
    mode="max",
    best_refer_metric=val_metric,
)


val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    interval_by="epoch",
    val_interval=1,
    val_on_train_end=True,
    log_interval=50,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=5e-4, weight_decay=1e-4),
        custom_param_mapper={
            "bias": dict(weight_decay=0.0),
            "norm_types": dict(weight_decay=0.0),
            nn.Embedding: dict(weight_decay=0.0),
        },
    ),
    batch_processor=batch_processor,
    num_epochs=250,
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
    val_metrics=val_metrics,
)

sensitive_path1 = os.path.join(
    ckpt_dir, "analysis", "output_prob_L1_sensitive_ops.pt"
)
sensitive_path2 = os.path.join(
    ckpt_dir, "analysis", "output_pred_L1_sensitive_ops.pt"
)

if os.path.exists(sensitive_path2):
    sensitive_table1 = torch.load(sensitive_path1)
    sensitive_table2 = torch.load(sensitive_path2)
    cali_qconfig_setter = (
        sensitive_op_calibration_8bit_weight_16bit_act_qconfig_setter(
            sensitive_table1,
            topk=86,
            ratio=None,
        ),
        sensitive_op_calibration_8bit_weight_16bit_act_qconfig_setter(
            sensitive_table2,
            topk=42,
            ratio=None,
        ),
        default_calibration_qconfig_setter,
    )
    qat_qconfig_setter = (
        sensitive_op_qat_8bit_weight_16bit_fixed_act_qconfig_setter(
            sensitive_table1,
            topk=86,
            ratio=None,
        ),
        sensitive_op_qat_8bit_weight_16bit_fixed_act_qconfig_setter(
            sensitive_table2,
            topk=42,
            ratio=None,
        ),
        default_qat_fixed_act_qconfig_setter,
    )
    print("Load sensitive table!")
else:
    cali_qconfig_setter = (default_calibration_qconfig_setter,)
    qat_qconfig_setter = (default_qat_fixed_act_qconfig_setter,)
    print("NOT Load sensitive table!")


# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(qat_data_loader)
calibration_data_loader.pop("sampler")
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0
calibration_data_loader["batch_size"] = 1
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

float2calibration = dict(
    type="Float2Calibration",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=cali_qconfig_setter,
)

float2qat = dict(
    type="Float2QAT",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=qat_qconfig_setter,
)

calibration_trainer = dict(
    type="Calibrator",
    model=cali_model,
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
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            float2calibration,
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        ckpt_callback,
        val_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=val_metrics,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                verbose=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=qat_data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=8 * 5e-6, weight_decay=1e-4),  # gpu: 8*4090
        custom_param_mapper={
            "bias": dict(weight_decay=0.0),
            "norm_types": dict(weight_decay=0.0),
            nn.Embedding: dict(weight_decay=0.0),
        },
    ),
    batch_processor=batch_processor,
    num_epochs=15,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[4],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=val_metrics,
)


# predictor
float_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
        converters=[
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


B = 1
A = 30
pl = 80
pt = 50
HT = ori_historical_sec * sample_fre  # if sample_fre==2 init HT=10

deploy_inputs = OrderedDict()
# Initialize agent as an ordered dictionary
deploy_inputs["agent"] = OrderedDict(
    {
        "valid_mask": (torch.rand(1, A, HT) > 0.2).bool(),  # 0
        "valid_mask_a2a": (torch.rand(1, HT, A, A) > 0.2).bool(),  # 1
        "agent_type": (torch.rand(1, A, 1) * 3).long(),  # 2
        "x_a_cur": [torch.rand(B, 1, A, 1) for _ in range(4)],  # 3 4 5 6
        "r_pl2a_cur": [torch.rand(B, 1, A, pl) for _ in range(3)],  # 7 8 9
        "r_t_cur": [
            torch.rand(B, 1, A, num_t2m_steps) for _ in range(4)
        ],  # 10 11 12 13 stream and reuse
        "r_a2a_cur": [torch.rand(B, 1, A, A) for _ in range(3)],  # 14 15 16
        "mask_a_cur": (torch.rand(B, A) > 0.2).bool(),  # [B, A]   17
        "mask_a2a_cur": (torch.rand(B, A, A) > 0.2).bool(),  # [B, A, A] 18
        "mask_t_key": (torch.rand(B, A, 2) > 0.2).bool(),  # [B, A, 2]   19
        "x_a_mid_emb": [
            torch.rand(B, A, time_span, hidden_dim)
            for _ in range(num_agent_layers)
        ],  # 20
    }
)

deploy_inputs["agent"]["x_a_his"] = torch.zeros(
    B, A, num_t2m_steps - 1, hidden_dim
)  # 21

# Initialize map_polygon as an ordered dictionary
deploy_inputs["map_polygon"] = OrderedDict(
    {
        "pl_type": (torch.rand(1, pl) * 3).long(),  # 22
        "is_intersection": (torch.rand(1, pl)).long(),  # 23
        "r_pl2pl": [torch.rand(B, 1, pl, pl) for _ in range(3)],  # 24 25 26
        "r_pt2pl": [torch.rand(B, 1, pl, pt) for _ in range(3)],  # 27 28 29
        "mask_pl2pl": (
            torch.ones([B, pl, pl]) - torch.eye(pl).unsqueeze(0)
        ).bool(),  # 30
    }
)
# Initialize map_point as an ordered dictionary
deploy_inputs["map_point"] = OrderedDict(
    {
        "magnitude": torch.rand(1, pl, pt),  # 31
        "pt_type": (torch.rand(1, pl, pt) * 3).long(),  # 32
        "side": (torch.rand(1, pl, pt) * 3).long(),  # 33
        "mask": (torch.rand(1, pl, pt) > 0.2).bool(),  # 34
    }
)
# Initialize decoder as an ordered dictionary
deploy_inputs["decoder"] = OrderedDict(
    {
        "mask_a2m": (torch.rand(1, A, A) > 0.2).bool(),  # 35
        "mask_dst": (torch.rand(1, A, 1) > 0.2).bool(),  # 36
    }
)
# Conditionally add additional keys if reuse_agent_rembs is False
if not reuse_agent_rembs:
    deploy_inputs["decoder"].update(
        {
            "r_t2m": [torch.rand(1, 1, A, num_t2m_steps) for _ in range(4)],
            "r_pl2m": [torch.rand(1, 1, A, pl) for _ in range(3)],
            "r_a2m": [torch.rand(1, 1, A, A) for _ in range(3)],
        }
    )

deploy_inputs["type_pl2pl"] = (torch.rand([B, pl, pl]) * 5).long()  # 37


deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
        dict(
            type="SetSoftMaxDivideStrategy",
        ),
    ],
)


hbir_save_dir = ckpt_dir
compile_dir = os.path.join(hbir_save_dir, "compile")


hbir_exporter = dict(
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=hbir_save_dir,
    model_name=task_name,
    march=march,
)

compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
    opt="O2",
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
)
calops_cfg = dict(method="hook")


# -------------------------- int validation config --------------------------

hbir_infer_model = dict(
    type="QCNetOEIrInfer",
    model_path=os.path.join(hbir_save_dir, "quantized.bc"),
    preprocess=dict(
        type="QCNetOEPreprocess",
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        agent_num=A,
        pl_num=pl,
        pt_num=pt,
        stream=True,
        deploy=True,
        quant_infer_cold_start=True,
    ),
    postprocess=dict(
        type="QCNetOEPostprocess",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
    ),
    quant_infer_cold_start=True,
    example_data=deploy_inputs,
)


int_infer_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        split="val",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        agent_num=30,
        pl_N=80,
        pt_N=50,
    ),
)

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=10,
)

# ----------------------- hbm infer validation config --------------------------
# Hbm infer tool test version
hbm_infer_model = dict(
    type="QCNetOEHbmInfer",  # "ClassifierIrInfer",
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(compile_dir, "model.hbm"),
    ),
    hbir_model=os.path.join(hbir_save_dir, "quantized.bc"),
    preprocess=dict(
        type="QCNetOEPreprocess",
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        agent_num=A,
        pl_num=pl,
        pt_num=pt,
        stream=True,
        deploy=True,
        quant_infer_cold_start=True,
    ),
    postprocess=dict(
        type="QCNetOEPostprocess",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
    ),
    quant_infer_cold_start=True,
    example_data=deploy_inputs,
)


hbm_infer_data_loader = copy.deepcopy(val_data_loader)
hbm_infer_data_loader["batch_size"] = 1
hbm_infer_data_loader["shuffle"] = False

hbm_val_batch_processor = copy.deepcopy(val_batch_processor)

hbm_infer_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_infer_data_loader,
    batch_processor=hbm_val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
# ----------------------- hbm infer validation config end --------------------------

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


def process_inputs(infer_inputs, transforms=None):
    with open(os.path.join(infer_inputs, "qcnet_data.pkl"), "rb") as f:
        data = pickle.load(f)
    collate_data = collate_qc_argoverse2(
        [data],
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        agent_num=30,
        pl_N=80,
        pt_N=None,
        pt_N_downsample_nums=50,
    )
    model_inputs = collate_data
    viz_inputs = {
        "data": data,
    }

    return model_inputs, viz_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    map_path = vis_inputs.get("math_path", None)
    data = vis_inputs["data"]
    viz_func(data, preds, map_path)
    return None


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "qcnet_data.pkl"), "wb") as f:
        pickle.dump(data, f)


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[100],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="Argoverse2Viz",
        num_historical_steps=num_historical_steps,
    ),
    process_outputs=process_outputs,
)

# ----- debug ----------------

analysis_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_data_loader=calibration_example_data_loader,
            qconfig_setter=(default_qat_fixed_act_qconfig_setter,),
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best-defaultQconfig.pth.tar"
            ),
        ),
    ],
)
quant_data_loader = copy.deepcopy(val_data_loader)
quant_data_loader["batch_size"] = 1

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=cali_model,
    device_id=0,
    dataloader=quant_data_loader,
    num_steps=1000,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=analysis_convert_pipeline,
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "analysis"),
)
