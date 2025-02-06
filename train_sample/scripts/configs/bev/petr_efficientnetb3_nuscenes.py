import copy
import os
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from hat.data.collates.nusc_collates import collate_nuscenes
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "petr_efficientnetb3_nuscenes"
num_classes = 10
batch_size_per_gpu = 1
dataloader_workers = 1
device_ids = [0, 1, 2, 3]  # [0, 1, 2, 3, 4, 5, 6, 7]  # 1 node
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_E
convert_mode = "fx"
qat_mode = "fuse_bn"

num_query = 900
num_levels = 4
query_align = 128

orig_shape = (3, 900, 1600)
resize_shape = (3, 792, 1408)
data_shape = (3, 512, 1408)
val_data_shape = (3, 512, 1408)

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

vt_input_hw = (16, 44)
data_rootdir = "./tmp_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/nuscenes/meta"

model = dict(
    type="Detr3d",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="PETRHead",
        num_query=num_query,
        query_align=query_align,
        in_channels=384,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_views=6,
        depth_num=64,
        depth_start=1,
        reg_out_channels=10,
        cls_out_channels=num_classes,
        bev_range=bev_range,
        position_range=position_range,
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        transformer=dict(
            type="PETRTransformer",
            decoder=dict(
                type="PETRDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                feedforward_channels=2048,
            ),
        ),
        int8_output=False,
        dequant_output=True,
    ),
    target=dict(
        type="Detr3dTarget",
        bev_range=bev_range,
        cls_cost=dict(
            type="FocalLossCost",
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
        ),
        reg_cost=dict(
            type="BBox3DL1Cost",
            weight=0.25,
        ),
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=num_classes + 1,
        loss_weight=2.0,
        alpha=0.25,
        gamma=2.0,
    ),
    loss_reg=dict(
        type="L1Loss",
        loss_weight=0.25,
    ),
    post_process=dict(
        type="Detr3dPostProcess",
        max_num=300,
        score_threshold=-1,
        bev_range=bev_range,
    ),
)

deploy_model = dict(
    type="Detr3d",
    compile_model=True,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="PETRHead",
        num_query=num_query,
        query_align=query_align,
        in_channels=384,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_views=6,
        depth_num=64,
        depth_start=1,
        reg_out_channels=10,
        cls_out_channels=num_classes,
        position_range=position_range,
        bev_range=bev_range,
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        transformer=dict(
            type="PETRTransformer",
            decoder=dict(
                type="PETRDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                feedforward_channels=2048,
            ),
        ),
        int8_output=False,
        dequant_output=True,
    ),
)


def get_deploy_input():
    inputs = {
        "img": torch.randn((6, 3, 512, 1408)),
        "pos_embed": torch.randn((1, 256, 96, 44)),
    }

    return inputs


deploy_inputs = get_deploy_input()


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=[
            dict(type="MultiViewsImgResize", size=(792, 1408)),
            dict(type="MultiViewsImgCrop", size=(512, 1408), random=False),
            dict(
                type="MultiViewsGridMask",
                use_h=True,
                use_w=True,
                rotate=1,
                offset=False,
                ratio=0.5,
                mode=1,
                prob=0.7,
            ),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        with_bev_mask=False,
        bev_range=bev_range,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=[
            dict(type="MultiViewsImgResize", size=(792, 1408)),
            dict(type="MultiViewsImgCrop", size=(512, 1408)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        with_bev_mask=False,
        bev_range=bev_range,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
    sampler=dict(type=torch.utils.data.DistributedSampler),
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
    log_prefix="loss_ " + task_name,
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
    step_log_freq=50000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
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
    val_model=None,
    init_with_train_model=False,
    val_interval=10,
    val_on_train_end=True,
    log_interval=200,
)

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    save_prefix="./WORKSPACE/results" + task_name,
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
                checkpoint_path=os.path.join(
                    "./tmp_pretrained_models/fcos3d_efficientnetb3_nuscenes/float-checkpoint-best.pth.tar",  # noqa: E501
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
            "backbone": dict(lr=2e-4),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=2e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=24,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=1e-6,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    sync_bn=True,
    val_metrics=[val_nuscenes_metric],
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
    val_metrics=[val_nuscenes_metric],
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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        params={
            "backbone": dict(lr=2e-5),
        },
        lr=2e-5,
        weight_decay=0.01,
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
    val_metrics=[val_nuscenes_metric],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="petr_test_model",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid", "ddr"],
    opt="O2",
    split_dim=dict(
        inputs={
            "0": [0, 6],
        }
    ),
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
    metrics=[val_nuscenes_metric],
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
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
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
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_transforms = [
    dict(type="MultiViewsImgResize", size=(792, 1408)),
    dict(type="MultiViewsImgCrop", size=(512, 1408)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

hbir_deploy_model = copy.deepcopy(deploy_model)
hbir_deploy_model["compile_model"] = False

hbir_infer_model = dict(
    type="Detr3dIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=qat_predictor["model_convert_pipeline"],
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    post_process=dict(
        type="Detr3dPostProcess", max_num=300, bev_range=bev_range
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
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesFromImage",
        src_data_dir="./tmp_orig_data/nuscenes",
        version="v1.0-trainval",
        split_name="val",
        transforms=infer_transforms,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        with_bev_mask=False,
        bev_range=bev_range,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo


def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]

    return resized_img, orig_shape


def process_inputs(infer_inputs, transforms=None):

    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]
    orig_imgs = []
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))
    image_dir_list.sort()
    for i, img in enumerate(image_dir_list):
        img = os.path.join(infer_inputs, img)
        img, orig_shape = process_img(img, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(os.path.join(infer_inputs, "ego2img.npy"))

    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))

    model_input = {
        "img": input_imgs,
        "ego2img": torch.tensor(homo),
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["meta"] = {"ego2img": homo}

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = {"ego_det": model_outs}
    viz_func(vis_inputs["img"], preds, vis_inputs["meta"])
    return None


single_infer_dataset = copy.deepcopy(align_bpu_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, (img_name, img_data) in enumerate(
        zip(data["img_name"], data["img"])
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")

    ego2img_path = os.path.join(save_path, "ego2img.npy")
    np.save(ego2img_path, np.array(data["ego2img"]))


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(type="NuscenesViz", is_plot=True),
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
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
)
