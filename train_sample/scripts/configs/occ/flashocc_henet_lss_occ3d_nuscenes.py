import copy
import math
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
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = True

task_name = "flashocc_henet_lss_occ3d_nuscenes"
num_classes = 18
batch_size_per_gpu = 4
val_batch_size_per_gpu = 1
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
qat_mode = "fuse_bn"
convert_mode = "fx"
dataset_type = "Occ3dNuscenesDataset"
train_data_path = "tmp_data/occ3d_nuscenes/train_lmdb/"
val_data_path = "tmp_data/occ3d_nuscenes/val_lmdb/"

train_interval = 1
val_interval = 1
val_log_interval = 4
num_epochs = 150
step_log_freq = 100
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


bn_kwargs = dict(eps=2e-5, momentum=0.1)
depth = 45
num_points = 10
bev_size = (40, 40, 0.625)
grid_size = (128, 128)

orig_shape = (3, 900, 1600)
data_shape = (3, 512, 960)  # (3, 256, 704)
val_data_shape = (3, 512, 960)  # (3, 256, 704)
resize_shape = (3, 540, 960)  # (3, 396, 704)

view_shape = [data_shape[1] / 32, data_shape[2] / 32]
vt_input_hw = [int(view_shape[0]), int(view_shape[1])]
depthview_shape = [6 * depth, view_shape[0] * view_shape[1]]
featview_shape = [view_shape[0] * 6, view_shape[1]]
grid_quant_scale = get_grid_quant_scale(grid_size, featview_shape)
depth_quant_scale = get_grid_quant_scale(grid_size, depthview_shape)

occ3d_seg_class = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

model = dict(
    type="ViewFusion",
    bev_feat_index=-1,
    bev_upscale=2,
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=[4, 3, 8, 6],
        embed_dims=[64, 128, 192, 384],
        attention_block_num=[0, 0, 0, 0],
        mlp_ratios=[2, 2, 2, 3],
        mlp_ratio_attn=2,
        act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
        use_layer_scale=[True, True, True, True],
        layer_scale_init_value=1e-5,
        num_classes=1000,
        include_top=False,
        extra_act=[False, False, False, False],
        final_expand_channel=0,
        feature_mix_channel=1024,
        block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
        down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
        patch_embed="origin",
    ),
    neck=dict(
        type="FPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        out_strides=[16, 32],
        out_channels=[256, 256],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="LSSTransformer",
        in_channels=256,
        feat_channels=64,
        z_range=(-1.0, 5.4),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="HENet",
            in_channels=64,
            block_nums=[4, 3, 8, 6],
            embed_dims=[64, 128, 192, 384],
            attention_block_num=[0, 0, 0, 0],
            mlp_ratios=[2, 2, 2, 3],
            mlp_ratio_attn=2,
            act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
            use_layer_scale=[True, True, True, True],
            layer_scale_init_value=1e-5,
            num_classes=1000,
            include_top=False,
            extra_act=[False, False, False, False],
            final_expand_channel=0,
            feature_mix_channel=1024,
            block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
            down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
            patch_embed="origin",
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 64, 4: 64, 8: 128, 16: 192, 32: 384}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=True,
        ),
    ),
    bev_decoders=[
        dict(
            type="FlashOccDetDecoder",
            use_mask=True,
            num_classes=num_classes,
            occ_head=dict(
                type="BEVOCCHead2D",
                in_dim=48,
                out_dim=128,
                Dz=16,
                num_classes=num_classes,
                use_predicter=True,
                use_upsample=True,
            ),
            loss_occ=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
            ),
        ),
    ],
)

deploy_model = copy.deepcopy(model)
deploy_model["compile_model"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True


bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

scale = float(resize_shape[2]) / float(orig_shape[2])
resize_aug = (-0.06, 0.11)
train_transforms = [
    dict(
        type="MultiViewsImgResize",
        scales=tuple(x + scale for x in resize_aug),
    ),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(type="MultiViewsImgFlip", prob=0.5),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(
        type="BevFeatureAug",
        bda_aug_conf=bda_aug_conf,
        is_train=True,
    ),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

test_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_path=train_data_path,
        load_interval=train_interval,
        transforms=train_transforms,
    ),
    val=dict(
        type=dataset_type,
        transforms=test_transforms,
        load_interval=val_interval,
        data_path=val_data_path,
    ),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["train"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["val"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=val_batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(1),
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss(metrics, batch, model_out):
    for metric in metrics:
        metric.update(model_out[1])


def val_update_metric_func(metrics, batch, model_outs):
    gt_semantics = batch["voxel_semantics"][0].squeeze()  # (Dx, Dy, Dz)
    mask_camera = batch["mask_camera"][0].squeeze()  # (Dx, Dy, Dz)
    semantics_pred = model_outs[1]["occ_pre"].squeeze()  # (Dx, Dy, Dz)

    masked_semantics_gt = gt_semantics[mask_camera]
    masked_semantics_pred = semantics_pred[mask_camera]

    results = {
        "label": masked_semantics_gt.reshape(-1),
        "preds": masked_semantics_pred.reshape(-1),
    }

    for metric in metrics:
        metric.update(**results)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric_func,
    step_log_freq=1000000,
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
    strict_match=True,
    mode="max",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    val_interval=val_log_interval,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=model,
    val_on_train_end=True,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path="./tmp_pretrained_models/henet_tinym_imagenet/float-checkpoint-best.pth.tar",
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=1e-4,
        weight_decay=1e-2,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
)


calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_val_callback["val_on_train_end"] = False
calibration_step = 10
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2Calibration", convert_mode=convert_mode),
    ],
)
calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

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
            dict(
                type="FixWeightQScale",
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        calibration_ckpt_callback,
    ],
    val_metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    log_interval=calibration_step / 10,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name + "_model",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source="pyramid,ddr,ddr",
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
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
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
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

hbir_deploy_model = copy.deepcopy(deploy_model)
hbir_deploy_model["compile_model"] = False

hbir_infer_model = dict(
    type="ViewFusionIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
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
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized.bc"),
    ),
    bev_decoder_infers=[
        dict(
            type="FlashOccDecoderInfer",
            name="occ_det",
        ),
    ],
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
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MeanIOU",
        seg_class=occ3d_seg_class,
        ignore_index=17,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
deploy_inputs = {
    "img": torch.randn((6, 3, data_shape[1], data_shape[2])),
    "points0": torch.randn(
        (num_points, 128, 128, 2),
    ),
    "points1": torch.randn(
        (num_points, 128, 128, 2),
    ),
}


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

    homo = np.load(os.path.join(infer_inputs, "ego2img.npy")).astype("float64")

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

    return model_input, None


def process_outputs(model_outs, viz_func, vis_inputs):
    semantics_pred = (
        model_outs[1]["occ_pre"].squeeze().numpy().astype(np.uint8)
    )

    viz_func(semantics_pred)
    return None


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, img_data in enumerate(data["img"]):
        save_name = f"img{image_idx}.jpg"
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
    viz_func=dict(
        type="OccViz",
        vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        vis_bev_2d=True,
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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
