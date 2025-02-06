import os

import torch
from horizon_plugin_pytorch.march import March

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

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
num_queries = 256

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
    compile_qim=True,
)

deploy_inputs = dict(
    output_embedding=torch.randn((1, 1, num_queries, 256), dtype=torch.float),
    query_pos=torch.randn((1, 1, num_queries, 256), dtype=torch.float),
    mask_query=torch.ones((1, 1, 1, num_queries), dtype=torch.float),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
    ],
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=os.path.join(ckpt_dir, "qim"),
    trace_inputs=deploy_inputs,
)


ckpt_callback = dict(
    type="Checkpoint",
    save_dir=os.path.join(ckpt_dir, "qim"),
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=False,
    mode="max",
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

qat_predictor = dict(
    type="Predictor",
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
        ],
    ),
)

compile_dir = os.path.join(ckpt_dir, "qim", "compile")
hbir_save_dir = os.path.join(ckpt_dir, "qim")
compile_cfg = dict(
    march=march,
    name="qim",
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr", "ddr", "ddr"],
    opt="O2",
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
    out_dir=os.path.join(ckpt_dir, "qim"),
)
