import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from pipeline_tools import (
    align_bpu_validation,
    calops,
    compile_perf,
    export_hbir,
    int_infer_viz_lidar,
    model_checker,
)

from hat.callbacks.checkpoint import Checkpoint
from hat.callbacks.lr_updater import CyclicLrUpdater
from hat.callbacks.metric_updater import MetricUpdater
from hat.callbacks.monitor import StatsMonitor
from hat.callbacks.validation import Validation
from hat.data.collates.collates import collate_lidar3d
from hat.data.datasets.kitti3d import Kitti3D
from hat.data.transforms.lidar_utils.lidar_transform_3d import (
    LidarReformat,
    ObjectNoise,
    ObjectRangeFilter,
    ObjectSample,
    PointGlobalRotation,
    PointGlobalScaling,
    PointRandomFlip,
    ShufflePoints,
)
from hat.data.transforms.lidar_utils.sample_ops import DataBaseSampler
from hat.engine.ddp_trainer import DistributedDataParallelTrainer
from hat.engine.ddp_trainer import launch as ddp_launch
from hat.engine.predictor import Predictor
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.engine.processors.processor import MultiBatchProcessor
from hat.metrics.kitti3d_detection import Kitti3DMetricDet
from hat.metrics.loss_show import LossShow
from hat.models.ir_modules.hbir_module import HbirModule
from hat.models.losses.cross_entropy_loss import CrossEntropyLoss
from hat.models.losses.focal_loss import FocalLossV2
from hat.models.losses.smooth_l1_loss import SmoothL1Loss
from hat.models.model_convert.ckpt_converters import LoadCheckpoint
from hat.models.model_convert.converters import Float2QAT
from hat.models.necks.second_neck import SECONDNeck
from hat.models.structures.detectors.pointpillars import (
    PointPillarsDetector,
    PointPillarsDetectorIrInfer,
)
from hat.models.task_modules.lidar import (
    Anchor3DGeneratorStride,
    GroundBox3dCoder,
    LidarTargetAssigner,
)
from hat.models.task_modules.lidar.pillar_encoder import (
    PillarFeatureNet,
    PointPillarScatter,
)
from hat.models.task_modules.pointpillars import (
    PointPillarsHead,
    PointPillarsLoss,
    PointPillarsPostProcess,
    PointPillarsPreProcess,
)
from hat.utils.distributed import get_dist_info
from hat.utils.logger import DisableLogger, init_rank_logger, rank_zero_info
from hat.utils.seed import seed_training
from hat.utils.setup_env import setup_hat_env
from hat.utils.thread_init import init_num_threads

AVAILABLE_STAGE = ["float", "qat", "int_infer"]


class PointPillarsModel:

    task_name = "pointpillars_kitti_car"

    @classmethod
    def model(cls):
        model = cls._build_pp_model(cls, is_deploy=False)
        return model

    @classmethod
    def deploy_model(cls):
        deploy_model = cls._build_pp_model(cls, is_deploy=True)
        return deploy_model

    @classmethod
    def hbir_model(cls, ckpt):
        hbir_model = cls._build_pp_model(
            cls, is_hbir_model=True, hbir_path=ckpt
        )
        return hbir_model

    @classmethod
    def deploy_inputs(cls):
        deploy_inputs = dict(  # noqa C408
            points=[
                torch.randn(150000, 4),
            ],
        )
        return deploy_inputs

    @classmethod
    def float_model(cls, pretrain_ckpt=None, use_deploy=False):
        if use_deploy:
            model = cls.deploy_model()  # float model
        else:
            model = cls.model()
        if pretrain_ckpt:
            ckpt_loader = LoadCheckpoint(
                pretrain_ckpt,
                allow_miss=False,
                ignore_extra=False,
                verbose=True,
            )
            model = ckpt_loader(model)

        return model

    @classmethod
    def qat_model(
        cls, pre_step_ckpt=None, pretrain_ckpt=None, use_deploy=False
    ):

        float_model = cls.float_model(pre_step_ckpt, use_deploy=use_deploy)
        qat_model = Float2QAT()(float_model)

        if pretrain_ckpt:
            ckpt_loader = LoadCheckpoint(
                pretrain_ckpt,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            )
            qat_model = ckpt_loader(qat_model)

        return qat_model

    @classmethod
    def int_model(
        cls,
        hbir_ckpt,
    ):
        model = cls.hbir_model(ckpt=hbir_ckpt)
        return model

    def _build_pp_model(
        self, is_deploy=False, is_hbir_model=False, hbir_path=None
    ):
        def get_feature_map_size(point_cloud_range, voxel_size):
            point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
            voxel_size = np.array(voxel_size, dtype=np.float32)
            grid_size = (
                point_cloud_range[3:] - point_cloud_range[:3]
            ) / voxel_size
            grid_size = np.round(grid_size).astype(np.int64)
            return grid_size

        # Voxelization cfg
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size = [0.16, 0.16, 4.0]
        max_points_in_voxel = 100
        max_voxels_num = 12000
        class_names = ["Car"]
        if not is_hbir_model:
            model = PointPillarsDetector(
                feature_map_shape=get_feature_map_size(pc_range, voxel_size),
                is_deploy=is_deploy,
                pre_process=PointPillarsPreProcess(
                    pc_range=pc_range,
                    voxel_size=voxel_size,
                    max_voxels_num=max_voxels_num,
                    max_points_in_voxel=max_points_in_voxel,
                ),
                reader=PillarFeatureNet(
                    num_input_features=4,
                    num_filters=(64,),
                    with_distance=False,
                    pool_size=(1, max_points_in_voxel),
                    voxel_size=voxel_size,
                    pc_range=pc_range,
                    bn_kwargs=None,
                    quantize=True,
                    use_4dim=True,
                    use_conv=True,
                ),
                backbone=PointPillarScatter(
                    num_input_features=64,
                    use_horizon_pillar_scatter=True,
                    quantize=True,
                ),
                neck=SECONDNeck(
                    in_feature_channel=64,
                    down_layer_nums=[3, 5, 5],
                    down_layer_strides=[2, 2, 2],
                    down_layer_channels=[64, 128, 256],
                    up_layer_strides=[1, 2, 4],
                    up_layer_channels=[128, 128, 128],
                    bn_kwargs=None,
                    quantize=True,
                ),
                head=PointPillarsHead(
                    num_classes=len(class_names),
                    in_channels=sum([128, 128, 128]),
                    use_direction_classifier=True,
                ),
                anchor_generator=Anchor3DGeneratorStride(
                    anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
                    anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
                    anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
                    rotations=[[0, 1.57]],  # noqa B006
                    class_names=class_names,
                    match_thresholds=[0.6],
                    unmatch_thresholds=[0.45],
                ),
                targets=LidarTargetAssigner(
                    box_coder=GroundBox3dCoder(n_dim=7),
                    class_names=class_names,
                    positive_fraction=-1,
                ),
                loss=PointPillarsLoss(
                    num_classes=len(class_names),
                    loss_cls=FocalLossV2(
                        alpha=0.25,
                        gamma=2.0,
                        from_logits=False,
                        reduction="none",
                        loss_weight=1.0,
                    ),
                    loss_bbox=SmoothL1Loss(
                        beta=1 / 9.0,
                        reduction="none",
                        loss_weight=2.0,
                    ),
                    loss_dir=CrossEntropyLoss(
                        use_sigmoid=False,
                        reduction="none",
                        loss_weight=0.2,
                    ),
                ),
                postprocess=PointPillarsPostProcess(
                    num_classes=len(class_names),
                    box_coder=GroundBox3dCoder(n_dim=7),
                    use_direction_classifier=True,
                    num_direction_bins=2,
                    # test_cfg
                    use_rotate_nms=False,
                    nms_pre_max_size=1000,
                    nms_post_max_size=300,
                    nms_iou_threshold=0.5,
                    score_threshold=0.4,
                    post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
                    max_per_img=100,
                ),
            )

            if is_deploy:
                model.anchor_generator = None
                model.targets = None
                model.loss = None
                model.postprocess = None
        else:
            assert hbir_path
            model = PointPillarsDetectorIrInfer(
                ir_model=HbirModule(hbir_path),
                anchor_generator=Anchor3DGeneratorStride(
                    anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
                    anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
                    anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
                    rotations=[[0, 1.57]],  # noqa B006
                    class_names=class_names,
                    match_thresholds=[0.6],
                    unmatch_thresholds=[0.45],
                ),
                postprocess=PointPillarsPostProcess(
                    num_classes=len(class_names),
                    box_coder=GroundBox3dCoder(n_dim=7),
                    use_direction_classifier=True,
                    num_direction_bins=2,
                    # test_cfg
                    use_rotate_nms=False,
                    nms_pre_max_size=1000,
                    nms_post_max_size=300,
                    nms_iou_threshold=0.5,
                    score_threshold=0.4,
                    post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
                    max_per_img=100,
                ),
            )
        return model

    @classmethod
    def optimizer(cls, model, stage):
        if stage == "float":
            return torch.optim.AdamW(
                params=model.parameters(),
                betas=(0.95, 0.99),
                lr=2e-4,
                weight_decay=0.01,
            )
        elif stage == "qat":
            return torch.optim.SGD(
                params=model.parameters(),
                lr=2e-4,
                momentum=0.9,
                weight_decay=0.0,
            )

        return None

    @classmethod
    def lr_schedule(cls, stage):
        if stage == "float":
            return CyclicLrUpdater(
                target_ratio=(10, 1e-4),
                cyclic_times=1,
                step_ratio_up=0.4,
                step_log_interval=50,
            )

        elif stage == "qat":
            return CyclicLrUpdater(
                target_ratio=(10, 1e-4),
                cyclic_times=1,
                step_ratio_up=0.4,
                step_log_interval=50,
            )

        return None

    @classmethod
    def train_metrics(cls):
        return LossShow()

    @classmethod
    def val_metrics(cls):
        class_names = ["Car"]
        val_metrics = Kitti3DMetricDet(
            compute_aos=True,
            current_classes=class_names,
            difficultys=[0, 1, 2],
        )
        return val_metrics


class DataHelper:

    data_dir = "./tmp_data/kitti3d"
    train_batch_size = 2
    val_batch_size = 1

    @classmethod
    def train_data_loader(cls):
        return cls.build_dataloader(cls, is_training=True)

    @classmethod
    def val_data_loader(cls, use_distributed=True):
        return cls.build_dataloader(
            cls,
            is_training=False,
            use_distributed=use_distributed,
        )

    def build_dataloader(self, is_training=True, use_distributed=True):

        transforms = self.build_transforms(self, self.data_dir, is_training)

        split_dir = "train_lmdb" if is_training else "val_lmdb"
        dataset = Kitti3D(
            data_path=os.path.join(self.data_dir, split_dir),
            transforms=transforms,
        )

        sampler = (
            torch.utils.data.DistributedSampler(dataset)
            if use_distributed
            else None
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.train_batch_size
            if is_training
            else self.val_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_lidar3d,
        )

        return dataloader

    def build_transforms(self, data_dir, is_training=True):

        class_names = ["Car"]
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]

        if is_training:
            transforms = torchvision.transforms.Compose(
                [
                    ObjectSample(
                        class_names=class_names,
                        remove_points_after_sample=False,
                        db_sampler=DataBaseSampler(
                            enable=True,
                            root_path="./tmp_data/kitti3d/",
                            db_info_path="./tmp_data/kitti3d/kitti3d_dbinfos_train.pkl",  # noqa E501
                            sample_groups=[dict(Car=15)],  # noqa C408
                            db_prep_steps=[  # noqa C408
                                dict(  # noqa C408
                                    type="DBFilterByDifficulty",
                                    filter_by_difficulty=[-1],
                                ),
                                dict(  # noqa C408
                                    type="DBFilterByMinNumPoint",
                                    filter_by_min_num_points=dict(  # noqa C408
                                        Car=5,
                                    ),
                                ),
                            ],
                            global_random_rotation_range_per_object=[0, 0],
                            rate=1.0,
                        ),
                    ),
                    ObjectNoise(
                        gt_rotation_noise=[-0.15707963267, 0.15707963267],
                        gt_loc_noise_std=[0.25, 0.25, 0.25],
                        global_random_rot_range=[0, 0],
                        num_try=100,
                    ),
                    PointRandomFlip(probability=0.5),
                    PointGlobalRotation(rotation=[-0.78539816, 0.78539816]),
                    PointGlobalScaling(min_scale=0.95, max_scale=1.05),
                    ShufflePoints(True),
                    ObjectRangeFilter(point_cloud_range=pc_range),
                    LidarReformat(),
                ]
            )
        else:
            transforms = torchvision.transforms.Compose([LidarReformat()])
        return transforms


def train_entrance(
    device,
    task_name,
    model,
    stage,
    march,
    num_epochs=1,
    seed=None,
    use_amp=False,
    log_rank_zero_only=True,
    cudnn_benchmark=True,
    out_dir: str = "./tmp_models",
):

    rank, _ = get_dist_info()
    disable_logger = rank != 0 and log_rank_zero_only
    # 1. init logger
    init_rank_logger(
        rank,
        save_dir=out_dir,
        cfg_file=task_name,
        step=stage,
        prefix="train-",
    )

    # 2. init num threads
    init_num_threads()

    # 3. seed training
    cudnn.benchmark = cudnn_benchmark
    if seed is not None:
        seed_training(seed)

    horizon.march.set_march(march)

    rank_zero_info("=" * 50 + f"BEGIN {stage.upper()} STAGE " + "TRAINING")

    # 4. build and run trainer
    with DisableLogger(disable_logger, logging.WARNING):

        # callbacks
        ckpt_callback = Checkpoint(
            save_dir=out_dir,
            name_prefix=stage + "-",
            strict_match=True,
            mode="max",
            monitor_metric_key="mAP_3D_moderate",
        )

        def update_loss(metrics, batch, model_outs):
            for metric in metrics:
                metric.update(model_outs)

        loss_updater = MetricUpdater(
            metric_update_func=update_loss,
            step_log_freq=50,
            epoch_log_freq=1,
            log_prefix="Loss",
        )

        def update_metric(metrics, batch, model_outs):
            for metric in metrics:
                metric.update(model_outs, batch)

        metric_updater = MetricUpdater(
            metric_update_func=update_metric,
            step_log_freq=10000,
            epoch_log_freq=1,
            log_prefix="Validation_" + task_name,
        )

        callbacks = [
            StatsMonitor(log_freq=50),
            loss_updater,
            PointPillarsModel.lr_schedule(stage),
            Validation(
                data_loader=DataHelper.val_data_loader(),
                batch_processor=MultiBatchProcessor(
                    need_grad_update=False,
                    loss_collector=None,
                ),
                callbacks=[metric_updater],
                val_model=None,
                val_on_train_end=True,
                val_interval=1,
                log_interval=200,
            ),
            ckpt_callback,
        ]

        trainer = DistributedDataParallelTrainer(  # noqa C408
            model=model,
            data_loader=DataHelper.train_data_loader(),
            optimizer=PointPillarsModel.optimizer(model, stage),
            batch_processor=MultiBatchProcessor(
                need_grad_update=True,
                enable_amp=use_amp,
                loss_collector=collect_loss_by_index(0),
            ),
            num_epochs=num_epochs,
            device=device,
            callbacks=callbacks,
            sync_bn=True,
            train_metrics=PointPillarsModel.train_metrics(),
            val_metrics=PointPillarsModel.val_metrics(),
        )

        trainer.fit()

    rank_zero_info("=" * 50 + f"END {stage.upper()} STAGE " + "TRAINING")


def train(
    model,
    stage,
    march,
    task_name,
    device_ids=None,
    total_epochs=1,
    use_amp=False,
    seed=None,
    log_rank_zero_only=True,
    cudnn_benchmark=True,
    out_dir="./tmp_models",
    dist_url="auto",
    dist_launcher=None,
):
    assert stage in AVAILABLE_STAGE

    setup_hat_env(stage)
    ids = list(map(int, device_ids.split(",")))

    ddp_launch(
        train_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=dist_launcher,
        args=(
            task_name,
            model,
            stage,
            march,
            total_epochs,
            seed,
            use_amp,
            log_rank_zero_only,
            cudnn_benchmark,
            out_dir,
        ),
    )


def predict_entrance(
    device,
    task_name,
    ckpt,
    stage,
    march,
    log_rank_zero_only=True,
    cudnn_benchmark=True,
    out_dir="./tmp_models",
):

    rank, _ = get_dist_info()
    disable_logger = rank != 0 and log_rank_zero_only
    # init logger
    init_rank_logger(
        rank,
        save_dir=out_dir,
        cfg_file=task_name,
        step=stage,
        prefix="prediction-",
    )

    cudnn.benchmark = cudnn_benchmark

    horizon.march.set_march(march)

    if stage == "float":
        model = PointPillarsModel.float_model(pretrain_ckpt=ckpt)
    elif stage == "qat":
        model = PointPillarsModel.qat_model(pretrain_ckpt=ckpt)
    elif stage == "int_infer":
        model = PointPillarsModel.int_model(hbir_ckpt=ckpt)
    else:
        raise ValueError(
            f"`stage` should be one of ['float', 'qat', 'int_infer'],"
            f"but get {stage}."
        )

    rank_zero_info(
        "=" * 50 + f"BEGIN {stage.upper()} STAGE PREDICTION" + "=" * 50
    )

    # 4. build and run trainer
    with DisableLogger(disable_logger, logging.WARNING):

        # callbacks
        def update_metric(metrics, batch, model_outs):
            for metric in metrics:
                metric.update(model_outs, batch)

        metric_updater = MetricUpdater(
            metric_update_func=update_metric,
            step_log_freq=10000,
            epoch_log_freq=1,
            log_prefix="Validation_" + task_name,
        )

        predictor = Predictor(
            model=model,
            data_loader=[DataHelper.val_data_loader(use_distributed=False)],
            batch_processor=MultiBatchProcessor(
                need_grad_update=False,
                loss_collector=collect_loss_by_index(0),
            ),
            device=device,
            metrics=PointPillarsModel.val_metrics(),
            callbacks=[metric_updater],
            log_interval=200,
        )
        predictor.fit()

    rank_zero_info(
        "=" * 50 + f"END {stage.upper()} STAGE PREDICTION" + "=" * 50
    )


def predict(
    ckpt,
    stage,
    march,
    task_name=None,
    device_ids=None,
    log_rank_zero_only=True,
    cudnn_benchmark=True,
    out_dir="./tmp_models",
    dist_url="auto",
    dist_launcher=None,
):
    assert stage in AVAILABLE_STAGE

    setup_hat_env(stage)
    ids = list(map(int, device_ids.split(",")))
    ddp_launch(
        predict_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=dist_launcher,
        args=(
            task_name,
            ckpt,
            stage,
            march,
            log_rank_zero_only,
            cudnn_benchmark,
            out_dir,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        default=None,
        required=False,
        help=("the training stages, multi values will be splitted by `,`"),
    )

    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default="0,1,2,3,4,5,6,7",
        help="GPU device ids like '0,1,2,3'",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        required=False,
        default=False,
        help="do train pipeline",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        required=False,
        default=False,
        help="Use AMP",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        required=False,
        default=False,
        help="do validation pipeline",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path used to predict",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        required=False,
        default=False,
        help="check model",
    )
    parser.add_argument(
        "--calops",
        action="store_true",
        required=False,
        default=False,
        help="Calculate ops of torch model.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        required=False,
        default=False,
        help="do compile hbm",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        default=None,
        help="compile output dir`",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=False,
        default="./tmp_models",
        help="checkpoint root dir`",
    )
    parser.add_argument(
        "--opt",
        type=str,
        required=False,
        default="O2",
        help="optimization options",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        required=False,
        default=4,
        help="number of threads launched during compiler optimization."
        " Default 0 means to use all available hardware concurrency.",
    )
    parser.add_argument(
        "--align-bpu-validation",
        action="store_true",
        required=False,
        default=False,
        help="align bpu validation",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        required=False,
        default=False,
        help="infer and viz",
    )

    parser.add_argument(
        "--input-points",
        type=str,
        required=False,
        default=None,
        help="input lidar cloudpoints for infer and viz",
    )
    parser.add_argument(
        "--is-plot",
        action="store_true",
        required=False,
        default=False,
        help="plot infer result",
    )
    parser.add_argument(
        "--export-hbir",
        action="store_true",
        required=False,
        default=False,
        help="export hbir",
    )
    parser.add_argument(
        "--march",
        type=str,
        required=False,
        default=horizon.march.March.NASH_E,
        help="march of bpu",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def main():
    args, args_env = parse_args()

    ckpt_dir = os.path.join(args.ckpt_dir, PointPillarsModel.task_name)

    horizon.march.set_march(args.march)

    if args.train:
        stage = args.stage
        assert stage in AVAILABLE_STAGE
        if stage == "float":
            model = PointPillarsModel.float_model()
            total_epochs = 160
        elif stage == "qat":
            pre_step_ckpt = os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            )
            model = PointPillarsModel.qat_model(pre_step_ckpt=pre_step_ckpt)
            total_epochs = 50
        else:
            raise ValueError(
                f"`stage` should be one of ['float', 'qat'],"
                f"but get {stage}."
            )

        train(
            model,
            stage,
            march=args.march,
            task_name=PointPillarsModel.task_name,
            device_ids=args.device_ids,
            out_dir=ckpt_dir,
            use_amp=args.amp,
            total_epochs=total_epochs,
        )

    if args.predict:
        stage = args.stage
        assert stage in AVAILABLE_STAGE
        assert args.ckpt
        predict(
            args.ckpt,
            stage,
            march=args.march,
            task_name=PointPillarsModel.task_name,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
        )

    if args.check_model:
        model_checker(
            deploy_model=PointPillarsModel.qat_model(use_deploy=True),
            deploy_inputs=PointPillarsModel.deploy_inputs(),
            march=args.march,
        )

    if args.calops:
        calops(
            deploy_model=PointPillarsModel.deploy_model(),
            deploy_inputs=PointPillarsModel.deploy_inputs(),
        )

    if args.align_bpu_validation:
        assert args.ckpt, "Must pass in the ckpt parameter"

        def update_metric(metrics, batch, model_outs):
            for metric in metrics:
                metric.update(model_outs, batch)

        metric_updater = MetricUpdater(
            metric_update_func=update_metric,
            step_log_freq=100000,
            epoch_log_freq=1,
            log_prefix="Validation " + PointPillarsModel.task_name,
        )
        val_callback = Validation(
            data_loader=DataHelper.val_data_loader(use_distributed=False),
            batch_processor=MultiBatchProcessor(
                need_grad_update=False,
            ),
            callbacks=[metric_updater],
            log_interval=1,
        )

        pre_step_ckpt = os.path.join(ckpt_dir, "qat-checkpoint-best.pth.tar")

        int_model = PointPillarsModel.int_model(
            hbir_ckpt=args.ckpt,
        )

        align_bpu_validation(
            int_model=int_model,
            data_loader=DataHelper.val_data_loader(use_distributed=False),
            val_batch_processor=MultiBatchProcessor(
                need_grad_update=False,
            ),
            val_metrics=PointPillarsModel.val_metrics(),
            val_callbacks=val_callback,
            ckpt_dir=ckpt_dir,
            device=args.device_ids,
        )

    if args.export_hbir:
        pre_step_ckpt = os.path.join(ckpt_dir, "qat-checkpoint-best.pth.tar")
        qat_model = PointPillarsModel.qat_model(
            pretrain_ckpt=pre_step_ckpt, use_deploy=True
        )
        export_hbir(
            qat_model,
            PointPillarsModel.deploy_inputs(),
            task_name=PointPillarsModel.task_name,
            march=args.march,
            output_dir=args.out_dir
            if args.out_dir
            else os.path.join(ckpt_dir, "compile"),
        )

    if args.compile:
        assert args.ckpt, "Must pass in the ckpt parameter"
        compile_perf(
            model_path=args.ckpt,
            march=args.march,
            opt=args.opt,
            out_path=args.out_dir
            if args.out_dir
            else os.path.join(ckpt_dir, "compile", "model.hbm"),
        )

    if args.visualize:
        assert args.input_points, "Must pass in the input_points parameter"
        assert args.ckpt, "Must pass in the ckpt parameter"
        int_model = PointPillarsModel.int_model(
            hbir_ckpt=args.ckpt,
        )
        int_infer_viz_lidar(
            model=int_model,
            input_points=args.input_points,
            device=args.device_ids,
            is_plot=args.is_plot,
        )
        pass


if __name__ == "__main__":
    main()
