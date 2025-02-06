import argparse
import logging
import os
import signal
from typing import Any

import horizon_plugin_pytorch as horizon
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from pipeline_tools import (
    calops,
    compile,
    compile_perf,
    export_hbir,
    int_infer_viz_lidar,
    model_checker,
)
from pointpillars import DataHelper, PointPillarsModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from hat.utils.apply_func import to_cuda

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)

AVAILABLE_STAGE = ["float", "qat", "int_infer"]


def setup_envs():
    os.environ["TORCH_NUM_THREADS"] = os.environ.get("TORCH_NUM_THREADS", "12")
    os.environ["OPENCV_NUM_THREADS"] = os.environ.get(
        "OPENCV_NUM_THREADS", "12"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get(
        "OPENBLAS_NUM_THREADS", "12"
    )
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "12")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "12")
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"


def align_bpu_validation(int_model, val_dataloader, val_metrics, device_ids):
    device = device_ids.split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    eval(
        model=int_model,
        val_dataloader=val_dataloader,
        val_metrics=val_metrics,
        device=torch.device("cuda"),
    )


def eval(model, val_dataloader, val_metrics, device):
    model.eval()
    model.to(device)

    for batch_data in tqdm.tqdm(val_dataloader):
        batch_data = to_cuda(batch_data, device=device)
        pred_outs = model(batch_data)
        val_metrics.update(pred_outs, batch_data)

    metric_names, metric_values = val_metrics.get()

    log_info = "\n"
    for name, value in zip(metric_names, metric_values):
        if isinstance(value, (int, float)):
            log_info += "%s[%.4f] " % (name, value)
        else:
            log_info += "%s[%s] " % (str(name), str(value))
    log_info += "\n"
    logger.info(log_info)

    return metric_values[0]


class Trainer:
    """Trainer.
    reference: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py.  # noqa
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stage: str,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_schedule: Any,
        gpu_id: int,
        save_every: int = 1,
        log_freq: int = 20,
        output_dir: str = "./tmp_models",
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.stage = stage
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.lr_schedule = lr_schedule
        self.output_dir = output_dir
        self.log_freq = log_freq

    def _run_epoch(self, epoch):
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        for batch_id, batch_data in enumerate(self.train_data):
            batch_data = to_cuda(batch_data, device=self.gpu_id)

            # one batch
            self.optimizer.zero_grad()
            self.lr_schedule.on_step_begin(
                optimizer=self.optimizer,
                epoch_id=epoch,
                step_id=batch_id,
                global_step_id=self.global_step_id,
            )
            output = self.model(batch_data)
            loss = sum(list(output.values()))
            loss.backward()
            self.optimizer.step()

            # log
            if (batch_id + 1) % self.log_freq == 0:
                loss_str = ""
                for k, v in output.items():
                    loss_str += f" {k} [{round(v.item(), 4)}]"
                loss_log = "Epoch[%d] Step[%d] GlobalStep[%d]: %s" % (
                    epoch,
                    batch_id,
                    self.global_step_id,
                    loss_str,
                )

                # only log on gpu 0
                if self.gpu_id == 0:
                    logger.info(loss_log)

            self.global_step_id += 1

    def _save_checkpoint(self, epoch):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        state = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict(),
        }
        ckpt_file = os.path.join(
            self.output_dir, f"{self.stage}-checkpoint-best.pth.tar"
        )
        torch.save(state, ckpt_file)
        logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {ckpt_file}"
        )

    def train(self, max_epochs: int):
        self.global_step_id = 0
        self.lr_schedule.on_loop_begin(
            self.optimizer,
            self.train_data,
            max_epochs,
        )
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            # validation and save checkpoint
            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == max_epochs:
                # validation
                eval(
                    self.model,
                    DataHelper.val_data_loader(use_distributed=True),
                    PointPillarsModel.val_metrics(),
                    self.gpu_id,
                )
                if self.gpu_id == 0:
                    # save checkpoint
                    self._save_checkpoint(epoch)


def train_entrance(
    rank: int,
    world_size: int,
    model,
    stage,
    march,
    output_dir,
    total_epochs,
):
    # set distribute
    dist.init_process_group(
        backend="NCCL",
        init_method="tcp://localhost:%s" % "12345",
        world_size=world_size,
        rank=rank,
    )

    horizon.march.set_march(march)

    train_data = DataHelper.train_data_loader()
    optimizer = PointPillarsModel.optimizer(model, stage)
    lr_schedule = PointPillarsModel.lr_schedule(stage)
    trainer = Trainer(
        model,
        stage,
        train_data,
        optimizer,
        lr_schedule,
        rank,
        output_dir=output_dir,
    )
    trainer.train(total_epochs)
    dist.destroy_process_group()


def train(
    model,
    stage,
    march,
    device_ids,
    output_dir,
    total_epochs,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    device_ids = device_ids.split(",")
    world_size = len(device_ids)
    try:
        mp.spawn(
            train_entrance,
            args=(
                world_size,
                model,
                stage,
                march,
                output_dir,
                total_epochs,
            ),
            nprocs=world_size,
        )
    # when press Ctrl+c, all sub processes will exits too.
    except KeyboardInterrupt as exception:
        logger.exception(str(exception))
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def predict(
    model,
    val_dataloader,
    val_metrics,
    device_ids,
    cudnn_benchmark=True,
):
    device = device_ids.split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    setup_envs()
    cudnn.benchmark = cudnn_benchmark
    eval(model, val_dataloader, val_metrics, device=torch.device("cuda"))


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
            device_ids=args.device_ids,
            output_dir=ckpt_dir,
            total_epochs=total_epochs,
        )

    if args.predict:
        stage = args.stage
        assert args.ckpt
        if stage == "float":
            model = PointPillarsModel.float_model(args.ckpt)
        elif stage == "qat":
            model = PointPillarsModel.qat_model(pretrain_ckpt=args.ckpt)
        elif stage == "int_infer":
            model = PointPillarsModel.int_model(hbir_ckpt=args.ckpt)
        else:
            raise ValueError(
                f"`stage` should be one of ['float', 'qat', 'int_infer'],"
                f"but get {stage}."
            )
        predict(
            model=model,
            val_dataloader=DataHelper.val_data_loader(use_distributed=False),
            val_metrics=PointPillarsModel.val_metrics(),
            device_ids=args.device_ids,
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
        int_model = PointPillarsModel.int_model(
            hbir_ckpt=args.ckpt,
        )

        align_bpu_validation(
            int_model=int_model,
            val_dataloader=DataHelper.val_data_loader(use_distributed=False),
            val_metrics=PointPillarsModel.val_metrics(),
            device_ids=args.device_ids,
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
            is_plot=args.is_plot,
            device=args.device_ids,
        )


if __name__ == "__main__":
    main()
