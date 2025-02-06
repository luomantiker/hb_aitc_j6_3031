import argparse
import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from hbdk4.compiler import visualize
from torch import Tensor
from torch.utils import data
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.datasets import CIFAR10
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from horizon_plugin_pytorch import get_march
from horizon_plugin_pytorch.quantization import FakeQuantState
from horizon_plugin_pytorch.quantization import set_fake_quantize

##############################################################################
# Example input for model tracing and hbir exporting.
##############################################################################


example_input = torch.rand(1, 3, 32, 32)


##############################################################################
# Hbir module wrapper to deal with in-out structures.
##############################################################################


class HbirModule(nn.Module):
    def __init__(self, hbir_module) -> None:
        super().__init__()
        self.hbir_module = hbir_module

    def forward(self, *inputs):
        rets = self.hbir_module.functions[0](inputs)
        return rets


##############################################################################
# Some common helper functions.
##############################################################################


def get_args():
    parser = argparse.ArgumentParser(description="Run mobilnet example.")
    parser.add_argument(
        "--stage",
        type=str,
        choices=("float", "calib", "qat", "int_infer", "compile", "visualize"),
        help=(
            "Pipeline stage, must be executed in following order: "
            "float -> calib(optional) -> qat(optional) -> int_infer -> compile"
        ),
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help=(
            "Path to the cifar-10 dataset. If cannot access network, please "
            "download dataset from <{}>, and put it under this "
            "path.".format(CIFAR10.url)
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/mobilenetv2",
        help=(
            "Where to save the model and other results. If cannot access "
            "network, please download pretrained model from <{}>, and put it "
            "under this path.".format(MobileNet_V2_Weights.IMAGENET1K_V1.url)
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=None,
        help=(
            "Rewrite the default training epoch number, pass 0 to skip "
            "training and only do evaluation (in stage 'float' or 'qat')"
        ),
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Specify which device to use, pass a negative value to use cpu",
    )
    parser.add_argument(
        "--opt",
        type=str,
        choices=["0", "1", "2"],
        default=0,
        help="Specity optimization level for compilation",
    )
    args = parser.parse_args()
    return args


def load_pretrain(model: nn.Module, model_path: str):
    state_dict = load_state_dict_from_url(
        MobileNet_V2_Weights.IMAGENET1K_V1.url,
        model_dir=model_path,
        progress=True,
    )

    # because num_classes is different, ignore the weight of last layer
    ignore_keys = []
    for k in state_dict:
        if "classifier" in k:
            ignore_keys.append(k)
    for k in ignore_keys:
        state_dict.pop(k)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )
    assert len(missing_keys) == 2
    assert len(unexpected_keys) == 0

    return model


def prepare_data_loaders(
    data_path: str, train_batch_size: int, eval_batch_size: int
) -> Tuple[data.DataLoader, data.DataLoader]:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_dataset = CIFAR10(
        data_path,
        True,
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    eval_dataset = CIFAR10(
        data_path,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=data.RandomSampler(train_dataset),
        num_workers=8,
        pin_memory=True,
    )

    eval_data_loader = data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=data.SequentialSampler(eval_dataset),
        num_workers=8,
        pin_memory=True,
    )

    return train_data_loader, eval_data_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    data_loader: data.DataLoader,
    device: torch.device,
) -> None:
    top1 = AverageMeter("Acc@1", ":6.3f")
    top5 = AverageMeter("Acc@5", ":6.3f")
    avgloss = AverageMeter("Loss", ":1.5f")

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1, image.size(0))
        top5.update(acc5, image.size(0))
        avgloss.update(loss, image.size(0))
        print(".", end="", flush=True)
    print()

    print(
        "Full cifar-10 train set: Loss {:.3f} Acc@1"
        " {:.3f} Acc@5 {:.3f}".format(avgloss.avg, top1.avg, top5.avg)
    )


def evaluate(
    model: nn.Module, data_loader: data.DataLoader, device: torch.device
) -> Tuple[AverageMeter, AverageMeter]:
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1, image.size(0))
            top5.update(acc5, image.size(0))
            print(".", end="", flush=True)
        print()

    return top1, top5


##############################################################################
# Define the logic of each step in the pipeline.
##############################################################################


def calibrate(
    calib_model,
    data_path,
    model_path,
    calib_batch_size,
    eval_batch_size,
    device,
    num_examples=float("inf"),
):
    # Please note that calibration need the model in eval mode
    # to make BatchNorm act properly.
    calib_model.eval()
    set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)

    train_data_loader, eval_data_loader = prepare_data_loaders(
        data_path, calib_batch_size, eval_batch_size
    )

    with torch.no_grad():
        cnt = 0
        for image, target in train_data_loader:
            image, target = image.to(device), target.to(device)
            calib_model(image)
            print(".", end="", flush=True)
            cnt += image.size(0)
            if cnt >= num_examples:
                break
        print()

    # Must set eval mode again before validation, because
    # set CALIBRATION state will make FakeQuantize in training mode.
    calib_model.eval()
    set_fake_quantize(calib_model, FakeQuantState.VALIDATION)

    top1, top5 = evaluate(
        calib_model,
        eval_data_loader,
        device,
    )
    print(
        "Calibration: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
            top1.avg, top5.avg
        )
    )

    torch.save(
        calib_model.state_dict(),
        os.path.join(model_path, "calib-checkpoint.ckpt"),
    )

    return calib_model


# Float and qat share the same training procedure.
def train(
    model: nn.Module,
    data_path: str,
    model_path: str,
    train_batch_size: int,
    eval_batch_size: int,
    epoch_num: int,
    device: torch.device,
    optim_config: Callable,
    stage: str,
):
    train_data_loader, eval_data_loader = prepare_data_loaders(
        data_path, train_batch_size, eval_batch_size
    )

    optimizer, scheduler = optim_config(model)

    best_acc = 0

    for nepoch in range(epoch_num):
        # Training/Eval state must be setted correctly
        # before `set_fake_quantize`
        model.train()
        if stage == "qat":
            set_fake_quantize(model, FakeQuantState.QAT)

        train_one_epoch(
            model,
            nn.CrossEntropyLoss(),
            optimizer,
            scheduler,
            train_data_loader,
            device,
        )

        model.eval()
        if stage == "qat":
            set_fake_quantize(model, FakeQuantState.VALIDATION)

        top1, top5 = evaluate(
            model,
            eval_data_loader,
            device,
        )
        print(
            "{} Epoch {}: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
                stage.capitalize(), nepoch, top1.avg, top5.avg
            )
        )

        if top1.avg > best_acc:
            best_acc = top1.avg

            torch.save(
                model.state_dict(),
                os.path.join(model_path, "{}-checkpoint.ckpt".format(stage)),
            )

    if nepoch == 0:
        model.eval()
        if stage == "qat":
            set_fake_quantize(model, FakeQuantState.VALIDATION)

        top1, top5 = evaluate(
            model,
            eval_data_loader,
            device,
        )
        print(
            "{} eval only: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
                stage.capitalize(), top1.avg, top5.avg
            )
        )

    print("Best Acc@1 {:.3f}".format(best_acc))

    return model


def int_infer(
    quantized_model,
    data_path,
    eval_batch_size,
    device,
):
    # hbir do not support dynamic batch size or cuda
    eval_batch_size = 1
    device = torch.device("cpu")

    quantized_model = HbirModule(quantized_model)

    _, eval_data_loader = prepare_data_loaders(
        data_path, eval_batch_size, eval_batch_size
    )

    top1, top5 = evaluate(
        quantized_model,
        eval_data_loader,
        device,
    )
    print(
        "Quantized: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
            top1.avg, top5.avg
        )
    )

    return quantized_model


def compile(
    quantized_model,
    model_path,
    compile_opt=0,
):
    from hbdk4.compiler import compile, hbm_perf

    hbm_path = os.path.join(model_path, "model.hbm")
    perf_dir = os.path.join(model_path, "perf")
    os.makedirs(perf_dir, exist_ok=True)

    compile(quantized_model, hbm_path, get_march(), compile_opt)
    hbm_perf(hbm_path, perf_dir)


##############################################################################
# Program entrance.
##############################################################################


def main(
    model: nn.Module,
    stage: str,
    data_path: str,
    model_path: str,
    train_batch_size: int,
    eval_batch_size: int,
    epoch_num: int,
    device_id: int = 0,
    compile_opt: int = 0,
):
    assert stage in (
        "float",
        "calib",
        "qat",
        "int_infer",
        "compile",
        "visualize",
    )

    # Specify random seed for repeatable results
    torch.manual_seed(191009)

    device = torch.device(
        "cuda:{}".format(device_id) if device_id >= 0 else "cpu"
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    def float_optim_config(model: nn.Module):
        # This is an example to illustrate the usage of QAT training tool, so
        # we do not fine tune the training hyper params to get optimized
        # float model accuracy.
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=2e-4)

        return optimizer, None

    def qat_optim_config(model: nn.Module):
        # QAT training is targeted at fine tuning model params to match the
        # numerical quantization, so the learning rate should not be too large.
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.0001, weight_decay=2e-4
        )

        return optimizer, None

    default_epoch_num = {
        "float": 30,
        "qat": 3,
    }

    if stage in ("float", "qat"):
        if epoch_num is None:
            epoch_num = default_epoch_num[stage]
        train(
            model,
            data_path,
            model_path,
            train_batch_size,
            eval_batch_size,
            epoch_num,
            device,
            float_optim_config if stage == "float" else qat_optim_config,
            stage,
        )
    elif stage == "calib":
        calibrate(
            model,
            data_path,
            model_path,
            train_batch_size,
            eval_batch_size,
            device,
        )
    elif stage == "int_infer":
        int_infer(
            model,
            data_path,
            eval_batch_size,
            device,
        )
    elif stage == "compile":
        compile(
            model,
            model_path,
            compile_opt,
        )
    else:
        visualize(model)
