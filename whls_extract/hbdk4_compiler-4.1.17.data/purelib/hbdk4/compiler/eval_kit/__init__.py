import os
import torch

from hbdk4.compiler import save, March, Module
from hbdk4.compiler._mlir_libs._hbdk import (
    _simplify,
    _compile_after_legalize,
    _compile_after_hbir_opt,
)

from hbdk4.compiler.torch import export

from hbdk4.compiler.eval_kit.model_gallery.utils import trace

from hbdk4.compiler.eval_kit.model_gallery.effdet import retrieve_efficientdet
from hbdk4.compiler.eval_kit.model_gallery.timm import (
    retrieve_timm_backbone,
    retrieve_timm_imagenet,
)
from hbdk4.compiler.eval_kit.model_gallery.detr import retrieve_detr
from hbdk4.compiler.eval_kit.model_gallery.yolo import retrieve_yolo
from hbdk4.compiler.eval_kit.model_gallery.bottleneck_lstm import (
    retrieve_bottleneck_lstm,
)
from hbdk4.compiler.eval_kit.model_gallery.gpt2 import retrieve_gpt2


# simplify hbir: canonicalize, fusion, layout optimize
def simplify(m: Module, march: March = March.nash_e):
    if not isinstance(march, March):
        march = March.get(march)
    args = {"march": march}
    _simplify(m.module.operation, m.module.context, args)


def compile(m: Module, march: March = March.nash_e, opt=2, log_path=""):
    module = m.module
    if len(log_path) != 0:
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    if not isinstance(march, March):
        march = March.get(march)

    stage_mlir = log_path + "/original.bc"
    try:
        save(m, stage_mlir)

        args = {"march": march, "log_path": log_path, "opt": str(opt)}
        _simplify(module.operation, module.context, args)

        stage_mlir_file = "after_hbir_legalize.bc"
        save(m, os.path.join(log_path, stage_mlir_file))

        _compile_after_legalize(module.operation, module.context, args)

        stage_mlir_file = "after_hbir_opt.bc"
        save(m, os.path.join(log_path, stage_mlir_file))

        _compile_after_hbir_opt(module.operation, module.context, args)

        stage_mlir_file = "after_%s_opt_O%d.bc" % (march, opt)
        save(m, os.path.join(log_path, stage_mlir_file))

    except:
        print(
            "\033[91m",
            "mlir program compilation failed when running pipeline",
            ", check",
            stage_mlir_file,
            "\033[0m",
        )
        exit(1)


def genTimmImagenet(backbone, batch, pretrained=False):
    traced_imagenet, ei = retrieve_timm_imagenet(backbone, batch, pretrained)
    print("run imagenet", backbone, "example input", ei.shape)
    return export(traced_imagenet, ei, name=backbone)


def genTimmBackbone(backbone, *sizes, pretrained=False):
    ei = torch.rand(*sizes)
    traced_backbone = retrieve_timm_backbone(backbone, ei, pretrained)
    print("run backbone", backbone, "example input", ei.shape)
    return export(traced_backbone, ei, name=backbone)


def genEfficientDet(backbone, pretrained=False, post_process=False):
    traced_effdet, ei = retrieve_efficientdet(
        backbone, pretrained, post_process=post_process
    )
    print("run efficidentdet", backbone, "example input", ei.shape)
    return export(traced_effdet, ei, name=backbone)


def genYolo(backbone, sizes=None, pretrained=False, post_process=False):
    traced_effdet, ei = retrieve_yolo(
        backbone, sizes, pretrained, post_process=post_process
    )
    print("run darknet", backbone, "example input", ei.shape)
    return export(traced_effdet, ei, name=backbone)


def genDetr(backbone, sizes, pretrained=False, post_process=False):
    traced_effdet, ei = retrieve_detr(backbone, sizes, pretrained, post_process)
    print("run detr", backbone, "example input", ei.shape)
    return export(traced_effdet, ei, name=backbone)


def genBottleneckLstm(batch):
    traced_lstm, ei = retrieve_bottleneck_lstm(batch)
    print("run bottleneck-lstm")
    return export(traced_lstm, ei, name="bottleneck-lstm")


def genGpt2(sizes, num_layers, dimension_model, num_heads):
    traced_gpt2, ei = retrieve_gpt2(sizes, num_layers, dimension_model, num_heads)
    print("run gpt2", "example input", ei.shape)
    return export(traced_gpt2, ei, name="gpt2")


def genCaseName(num, model, batch=None, shape=None):
    name = "%d_%s" % (num, model)
    assert batch is None or shape is None, "cannot specify both batch and shape"
    if batch:
        assert isinstance(batch, int), "invalid batch %s" % batch
        name += "_batch%d" % batch
    if shape:
        assert isinstance(shape, (list, tuple)), "invalid shape %s" % shape
        assert all(isinstance(x, int) for x in shape), "invalid shape %s" % shape
        name += "_" + "x".join([str(s) for s in shape])
    return name


def parseArgs():
    import argparse

    parser = argparse.ArgumentParser(description="Configure run")
    parser.add_argument(
        "--opt", type=int, required=False, default=1, help="Optimization level"
    )
    parser.add_argument(
        "--full", type=bool, required=False, default=False, help="Run test completely?"
    )
    parser.add_argument(
        "--batch", type=int, required=False, default=1, help="Specify batch of runs"
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        required=False,
        default=False,
        help="Use real weight or not",
    )
    parser.add_argument(
        "--post_process",
        type=bool,
        required=False,
        default=False,
        help="Detection with post process or not",
    )
    return parser.parse_args()
