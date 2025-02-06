import logging
import os
import pprint
import sys

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from hbdk4.compiler import Module, compile, convert, hbm_perf, load, save
from horizon_plugin_pytorch.quantization.hbdk4 import export

from hat.engine.predictor import Predictor
from hat.utils.apply_func import _as_list
from hat.utils.checkpoint import load_state_dict
from hat.utils.logger import init_logger
from hat.utils.statistics import cal_ops
from hat.visualize.lidar_det import lidar_det_visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_checker(deploy_model, deploy_inputs, march):
    try:
        deploy_model.eval()
        qat_hbir = export(deploy_model, deploy_inputs)
        convert(qat_hbir, march)
        logger.info("This model is supported!")
    except Exception:
        logger.error("Failed to pass hbdk checker")
        sys.exit(1)


def calops(deploy_model, deploy_inputs):
    deploy_model.eval()
    total_ops, total_params = cal_ops(
        model=deploy_model,
        inputs=deploy_inputs,
    )
    print("Params: %.6f M" % (total_params / (1000 ** 2)))
    print("FLOPs: %.6f G" % (total_ops / (1000 ** 3)))


def align_bpu_validation(
    int_model,
    data_loader,
    val_batch_processor,
    val_metrics,
    val_callbacks,
    device,
    ckpt_dir,
):
    init_logger(f"{ckpt_dir}/align_bpu_validation")
    logger.info("=" * 50 + "BEGIN ALIGN BPU VALIDATION" + "=" * 50)

    device_id = list(map(int, device.split(",")))[0]
    torch.cuda.set_device(device_id)

    int_model.eval()

    predictor = Predictor(
        model=int_model,
        data_loader=data_loader,
        batch_processor=val_batch_processor,
        device=device_id,
        callbacks=_as_list(val_callbacks),
        num_epochs=0,
        log_interval=1,
    )
    predictor.val_metrics = _as_list(val_metrics)
    predictor.fit()
    logger.info("=" * 50 + "END ALIGN BPU VALIDATION" + "=" * 50)


def int_infer_viz_lidar(
    model,
    input_points,
    device,
    is_plot=True,
):
    logger.info("=" * 50 + "BEGIN LIDAR INFER" + "=" * 50)

    device_id = list(map(int, device.split(",")))[0]
    torch.cuda.set_device(device_id)
    model.cuda()

    points = np.fromfile(input_points, dtype=np.float32).reshape((-1, 4))
    points = torch.from_numpy(points).cuda()
    data = {
        "points": [points],
    }

    model.eval()
    model_out = model(data)

    lidar_det_visualize(
        points=points,
        predictions=model_out[0],
        score_thresh=0.4,
        is_plot=is_plot,
    )
    logger.info("=" * 50 + "END LIDAR INFER" + "=" * 50)


def export_hbir(
    deploy_model,
    deploy_inputs,
    march,
    task_name="model",
    output_dir="./tmp_models",
    debug=False,
):
    deploy_inputs = _as_list(deploy_inputs)

    deploy_model.eval()

    logger.info("Export qat model to qat hbir.")
    qat_hbir = export(
        deploy_model,
        deploy_inputs,
        name=task_name,
    )

    logger.info(f"Saving qat hbir to {output_dir}")
    save_hbir(qat_hbir, os.path.join(output_dir, "qat"), debug)

    logger.info("Converting hbir to quantized.")
    quantized_hbir = convert(qat_hbir, march)
    logger.info(f"Saving quantized hbir to {output_dir}")
    save_hbir(
        quantized_hbir,
        os.path.join(output_dir, "quantized"),
        debug,
    )


def compile_perf(
    model_path,
    march,
    opt="O2",
    jobs=4,
    out_path="./tmp_models/model.hbm",
):
    model = load(model_path)

    dst_dir = os.path.dirname(out_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    model = convert(model, march)
    remove_quant_dequant(model)
    opt = int(opt[-1])
    compile(
        model,
        path=out_path,
        march=march,
        opt=opt,
        jobs=jobs,
    )


def save_hbir(model, path: str, debug: bool):
    dir_name = os.path.dirname(path)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)
    if debug:
        if isinstance(model, Module):
            model = model.module

        if not path.endswith(".mlir"):
            path += ".mlir"
        with open(path, "w") as f:
            f.writelines(
                str(
                    model.operation.get_asm(
                        enable_debug_info=True, pretty_debug_info=False
                    )
                )
            )
    else:
        if not path.endswith(".bc"):
            path += ".bc"
        save(model, path)


def remove_quant_dequant(model):
    def _remove_cpu_qcast_dcast(args):
        for arg in args:
            removable, diagnostic = arg.is_removable
            if removable:
                attached_op = arg.get_attached_op[0]
                if attached_op.type == "hbtl.call":
                    schema = attached_op.schema
                    if schema.namespace == "quant" and schema.signature in [
                        "qcast",
                        "dcast",
                    ]:
                        removed, diagnostic = arg.remove_attached_op()
                        if removed is False:
                            raise RuntimeError(
                                "Remove quant/dequant failed, reason is:"
                                "\n{}".format(diagnostic)
                            )

    _remove_cpu_qcast_dcast(model[0].inputs)
    _remove_cpu_qcast_dcast(model[0].outputs)
