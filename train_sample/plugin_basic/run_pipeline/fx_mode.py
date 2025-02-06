import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.quantization
from common import example_input, get_args, load_pretrain, main
from hbdk4.compiler import convert
from torch import Tensor
from torch.quantization import DeQuantStub
from torchvision.models.mobilenetv2 import MobileNetV2

from horizon_plugin_pytorch.march import March, set_march
from horizon_plugin_pytorch.quantization import QuantStub
from horizon_plugin_pytorch.quantization import hbdk4 as hb4
from horizon_plugin_pytorch.quantization import prepare
from horizon_plugin_pytorch.quantization.qconfig_template import (
    default_calibration_qconfig_setter,
    default_qat_qconfig_setter,
)

##############################################################################
# Do necessary modify to the MobilenetV2 model from torchvision.
# 1. Insert QuantStub before first layer and DequantStub after last layer.
# Operation replacement and fusion will be carried out automatically (^_^).
##############################################################################


class FxQATReadyMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 0.5,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
    ):
        super().__init__(
            num_classes, width_mult, inverted_residual_setting, round_nearest
        )
        self.quant = QuantStub(scale=1 / 128)
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)

        return x


def get_model_fx(
    stage: str,
    model_path: str,
    device: torch.device,
    march=March.NASH_E,
) -> nn.Module:
    model_kwargs = dict(num_classes=10, width_mult=1.0)

    float_model = FxQATReadyMobileNetV2(**model_kwargs).to(device)

    if stage == "float":
        # We also could use the origin MobileNetV2 model for float training,
        # because modified QAT ready model can load its params seamlessly.
        float_model = MobileNetV2(**model_kwargs).to(
            device
        )  # these lines are optional

        # Load pretrained model (on ImageNet) to speed up float training.
        load_pretrain(float_model, model_path)

        return float_model

    float_ckpt_path = os.path.join(model_path, "float-checkpoint.ckpt")
    assert os.path.exists(float_ckpt_path)
    float_state_dict = torch.load(float_ckpt_path, map_location=device)

    # A global march indicating the target hardware version must be setted
    # before prepare qat.
    set_march(march)

    float_model.load_state_dict(float_state_dict)

    # We recommand to use the default_xxx_qconfig_setter, it will
    # enable high precision output if possible.
    if stage == "calib":
        qconfig_setter = default_calibration_qconfig_setter
    else:
        qconfig_setter = default_qat_qconfig_setter
    # The op fusion is included in `prepare`.
    qat_calib_model = prepare(
        # Catch computation graph of eval mode.
        float_model.eval(),
        # Must give example input to apply model tracing and do model check.
        example_input.to(device),
        qconfig_setter,
    )

    if stage == "calib":
        return qat_calib_model

    calib_ckpt_path = os.path.join(model_path, "calib-checkpoint.ckpt")
    if stage == "qat":
        if os.path.exists(calib_ckpt_path):
            calib_state_dict = torch.load(calib_ckpt_path, map_location=device)
            float_state_dict = None
        else:
            calib_state_dict = None

        if calib_state_dict is not None:
            qat_calib_model.load_state_dict(calib_state_dict)
        return qat_calib_model

    # int_infer and compile
    qat_ckpt_path = os.path.join(model_path, "qat-checkpoint.ckpt")
    if os.path.exists(qat_ckpt_path):
        qat_calib_model.load_state_dict(
            torch.load(qat_ckpt_path, map_location=device)
        )
    elif os.path.exists(calib_ckpt_path):
        qat_calib_model.load_state_dict(
            torch.load(calib_ckpt_path, map_location=device)
        )
    else:
        raise FileNotFoundError(
            "Do not find saved calib_model or qat_model ckpt "
            "to do int inference."
        )

    # If model has multi inputs, pass them as a tuple.
    hbir_model = hb4.export(qat_calib_model, example_input.to(device))
    quantized_hbir_model = convert(hbir_model, march)

    return quantized_hbir_model


if __name__ == "__main__":
    args = get_args()
    device = torch.device(
        "cuda:{}".format(args.device_id) if args.device_id >= 0 else "cpu"
    )
    model = get_model_fx(args.stage, args.model_path, device)
    main(
        model,
        args.stage,
        args.data_path,
        args.model_path,
        args.train_batch_size,
        args.eval_batch_size,
        args.epoch_num,
        args.device_id,
        compile_opt=args.opt,
    )
