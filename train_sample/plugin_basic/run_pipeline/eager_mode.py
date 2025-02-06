import os
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.quantization
from common import example_input, get_args, load_pretrain, main
from hbdk4.compiler import convert
from torch import Tensor
from torch.quantization import DeQuantStub
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2
from torchvision.ops.misc import ConvNormActivation

from horizon_plugin_pytorch.march import March, set_march
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import (
    PrepareMethod,
    QuantStub,
    fuse_known_modules,
)
from horizon_plugin_pytorch.quantization import hbdk4 as hb4
from horizon_plugin_pytorch.quantization import prepare
from horizon_plugin_pytorch.quantization.qconfig_template import (
    default_calibration_qconfig_setter,
    default_qat_qconfig_setter,
)

##############################################################################
# Do necessary modify to the MobilenetV2 model from torchvision.
# 1. Insert QuantStub before first layer and DequantStub after last layer.
# 2. Replace unsupported torch func ops with their nn.Module counterpart.
#    In this case, we
#    a. replace the plus sign with `FloatFunctional().add`.
#    b. replace `nn.functional.adaptive_avg_pool2d`
#       with `nn.AdaptiveAvgPool2d`.
# 3. Manually define the ops to be fused (by Module's name).
#    All availiable fuse patterns can be accessed by
#    `horizon_plugin_pytorch.quantization.fuse_modules.get_op_list_to_fuser_mapping()`.
#    Note: User should fuse as many ops as possible, or model accuracy and
#          execution speed will be effected.
##############################################################################


class EagerQATReadyInvertedResidual(InvertedResidual):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inp, oup, stride, expand_ratio, norm_layer)

        if self.use_res_connect:
            # Must register the FloatFunctional as submodule,
            # or the quantization state will not be handled correctly.
            self.skip_add = FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(self.conv(x), x)
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                if not self.use_res_connect:
                    # Fuse conv+bn
                    torch.quantization.fuse_modules(
                        self.conv,
                        [str(idx), str(idx + 1)],
                        inplace=True,
                        fuser_func=fuse_known_modules,
                    )
                else:
                    # Fuse conv+bn+add
                    torch.quantization.fuse_modules(
                        self,
                        [
                            "conv." + str(idx),
                            "conv." + str(idx + 1),
                            "skip_add",
                        ],
                        inplace=True,
                        fuser_func=fuse_known_modules,
                    )


class EagerQATReadyMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 0.5,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
    ):
        super().__init__(
            num_classes,
            width_mult,
            inverted_residual_setting,
            round_nearest,
            block=EagerQATReadyInvertedResidual,
        )
        # Horizon QuantStub support user-specified scale.
        # The `input_source` param of `compile_model` can be set to "pyramid"
        # only if input scale is equal to 1/128.
        self.quant = QuantStub(scale=1 / 128)
        self.dequant = DeQuantStub()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # torch.flatten is supported
        x = self.classifier(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvNormActivation):
                # Fuse conv+bn+relu
                torch.quantization.fuse_modules(
                    m,
                    ["0", "1", "2"],
                    inplace=True,
                    fuser_func=fuse_known_modules,
                )
            if type(m) == EagerQATReadyInvertedResidual:
                m.fuse_model()


def get_model_eager(
    stage: str,
    model_path: str,
    device: torch.device,
    march=March.NASH_E,
) -> nn.Module:
    model_kwargs = dict(num_classes=10, width_mult=1.0)

    float_model = EagerQATReadyMobileNetV2(**model_kwargs).to(device)

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
    # Manually do op fusion.
    float_model.fuse_model()

    # We recommand to use the default_xxx_qconfig_setter, it will
    # enable high precision output if possible.
    if stage == "calib":
        qconfig_setter = default_calibration_qconfig_setter
    else:
        qconfig_setter = default_qat_qconfig_setter
    qat_calib_model = prepare(
        float_model,
        # Must give example input to apply qconfig_setter and do model check.
        example_input.to(device),
        qconfig_setter,
        PrepareMethod.EAGER,
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
    model = get_model_eager(args.stage, args.model_path, device)
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
