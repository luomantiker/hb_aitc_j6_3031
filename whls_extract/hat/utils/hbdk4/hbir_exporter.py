# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import List

import horizon_plugin_pytorch as horizon
import torch
from torch.utils._pytree import tree_flatten

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages

try:
    import hbdk4.compiler as hbdk4_compiler
    from horizon_plugin_pytorch.quantization.hbdk4 import export
except Exception:
    hbdk4_compiler = None
    export = None

__all__ = ["HbirExporter"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirExporter:
    """Export hbir from qat nn.Module and convert hbir to quantized.

    Args:
        model: Input model.
        model_convert_pipeline:
            Model convert pipeline that converts model to qat mode.
        example_inputs: Model input for exporting.
            **NOTE**: Hbir model do not support dynamic shape currently.
        march: BPU march.
        save_path: Where to save exported hbir files.
        with_check: Whether check the mlir model forward with example input.
        model_name: Name of compiled model.
        kwargs: other params of export.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model: torch.nn.Module,
        model_convert_pipeline: List[callable],
        example_inputs: dict,
        march: str,
        save_path: str,
        with_check: bool,
        model_name: str = "model",
        **kwargs
    ) -> None:
        horizon.march.set_march(march)

        self.model = model_convert_pipeline(model)
        self.example_inputs = _as_list(example_inputs)
        self.march = march
        self.save_path = save_path
        self.with_check = with_check
        self.model_name = model_name
        self.kwargs = kwargs

    def get_hbir_input(self, example_inputs):
        flat_inputs, _ = tree_flatten(example_inputs)
        flat_inputs = horizon.utils.misc.pytree_convert(
            flat_inputs, torch.Tensor, lambda x: x.cpu().numpy()
        )

        return flat_inputs

    def save_hbir(self, model, path: str, debug: bool):
        dir_name = os.path.dirname(path)
        if len(dir_name) > 0:
            os.makedirs(dir_name, exist_ok=True)

        if debug:
            if isinstance(model, hbdk4_compiler.Module):
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
            hbdk4_compiler.save(model, path)

    def __call__(self, debug: bool) -> None:
        logger.info(
            "Exporting hbir with input {}".format(
                horizon.utils.misc.tensor_struct_repr(self.example_inputs)
            )
        )

        self.model.eval()
        qat_hbir = export(
            self.model,
            self.example_inputs,
            name=self.model_name,
            native_pytree=False,
            **self.kwargs
        )

        logger.info(
            format_msg(
                "Saving qat hbir to {}".format(self.save_path), MSGColor.GREEN
            )
        )
        self.save_hbir(qat_hbir, os.path.join(self.save_path, "qat"), debug)

        if self.with_check:
            logger.info("Checking qat hbir with example_inputs.")
            qat_hbir.functions[0](*self.get_hbir_input(self.example_inputs))
            logger.info(format_msg("Qat hbir check passed.", MSGColor.GREEN))

        logger.info("Converting hbir to quantized.")
        quantized_hbir = hbdk4_compiler.convert(qat_hbir, self.march)

        logger.info(
            format_msg(
                "Saving quantized hbir to {}".format(self.save_path),
                MSGColor.GREEN,
            )
        )
        self.save_hbir(
            quantized_hbir,
            os.path.join(self.save_path, "quantized"),
            debug,
        )

        if self.with_check:
            logger.info("Checking quantized hbir with example_inputs.")
            quantized_hbir.functions[0](
                *self.get_hbir_input(self.example_inputs)
            )
            logger.info(
                format_msg("Quantized hbir check passed.", MSGColor.GREEN)
            )
