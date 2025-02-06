# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from distutils.version import LooseVersion

from hat.registry import OBJECT_REGISTRY
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages

try:
    import hbdk4.compiler as hbdk4_compiler
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
    )
except Exception:
    hbdk4_compiler = None
    get_hbir_input_flattener = None

__all__ = ["HbirCompiler"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirCompiler:
    """Compile mlir.Module and estimate performance.

    Args:
        model_path: QAT hbir module file path.
        out_path: A filesystem path to save hbm. Must ends with ".hbm".
        march: BPU march, options are "bayes", "b25", "b25e", "b30", "b30g".
        opt: Optimization level. Defaults to 2.
        jobs: Number of threads launched during compiler optimization.
            Defaults to 4.
        max_time_per_fc:
            Set maximum time constraint (unit:us) for per funccall.
        debug: Set whether to contain debug info in HBM.
        preserve_quant_dequant:
            Whether preserve the quantize node on model input and
            dequantize node on model output.
        input_source:
            A structure of strings has the same structure with model input.
            Supported options are "ddr" and "pyramid".
            Defaults to None.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model_path: str,
        out_path: str,
        march: str,
        opt: int,
        jobs: int,
        max_time_per_fc: float,
        debug: bool,
        preserve_transpose: bool,
        preserve_quant_dequant: bool,
        input_source=None,
    ) -> None:
        if preserve_quant_dequant and not preserve_transpose:
            raise ValueError(
                "Cannot preserve quant dequant when dropping transpose"
            )

        self.model = hbdk4_compiler.load(model_path)
        self.out_path = out_path
        self.march = march
        self.opt = opt
        self.jobs = jobs
        self.debug = debug
        self.max_time_per_fc = max_time_per_fc

        self.preserve_transpose = preserve_transpose
        self.preserve_quant_dequant = preserve_quant_dequant
        if input_source is None:
            self.input_source = input_source
        else:
            if isinstance(input_source, str):
                input_source = input_source.split(",")
            self.input_source = get_hbir_input_flattener(self.model)(
                input_source
            )
            self.input_source = [
                source.strip() for source in self.input_source
            ]

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def set_input_source(self, input_node, source):
        if source == "ddr":
            pass
        elif source == "pyramid":
            # Convert NHWC data from pyramid to NCHW
            input_node.insert_transpose([0, 3, 1, 2])
            input_node.insert_image_preprocess(
                mode=None, divisor=1, mean=[128, 128, 128], std=[128, 128, 128]
            )
            input_node.insert_image_convert("nv12")
        else:
            msg = "Unsupported input source {}".format(source)
            logger.error(msg)
            raise ValueError(msg)

    def remove_cpu_qcast_dcast(self, args):
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

    def remove_quant_dequant(self, model):
        from hbdk4.compiler import version

        if LooseVersion(version.VERSION) > LooseVersion("4.0.13"):
            self.remove_cpu_qcast_dcast(model[0].inputs)
            self.remove_cpu_qcast_dcast(model[0].outputs)
        else:
            from hbdk4.compiler.toolbox import remove_function_io  # noqa: E402

            remove_function_io(model, "qnt.quantize")
            remove_function_io(model, "qnt.dequantize")

    def __call__(self):
        if not self.preserve_quant_dequant:
            self.remove_quant_dequant(self.model)
        if self.input_source is not None:
            func = self.model.functions[0]
            for node, source in zip(func.inputs, self.input_source):
                self.set_input_source(node, source)

        quantized_model = hbdk4_compiler.convert(self.model, self.march)

        hbdk4_compiler.compile(
            quantized_model,
            path=self.out_path,
            march=self.march,
            opt=self.opt,
            jobs=self.jobs,
            max_time_per_fc=self.max_time_per_fc,
            debug=self.debug,
        )

        logger.info(
            format_msg("Compiled model: %s" % self.out_path, MSGColor.GREEN)
        )
