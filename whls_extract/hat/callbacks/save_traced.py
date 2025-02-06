# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
import pprint
from distutils.version import LooseVersion
from typing import Dict, Tuple, Union

import horizon_plugin_pytorch as horizon
import torch

try:
    from horizon_plugin_pytorch.utils.serialization import save_with_version
except ImportError:
    save_with_version = None

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import rank_zero_only
from hat.utils.hash import generate_sha256_file
from .callbacks import CallbackMixin

__all__ = ["SaveTraced"]

logger = logging.getLogger(__name__)


DEPLOY_FILE_FORMAT = "%sdeploy-checkpoint-%s.pt"


@OBJECT_REGISTRY.register
class SaveTraced(CallbackMixin):  # noqa: D205,D400
    """
    SaveTraced is used to trace a model and save it to a file.

    Args:
        save_dir: Directory to save traced model.
        trace_inputs: Example inputs for tracing.
        name_prefix: name prefix of saved model.
        allow_anno_miss: Whether to allow annotation attr missed in
            outputs of traced model.
        save_hash: Whether to save the hash value to the name of the
            pt file. Default is True.
        forward_before_trace: Whether to have a forward before trace. It can
            check model, tigger some process and so on.
        save_gpu_pt: Whether to save the pt with gpu mode. Default is False.
    """

    def __init__(
        self,
        save_dir: str,
        trace_inputs: Union[Tuple, Dict],
        name_prefix: str = "",
        allow_anno_miss: bool = True,
        save_hash: bool = True,
        forward_before_trace: bool = False,
        save_gpu_pt: bool = False,
    ):
        self.save_dir = save_dir
        self.trace_inputs = trace_inputs

        self.name_prefix = name_prefix
        self.allow_anno_miss = allow_anno_miss
        self.save_hash = save_hash
        self.forward_before_trace = forward_before_trace
        self.save_gpu_pt = save_gpu_pt

    @rank_zero_only
    def on_loop_end(self, model, **kwargs):

        os.makedirs(self.save_dir, exist_ok=True)

        if self.save_gpu_pt:
            model.cuda()
            for key in self.trace_inputs:
                for i in range(len(self.trace_inputs[key])):
                    self.trace_inputs[key][i] = self.trace_inputs[key][i].to(
                        "cuda"
                    )
        else:
            model.cpu()
        model.eval()

        if self.forward_before_trace:
            model(self.trace_inputs)
        script_module = torch.jit.trace(
            func=model,
            example_inputs=self.trace_inputs,
        )

        # TODO (shuqian,qu, ?), get_output_annotation supports more complicated
        #  output format.
        per_tensor_anno = horizon.get_output_annotation(script_module)
        logger.info(
            "annotation str of each output tensor:\n%s"
            % pprint.pformat(per_tensor_anno)
        )
        if not self.allow_anno_miss:
            for i, anno in enumerate(per_tensor_anno):
                assert anno is not None, (
                    f"annotation of the {i}th output "
                    f"tensor is None, two reasons may cause this error:"
                    f"(1) have not set annotation for this tensor. "
                    f"(2) bug of horizon.get_output_annotation(). "
                    f"you can set allow_anno_miss=True to skip this check"
                )

        pt_file = os.path.join(
            self.save_dir,
            DEPLOY_FILE_FORMAT % (self.name_prefix, "last"),
        )

        # may override
        if LooseVersion(horizon.__version__) >= LooseVersion("1.5.0"):
            # horizon.jit.save will save plugin version in pt
            horizon.jit.save(script_module, pt_file)
        elif save_with_version is not None:
            save_with_version(script_module, pt_file, _extra_files={})
        else:
            logger.warning(
                "Please update your horizon-plugin-pytorch. "
                "Plugin version should be >= 0.14.6. If not, "
                "the saved pt will not have plugin version info."
            )
            torch.jit.save(script_module, pt_file)

        if self.save_hash:
            pt_file = generate_sha256_file(pt_file, remove_old=True)
        logger.info("Save last traced deploy_model checkpoint: %s" % pt_file)
