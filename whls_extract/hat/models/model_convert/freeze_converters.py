# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import List

from hat.registry import OBJECT_REGISTRY
from .converters import BaseConverter

__all__ = ["FreezePartialModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FreezePartialModule(BaseConverter):
    """Freeze part module of model.

    Args:
        modules: name of modules to freeze.
        only_batchnorm: Only freeze batchnorm, with valid gradient.
            Default is False.
    """

    def __init__(self, modules: List[str], only_batchnorm: bool = False):
        self.modules = modules
        self.only_batchnorm = only_batchnorm

    def __call__(self, model):
        for module_name in self.modules:
            m = model
            for sub_name in module_name.split("."):
                m = getattr(m, sub_name)

            # set batchnorm and dropout in eval mode
            m.eval()

            if self.only_batchnorm:
                logger.info(f"[FreezePartModule] freeze bn in {module_name}.")
            else:
                # disable grad
                logger.info(
                    f"[FreezePartModule] freeze {module_name} to disable grad."
                )
                for param in m.parameters():
                    param.requires_grad = False

        return model
