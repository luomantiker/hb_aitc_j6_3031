# Copyright (c) Horizon Robotics. All rights reserved.

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .converters import BaseConverter

try:
    import hbdk4.compiler as hbdk4_compiler
except Exception:
    hbdk4_compiler = None


__all__ = ["LoadHbir"]


@OBJECT_REGISTRY.register
class LoadHbir(BaseConverter):
    """Load hbir module from file.

    Args:
        path: hbir model path
    """

    @require_packages("hbdk4")
    def __init__(self, path):
        self.path = path
        super(LoadHbir, self).__init__()

    def __call__(self, model):
        model = hbdk4_compiler.load(self.path)
        return model
