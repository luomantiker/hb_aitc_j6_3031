# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    import torchvision
except ImportError:
    torchvision = None

__all__ = ["TorchVisionAdapter"]


@OBJECT_REGISTRY.register
class TorchVisionAdapter(object):
    """Mapping interface of torchvision to HAT.

    Current adapter only supports transforms in torchvision.

    Args:
        interface: Func or classes in torchvision.
        affect_keys: affect key, str or list of str.
    """

    @require_packages("torchvision")
    def __init__(
        self, interface: Callable, affect_keys: str = "img", **kwargs
    ):
        if isinstance(interface, str) and interface in dir(
            torchvision.transforms
        ):
            interface = getattr(torchvision.transforms, interface)
        assert callable(interface)
        self.interface = interface
        if isinstance(affect_keys, (list, tuple)):
            self.affect_keys = affect_keys
        else:
            assert isinstance(affect_keys, str)
            self.affect_keys = [affect_keys]
        self.adapter = interface(**kwargs)

        if self.interface not in [
            getattr(torchvision.transforms, x)
            for x in dir(torchvision.transforms)
        ]:
            raise NotImplementedError(
                "Current adapter only support transforms in torchvision!"
            )

    def __call__(self, data):
        if self.interface in [
            getattr(torchvision.transforms, x)
            for x in dir(torchvision.transforms)
        ]:
            for key in self.affect_keys:
                if data[key] is not None:
                    data[key] = self.adapter(data[key])
        return data
