import logging

from torch import nn

__all__ = ["DeformConv2d", "RoIAlign"]

logger = logging.getLogger(__name__)

try:
    from torchvision.ops import DeformConv2d, RoIAlign
except ImportError:
    logger.warning(
        "torchvision not found, DeformConv2d and RoIAlign are not available."
    )

    class DeformConv2d(nn.Module):
        def __init__(self, *args, **kwargs):
            msg = "DeformConv2d is not available because `torchvision` is not found"  # noqa E501
            logger.fatal(msg)
            raise RuntimeError(msg)

    class RoIAlign(nn.Module):
        def __init__(self, *args, **kwargs):
            msg = (
                "RoIAlign is not available because `torchvision` is not found"
            )
            logger.fatal(msg)
            raise RuntimeError(msg)
