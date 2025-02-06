import logging

logger = logging.getLogger(__name__)

try:
    import torchvision

    ops = torchvision.ops
except (ImportError, AttributeError):
    logger.warning(
        "torchvision not found, DeformConv2d and RoIAlign are not available."
    )
