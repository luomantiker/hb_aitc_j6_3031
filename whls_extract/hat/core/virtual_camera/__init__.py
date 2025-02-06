from enum import Enum

from .camera_base import CameraBase
from .cameras import (
    CylindricalCamera,
    FisheyeCamera,
    IPMCamera,
    PinholeCamera,
    SphericalCamera,
)

__all__ = [
    "CameraBase",
    "CylindricalCamera",
    "FisheyeCamera",
    "PinholeCamera",
    "SphericalCamera",
    "IPMCamera",
]


class CameraModelType(Enum):
    """Camera Model Type."""

    pinhole = PinholeCamera
    fisheye = FisheyeCamera
    spherical = SphericalCamera
    cylindrical = CylindricalCamera
    Pinhole = PinholeCamera
    Fisheye = FisheyeCamera
    Spherical = SphericalCamera
    Cylindrical = CylindricalCamera
    CylindricalCamera = CylindricalCamera
    FisheyeCamera = FisheyeCamera
    PinholeCamera = PinholeCamera
    SphericalCamera = SphericalCamera
    IPMCamera = IPMCamera
