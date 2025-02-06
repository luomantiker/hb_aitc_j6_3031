from hat.core.virtual_camera import (
    CylindricalCamera,
    FisheyeCamera,
    PinholeCamera,
)
from .camera import (
    DifferentiableCylindricalCamera,
    DifferentiableFisheyeCamera,
    DifferentiablePinholeCamera,
)

VIRTUAL_CAMERA_MAP = {
    FisheyeCamera: DifferentiableFisheyeCamera,
    PinholeCamera: DifferentiablePinholeCamera,
    CylindricalCamera: DifferentiableCylindricalCamera,
}
