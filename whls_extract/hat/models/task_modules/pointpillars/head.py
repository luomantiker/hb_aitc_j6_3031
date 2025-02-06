import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["PointPillarsHead"]


@OBJECT_REGISTRY.register
class PointPillarsHead(nn.Module):
    """Basic module of PointPillarsHead.

    Args:
        in_channels: Channel number of input feature.
        num_classes: Number of class.
        anchors_num_per_class: Anchor number for per class.
        use_direction_classifier: Whether to use direction.
        num_direction_bin: Number of direction bins.
        box_code_size: BoxCoder size.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        anchors_num_per_class: int = 2,
        use_direction_classifier: bool = True,
        num_direction_bins: int = 2,
        box_code_size: int = 7,
    ):

        super().__init__()

        self.use_direction_classifier = use_direction_classifier
        self.num_anchor_per_loc = num_classes * anchors_num_per_class  # noqa
        self.num_classes = num_classes

        num_cls = self.num_anchor_per_loc * num_classes
        num_box = self.num_anchor_per_loc * box_code_size

        self.conv_box = nn.Conv2d(in_channels, num_box, 1)
        self.conv_cls = nn.Conv2d(in_channels, num_cls, 1)

        if self.use_direction_classifier:
            num_dir = self.num_anchor_per_loc * num_direction_bins
            self.conv_dir = nn.Conv2d(in_channels, num_dir, 1)

        self.dequant = DeQuantStub()

    def forward(self, x):
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        if self.use_direction_classifier:
            dir_preds = self.conv_dir(x)
            dir_preds = self.dequant(dir_preds)
            # dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_preds = None

        box_preds = self.dequant(box_preds)
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = self.dequant(cls_preds)
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        return box_preds, cls_preds, dir_preds

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()
