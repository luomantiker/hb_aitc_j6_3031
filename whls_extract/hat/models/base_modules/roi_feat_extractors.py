# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from horizon_plugin_pytorch.nn import MultiScaleRoIAlign as _OP
except ImportError:
    _OP = None
import horizon_plugin_pytorch.nn as hnn
from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    FixedScaleObserver,
    QuantStub,
    default_weight_8bit_fake_quant,
    get_default_qat_qconfig,
)
from torch.quantization import QConfig

from hat.core.point_geometry import coor_transformation
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    import torchvision
except ImportError:
    torchvision = None


@OBJECT_REGISTRY.register
class MultiScaleRoIAlign(nn.Module):
    @require_packages("horizon_plugin_pytorch>=0.11.0")
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aligner = _OP(*args, **kwargs)
        self.quant_roi = QuantStub(scale=0.25)

        def hook(state_dict, prefix, *args):
            for name, param in self.state_dict().items():
                key = prefix + name
                if "quant_roi" in name and key not in state_dict:
                    state_dict[key] = param

        self._register_load_state_dict_pre_hook(hook)

    def forward(
        self,
        featmaps: List[torch.Tensor],
        boxes: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
    ):
        # boxes must be convert to List[QTensor]
        if (not isinstance(boxes, QTensor)) and (
            not (isinstance(boxes, list) and isinstance(boxes[0], QTensor))
        ):
            boxes = [self.quant_roi(b) for b in boxes]
        return self._aligner(featmaps, boxes, **kwargs)

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()
        self.quant_roi.qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=FixedScaleObserver,
                quant_min=qinfo("qint16").min,
                quant_max=qinfo("qint16").max,
                saturate=False,
                scale=0.25,
                dtype="qint16",
            ),
            weight=default_weight_8bit_fake_quant,
        )


@OBJECT_REGISTRY.register
class CropperQAT(nn.Module):
    """
    Crop feature from multi-scale feature maps, implemented with \
    torchvision.ops.RoIAlign, supporting qat training.

    Args:
        size: Spacial size of cropped feature.
        strides: strides of the input multi-scale feature maps.

    Shape:
        - Input:
        * feature_maps: [N x C_i x H_i x W_i] x scale_num.
        * pixel: agent_num x 2.
        * batch_index: agent_num.
        * rois: [[agent_num_i x 4] x N] x scale_num.
        - Output: agent_num x sum(C_i) x size x size.
    """

    @require_packages("torchvision")
    def __init__(self, size: int, strides: List[int]):
        super(CropperQAT, self).__init__()

        self.size = size
        self.strides = strides
        self.roi_align = torchvision.ops.RoIAlign(
            output_size=size, spatial_scale=1, sampling_ratio=1, aligned=None
        )
        self.cat_op = nn.quantized.FloatFunctional()
        self.quant_roi = QuantStub(scale=0.25)

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        pixel: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        rois: List[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if rois is not None:
            rois = [[self.quant_roi(x) for x in tmp] for tmp in rois]
        else:
            rois = self.generate_roi(pixel, batch_index)

        agent_features = []
        for i, f in enumerate(feature_maps):
            agent_features.append(self.roi_align(f, rois[i]))
        agent_feature = self.cat_op.cat(agent_features, dim=1)
        return agent_feature

    def generate_roi(
        self, pixel: torch.Tensor, batch_index: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generate rois with specific size from pixels of agents.

        Args:
            pixel: The position of the agents on the original raster_map.
            batch_index: The index of the corresponding pixel in the batch.

        Shape:
            - Input:
            * pixel: agent_num x 2.
            * batch_index: agent_num.
            - Output: [agent_num x 5] x scale_num
        """
        rois = []
        for s in self.strides:
            tl = torch.clamp(pixel / s - self.size / 2, min=0)
            br = torch.clamp(pixel / s + self.size / 2, min=0)
            roi = torch.cat([batch_index[:, None], tl, br], dim=-1)
            rois.append(self.quant_roi(roi))
        return rois

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()
        self.quant_roi.qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=FixedScaleObserver,
                quant_min=qinfo("qint16").min,
                quant_max=qinfo("qint16").max,
                saturate=False,
                scale=0.25,
                dtype="qint16",
            ),
            weight=default_weight_8bit_fake_quant,
        )


@OBJECT_REGISTRY.register
class Cropper(nn.Module):
    """
    Crop feature from multi-scale feature maps, implemented with \
    torchvision.ops.RoIAlign. It does not support qat training, but support \
    heading norm for trajectory prediction task.

    Args:
        size: Spacial size of cropped feature.
        strides: strides of the input multi-scale feature maps.
        heading_norm: Whether to do heading norm.

    Shape:
        - Input:
        * feature_maps: [N x C_i x H_i x W_i] x scale_num.
        * pixel: agent_num x 2.
        * batch_index: agent_num.
        * angle: agent_num.
        - Output: agent_num x sum(C_i) x size x size.
    """

    def __init__(
        self, size: int, strides: List[int], heading_norm: bool = False
    ):
        super(Cropper, self).__init__()
        self.size = size
        self.strides = strides
        self.heading_norm = heading_norm

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        pixel: torch.Tensor,
        batch_index: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        crop_features = []
        for i, f in enumerate(feature_maps):
            crop_features.append(
                self.crop_single_stride(
                    f, pixel / self.strides[i], batch_index, angle
                )
            )
        crop_feature = torch.cat(crop_features, dim=1)
        return crop_feature

    def crop_single_stride(
        self,
        feature: torch.Tensor,
        pixel: torch.Tensor,
        batch_index: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Crop feature from single feature_map.

        Args:
            feature: Feature map to be cropped.
            pixel: The position of the agents on the original raster_map.
            batch_index: The index of the corresponding pixel in the batch.
            angle: The heading in img coordination of the agents.

        Shape:
            - Input:
            * feature: batch_sie x C x H x W.
            * pixel: agent_num x 2.
            * batch_index: agent_num.
            * angle: agent_num.
            - Output: agent_num x C x size x size.
        """
        S = self.size

        pixel = torch.round(pixel)
        batch_size, channel, H, W = feature.shape
        feature = torch.movedim(feature, 1, 3)
        feature = torch.reshape(feature, [batch_size * H * W, channel])

        N = pixel.shape[0]
        start = -int((S - 1) / 2)
        end = int((S + 1) / 2)
        shift = np.array(
            [[x, y] for x in range(start, end, 1) for y in range(start, end)]
        )

        if angle is not None and self.heading_norm:
            shift = coor_transformation(
                shift[None],
                angle.detach().cpu().numpy()[..., None],
            )

        shift = pixel.new_tensor(shift)
        pixel = pixel[..., None, :] + shift

        if angle is not None and self.heading_norm:
            tr = pixel.floor()
            bl = pixel.ceil()
            tr_weight = bl - pixel
            bl_weight = pixel - tr
            weight = torch.stack(
                [
                    tr_weight,
                    torch.stack(
                        [tr_weight[..., 0], bl_weight[..., 1]], axis=-1
                    ),
                    bl_weight,
                    torch.stack(
                        [bl_weight[..., 0], tr_weight[..., 1]], axis=-1
                    ),
                ],
                axis=-2,
            )
            weight = weight[..., 0] * weight[..., 1]
            weight = torch.reshape(weight, [N, S, S, 4])
            pixel = torch.stack(
                [
                    tr,
                    torch.stack([tr[..., 0], bl[..., 1]], axis=-1),
                    bl,
                    torch.stack([bl[..., 0], tr[..., 1]], axis=-1),
                ],
                axis=-2,
            )
            pixel = torch.reshape(pixel, [N, S * S * 4, 2])
        else:
            pixel = torch.reshape(pixel, [N, S * S, 2])
        pixel = torch.maximum(
            torch.minimum(pixel, pixel.new_tensor([H - 1, W - 1])),
            pixel.new_tensor(0),
        )

        batch_index = batch_index * H * W
        index = torch.reshape(
            batch_index[:, None] + (pixel[:, :, 0] * W + pixel[:, :, 1]), (-1,)
        )
        agent_feature = torch.index_select(
            feature, dim=0, index=index.to(torch.int32)
        )

        if angle is not None and self.heading_norm:
            agent_feature = agent_feature.reshape(N, S, S, 4, channel)
            agent_feature = torch.sum(
                agent_feature * weight[..., None], axis=-2
            )
        else:
            agent_feature = torch.reshape(agent_feature, [N, S, S, channel])
        agent_feature = torch.movedim(agent_feature, 3, 1)
        return agent_feature


@OBJECT_REGISTRY.register
class RoiResize(nn.Module):
    """
    Crop and Resize feature from feature_map.

    Args:
        in_strides: A list contains the strides of feature maps
        roi_resize_cfgs: A list contains the cfg dict of roi_resize.
        [
            {
                in_stride: the stride of roi_resize will use.
                output_size: the output size of roi_resize, (h,w).
                roi_box: the crop region of roi_resize, [x1,y1,x2,y2].
                roi_feat_name: the name of feature.
            },
        ]
        e.g.,
        [
            {
                "in_stride": 2,
                "output_size": (512, 512),
                "roi_box": [128,128,384,384],
                "roi_feat_name": 'roi_feat_0',
            },
        ]
        resize_mode: the interpolate method, by default "bilinear",
            support "bilinear" and "nearest".

    """

    def __init__(
        self,
        in_strides: List[int],
        roi_resize_cfgs: Union[dict, List[dict]],
        resize_mode: str = "bilinear",
    ):
        super(RoiResize, self).__init__()
        self.in_strides = in_strides
        if not isinstance(roi_resize_cfgs, list):
            roi_resize_cfgs = [roi_resize_cfgs]
        self.roi_resize_cfg = roi_resize_cfgs
        if "output_size" in roi_resize_cfgs[0]:
            self.resize_modules = nn.ModuleList()
            for roi_resize_cfg in roi_resize_cfgs:
                self.resize_modules.append(
                    hnn.Interpolate(
                        size=roi_resize_cfg["output_size"], mode=resize_mode
                    )
                )
        else:
            self.resize_modules = None

    def _crop_feature(self, feature, crop_box):
        r"""Crop feature.

        Args:
            feature: NxCxHxW
            crop_box (list): [x1,y1,x2,y2].

        """
        crop_box = [int(x) for x in crop_box]
        x1, y1, x2, y2 = crop_box
        feature = feature[:, :, y1:y2, x1:x2]
        return feature

    def forward(self, data):
        out = OrderedDict()
        for i, roi_resize_cfg in enumerate(self.roi_resize_cfg):
            cur_feat = data[self.in_strides.index(roi_resize_cfg["in_stride"])]
            roi_feat = self._crop_feature(cur_feat, roi_resize_cfg["roi_box"])
            if self.resize_modules is not None:
                roi_resize_feat = self.resize_modules[i](roi_feat)
            else:
                roi_resize_feat = roi_feat
            if "roi_feat_name" in roi_resize_cfg:
                out[roi_resize_cfg["roi_feat_name"]] = roi_resize_feat
            else:
                out[f"roi_feat{i}"] = roi_resize_feat

        return out

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class RoiCropResize(nn.Module):
    """
    Crop and Resize feature from feature_map.

    Args:
        in_strides: the strides of input feature maps
        target_stride: the target stride of roi_resize will use.
        output_size: the output size of roi_resize, (h,w).
        roi_box: the crop region of roi_resize, [x1,y1,x2,y2].
        resize_mode: the interpolate method, by default "bilinear",
            support "bilinear" and "nearest".

    """

    def __init__(
        self,
        in_strides: List[int],
        target_stride: int,
        output_size: Tuple[int, int],
        roi_box: List[int],
        resize_mode: str = "bilinear",
    ):
        super(RoiCropResize, self).__init__()
        self.in_strides = in_strides
        self.target_stride = target_stride
        self.roi_box = roi_box
        self.resize_module = hnn.Interpolate(
            size=output_size, mode=resize_mode
        )

    def _crop_feature(self, feature, crop_box):
        r"""Crop feature.

        Args:
            feature: NxCxHxW
            crop_box (list): [x1,y1,x2,y2].

        """
        crop_box = [int(x) for x in crop_box]
        x1, y1, x2, y2 = crop_box
        feature = feature[:, :, y1:y2, x1:x2]
        return feature

    def forward(self, data):
        cur_feat = data[self.in_strides.index(self.target_stride)]
        roi_feat = self._crop_feature(cur_feat, self.roi_box)
        roi_resize_feat = self.resize_module(roi_feat)

        return roi_resize_feat

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
