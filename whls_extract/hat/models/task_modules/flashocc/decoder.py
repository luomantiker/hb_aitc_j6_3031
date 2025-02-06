from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.quantization import DeQuantStub

from hat.models.task_modules.view_fusion.decoder import BevDecoderInfer
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = [
    "FlashOccDetDecoder",
    "FlashOccDecoderInfer",
    "BevformerOccDetDecoder",
]


@OBJECT_REGISTRY.register
class FlashOccDetDecoder(nn.Module):
    """The basic decoder structure of flashocc.

    Args:
        occ_head: Head module.
        loss_occ: Classify loss module.
        roi_resizer: RoI resizer module.
        use_mask: Whether to use masking in the decoder.
        lidar_input: Whether to use LiDAR input.
        camera_input: Whether to use camera input.
        num_classes: Number of classes for prediction.
        is_compile: Whether for compile.
    """

    def __init__(
        self,
        occ_head: Optional[nn.Module] = None,
        loss_occ: nn.Module = None,
        roi_resizer: nn.Module = None,
        use_mask: bool = True,
        lidar_input: bool = False,
        camera_input: bool = True,
        num_classes: bool = 18,
        is_compile: bool = False,
        **kwargs
    ):
        super(FlashOccDetDecoder, self).__init__(**kwargs)
        self.occ_head = occ_head
        self.roi_resizer = roi_resizer
        self.is_compile = is_compile
        self.loss_occ = loss_occ
        self.use_mask = use_mask
        self.lidar_input = lidar_input
        self.camera_input = camera_input
        self.num_classes = num_classes
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _post_process(self, occ_preds, data):
        """Post process."""
        if self.training:
            voxel_semantics = data["voxel_semantics"]  # (B, Dx, Dy, Dz)
            if self.lidar_input and self.camera_input:
                mask = data["mask_lidar"] | data["mask_camera"]
            if self.lidar_input:
                mask = data["mask_lidar"]  # (B, Dx, Dy, Dz)
            elif self.camera_input:
                mask = data["mask_camera"]
            loss_occ = self._loss(
                occ_preds,  # (B, Dx, Dy, Dz, n_cls)
                voxel_semantics,
                mask,
            )
            return loss_occ
        else:
            occ_pred = self._decode(occ_preds)
            return {"occ_pre": occ_pred}

    def _loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Compute losses for occ task.

        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)

        Returns:
            loss: dict of loss.
        """
        loss = {}
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)  # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,  # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,  # (B*Dx*Dy*Dz, )
                mask_camera,  # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples,
            )
            loss["loss_occ"] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds, voxel_semantics, avg_factor=num_total_samples
            )

            loss["loss_occ"] = loss_occ
        return loss

    def _decode(self, pred: Tensor):
        """
        Decode the predicted values using the provided meta information.

        Args:
            pred: The predicted values. Shap: (B, Dx, Dy, Dz, C).

        Returns:
            The predicted values.
        """

        occ_score = pred.softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        return occ_res

    def forward(self, bev_feat: List[Tensor], data: Dict):
        """Perform the forward pass of the model.

        Args:
            bev_feat: The input features.
            data: The meta information.

        Returns:
            pred: The predictions of the model,including
                  occ_preds and loss dict.
        """
        if isinstance(bev_feat, Sequence):
            bev_feat = bev_feat[0]
        if self.roi_resizer is not None:
            bev_feat = self.roi_resizer([bev_feat])["roi_feat0"]

        occ_preds = self.occ_head(bev_feat)
        occ_preds = self.dequant(occ_preds)
        if self.is_compile:
            return [occ_preds], {}
        return [occ_preds], self._post_process(occ_preds, data)


@OBJECT_REGISTRY.register
class BevformerOccDetDecoder(FlashOccDetDecoder):
    def forward(self, bev_feat: List[Tensor], data: Dict):
        if not self.is_compile:
            gt_occ_info = data["seq_meta"][0]["gt_occ_info"]
        else:
            gt_occ_info = {}
        pred, result = super(BevformerOccDetDecoder, self).forward(
            bev_feat, gt_occ_info
        )
        if self.is_compile:
            return pred
        if self.training:
            return result
        else:
            return [result["occ_pre"]]


@OBJECT_REGISTRY.register
class FlashOccDecoderInfer(BevDecoderInfer):
    def forward(self, occ_pred, data):
        occ_score = occ_pred[0].softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        return None, {"occ_pre": occ_res}
