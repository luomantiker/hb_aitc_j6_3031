# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Any, Dict, List, Tuple

from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["Detr3d", "Detr3dIrInfer"]


@OBJECT_REGISTRY.register
class Detr3d(nn.Module):
    """The basic structure of detr3d.

    Args:
        backbone: backbone module.
        neck: neck module.
        head: head module with transformer architecture.
        target: detr3d target generator.
        post_process: post process module.
        loss_cls: loss module for classification.
        loss_reg: loss module for regression.
        compile_model: Whether in compile model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        target: nn.Module = None,
        post_process: nn.Module = None,
        loss_cls: nn.Module = None,
        loss_reg: nn.Module = None,
        compile_model: bool = False,
    ):
        super(Detr3d, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.compile_model = compile_model
        self.post_process = post_process

    def extract_feat(self, img: Tensor) -> Tensor:
        """Directly extract features from the backbone + neck.

        Args:
            img: The input image to be encoded.

        Returns:
            The encoded features of the input image.
        """
        x = self.backbone(img)

        if self.neck is not None:
            x = self.neck(x)
        return x

    def _get_val(self, loss: Any) -> Tensor:
        """Retrieve the value of the loss.

        Args:
            loss: The loss value.

        Returns:
            loss: The loss value tensor.
        """
        if isinstance(loss, Dict):
            for _, v in loss.items():
                return v
        return loss

    def export_reference_points(
        self, data: Dict, feat_wh: Tuple[int, int]
    ) -> Dict:
        """Export the reference points.

        Args:
            data: The data used for exporting the reference points.
            feat_wh: The size of the feature map.
        Returns:
            The exported reference points.
        """
        return self.head.export_reference_points(data, feat_wh)

    def forward(self, data: Dict) -> Dict:
        """Perform the forward pass of the model.

        Args:
            data: A dictionary containing the input data.

        Returns:
            A dictionary containing the output of the forward pass.
        """
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        (cls_list, reg_list), ref_p = self.head(
            feats, data, self.compile_model
        )
        return self._post_process(data, cls_list, reg_list, ref_p)

    @fx_wrap()
    def _post_process(
        self,
        data: Dict,
        cls_list: List[Tensor],
        reg_list: List[Tensor],
        reference_points: Tensor,
    ) -> Dict:
        """Perform the post-processing of the model's output.

        Args:
            data: The input data.
            cls_list: A list of predicted class probabilities.
            reg_list: A list of predicted bounding boxes.
            reference_points: The reference points used for prediction.

        Returns:
            If in training mode, returns a dictionary containing
            the losses for each predicted class and regression;
            otherwise, returns the predicted class probabilities
            and bounding boxes, or the results of the post-processing
            if specified.
        """
        if self.training:
            labels = data["ego_bboxes_labels"]
            loss_dict = {}
            for i in range(len(cls_list)):
                cls_target, reg_target = self.target(
                    labels, cls_list[i], reg_list[i], reference_points
                )

                cls_loss = self._get_val(self.loss_cls(**cls_target))
                reg_loss = self._get_val(self.loss_reg(**reg_target))
                loss_dict.update({f"cls{i}": cls_loss})
                loss_dict.update({f"reg{i}": reg_loss})

            return loss_dict
        else:
            cls_pred = cls_list[-1]
            reg_pred = reg_list[-1]
            if self.post_process is None:
                return cls_pred, reg_pred
            results = self.post_process(cls_pred, reg_pred, reference_points)
            return results

    def fuse_model(self) -> None:
        """Fuse the model."""
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def set_calibration_qconfig(self):
        """Set the calibration qconfig."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()


@OBJECT_REGISTRY.register
class Detr3dIrInfer(nn.Module):
    """
    The basic structure of Detr3dIrInfer.

    Args:
        deploy_model: The deploy model to generate refpoints.
        model_convert_pipeline: Define the process of model convert.
        vt_input_hw: Feature map shape.
        ir_model: The ir model.
        post_process: Postprocess module.
    """

    def __init__(
        self,
        deploy_model: nn.Module,
        model_convert_pipeline: List[callable],
        vt_input_hw: List[int],
        ir_model: nn.Module,
        post_process: nn.Module = None,
    ):
        super(Detr3dIrInfer, self).__init__()

        self.deploy_model = model_convert_pipeline(deploy_model)
        self.deploy_model.eval()
        self.vt_input_hw = vt_input_hw
        self.post_process = post_process
        self.ir_model = ir_model

    def forward(self, data):
        meta = self.deploy_model.export_reference_points(
            data, self.vt_input_hw
        )

        ref_p = meta["reference_points"]
        meta.pop("reference_points")
        data.update(meta)
        outputs = self.ir_model(data)
        results = self.post_process(outputs[0], outputs[1], ref_p)
        return results
