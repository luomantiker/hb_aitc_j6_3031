from typing import Tuple

import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["FCOS3D", "FCOS3DIrInfer"]


@OBJECT_REGISTRY.register
class FCOS3D(nn.Module):
    """The basic structure of fcos3d.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        post_process: post_process module.
        loss: loss module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        post_process: nn.Module = None,
        loss: nn.Module = None,
    ):
        super(FCOS3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.targets = targets
        self.post_process = post_process
        self.loss = loss

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_pred,
        attr_pred,
        centerness,
        data,
    ):
        if self.training and self.targets is not None:
            (
                labels_3d,
                bbox_targets_3d,
                centerness_targets,
                attr_targets,
            ) = self.targets(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                gt_bboxes_list=data["gt_bboxes"],
                gt_labels_list=data["gt_labels"],
                gt_bboxes_3d_list=data["gt_bboxes_3d"],
                gt_labels_3d_list=data["gt_labels_3d"],
                centers2d_list=data["centers2d"],
                depths_list=data["depths"],
                attr_labels_list=data["attr_labels"],
            )
            loss = self.loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                dir_cls_preds=dir_cls_pred,
                attr_preds=attr_pred,
                centernesses=centerness,
                labels_3d=labels_3d,
                bbox_targets_3d=bbox_targets_3d,
                centerness_targets=centerness_targets,
                attr_targets=attr_targets,
            )
            return loss
        else:
            bbox_outputs = self.post_process(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                dir_cls_preds=dir_cls_pred,
                attr_preds=attr_pred,
                centernesses=centerness,
                img_metas=data,
                cfg=None,
                rescale=True,
            )
            return bbox_outputs

    def forward(self, data):
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        if (self.training and self.targets is not None) or (
            self.post_process is not None
        ):
            (
                cls_scores,
                bbox_preds,
                dir_cls_pred,
                attr_pred,
                centerness,
            ) = self.head(feats)
            return self._post_process(
                cls_scores,
                bbox_preds,
                dir_cls_pred,
                attr_pred,
                centerness,
                data,
            )
        else:
            # cls_scores, bbox_preds1, bbox_preds2, bbox_preds3, bbox_preds4,
            # bbox_preds5, dir_cls_pred, attr_pred, centerness
            return self.head(feats)

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class FCOS3DIrInfer(nn.Module):
    """
    The basic structure of FCOS3DIrInfer.

    Args:
        ir_model: The ir model.
        post_process: Postprocess module.
        strides: A list of strides.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        post_process: nn.Module,
        strides: Tuple[int],
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        self.strides = strides

    def _process_outputs(self, outputs):
        cls_scores = outputs[0]
        bbox_preds = []
        for i in range(len(outputs[0])):
            bbox_pred = []
            for j in list(range(1, 6)):
                bbox_pred.append(outputs[j][i])

            bbox_pred = self._decode(bbox_pred, self.strides[i])
            bbox_preds.append(bbox_pred)

        dir_cls_pred = outputs[6]
        attr_pred = outputs[7]
        centerness = outputs[8]
        return cls_scores, bbox_preds, dir_cls_pred, attr_pred, centerness

    def forward(self, data):

        outputs = self.ir_model(data)

        (
            cls_scores,
            bbox_preds,
            dir_cls_pred,
            attr_pred,
            centerness,
        ) = self._process_outputs(outputs)

        bbox_outputs = self.post_process(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            dir_cls_preds=dir_cls_pred,
            attr_preds=attr_pred,
            centernesses=centerness,
            img_metas=data,
            cfg=None,
            rescale=True,
        )
        return bbox_outputs

    def _decode(self, bbox_pred, stride):
        bbox_pred = torch.cat(bbox_pred, dim=1)
        clone_bbox = bbox_pred.clone()
        bbox_pred[:, :2] = clone_bbox[:, :2].float()
        bbox_pred[:, 2] = clone_bbox[:, 2].float()
        bbox_pred[:, 3:6] = clone_bbox[:, 3:6].float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp()
        bbox_pred[:, :2] *= stride
        return bbox_pred
