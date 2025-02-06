from typing import List, Optional

import torch
import torch.nn as nn

from hat.models.task_modules.pointpillars.preprocess import (
    PointPillarsPreProcess,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

torch.fx.wrap("len")

__all__ = ["PointPillarsDetector", "PointPillarsDetectorIrInfer"]


@OBJECT_REGISTRY.register
class PointPillarsDetector(nn.Module):
    """
    The basic structure of PointPillars.

    Args:
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        out_size_factor: Downsample factor.
        reader: Reader module.
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        anchor_generator: Anchor generator module.
        targets: Target generator module.
        loss: Loss module.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        feature_map_shape: List[int],
        pre_process: Optional[nn.Module] = None,
        reader: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        anchor_generator: Optional[nn.Module] = None,
        targets: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        quant_begin_neck: bool = False,
        is_deploy: bool = False,
    ):
        super(PointPillarsDetector, self).__init__()

        self.pre_process = pre_process
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.targets = targets
        self.anchor_generator = anchor_generator
        self.loss = loss
        self.postprocess = postprocess
        self.quant_begin_neck = quant_begin_neck

        self.feature_map_shape = feature_map_shape
        self.is_deploy = is_deploy

        if self.pre_process and isinstance(
            self.pre_process, PointPillarsPreProcess
        ):  # noqa E501
            self.use_horizon_preprocess = True
        else:
            self.use_horizon_preprocess = False

    def forward(self, example):

        # train: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> Neck && Head -> output -> Loss
        # eval: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> Neck && Head -> output -> post_process
        # deploy: -------------------------------------------------------
        #    [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #       -> voxel-feature -> Neck && Head -> output

        if self.is_deploy and self.quant_begin_neck:
            data = dict(  # noqa C408
                features=example["feature"],
            )
            x = data["features"]
        else:
            if self.pre_process:
                features, coords, num_points_in_voxel = self.pre_process(
                    example["points"], self.is_deploy
                )
                data = dict(  # noqa C408
                    features=features,
                    coors=coords,
                    num_points_in_voxel=num_points_in_voxel,
                    batch_size=len(example["points"]),
                    input_shape=self.feature_map_shape,
                )
            else:
                data = dict(  # noqa C408
                    features=example["voxels"],
                    num_points_in_voxel=example["num_points"],
                    coors=example["coordinates"],
                    batch_size=example["num_voxels"].shape[0],
                    input_shape=self.feature_map_shape,
                )

            input_features = self.reader(
                features=data["features"],
                num_voxels=data["num_points_in_voxel"],
                coors=data["coors"],
                horizon_preprocess=self.use_horizon_preprocess,
            )

            x = self.backbone(
                input_features,
                data["coors"],
                data["batch_size"],
                torch.tensor(self.feature_map_shape),
            )
        x = self.neck(x, quant=self.quant_begin_neck)
        box_preds, cls_preds, dir_preds = self.head(x)
        if self.is_deploy:
            return box_preds, cls_preds, dir_preds

        # (B, C, H, W) -> (B, H, W, C)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()

        return self._post_process(example, (box_preds, cls_preds, dir_preds))

    @fx_wrap()
    def _post_process(self, data, preds):
        """
        Perform post-processing of model outputs.

        Args:
            data: A dictionary containing the input data.
            preds: Model outputs to be processed.

        Returns:
            results: Processed results, which can be losses during training or
                     bounding box results during evaluation.
        """
        # (1, H, W)
        box_preds, cls_preds, dir_preds = preds
        feature_map_size = [1, box_preds.shape[1], box_preds.shape[2]]

        device = box_preds.device

        anchor_dict = self.anchor_generator(feature_map_size, device=device)
        anchors = torch.cat(anchor_dict["anchors"], dim=-2)

        if self.training and self.targets is not None:
            bbox_targets, cls_labels, reg_weights = self.targets(
                anchor_dict["anchors"],
                anchor_dict["matched_thresholds"],
                anchor_dict["unmatched_thresholds"],
                data["annotations"],
                device=device,
            )

            return self.loss(
                anchors,
                cls_labels,
                bbox_targets,
                box_preds,
                cls_preds,
                dir_preds,
            )

        else:
            return self.postprocess(
                box_preds,
                cls_preds,
                dir_preds,
                anchors,
            )

    def fuse_model(self):
        """Fuse quantizable modules in the model, used in `eager` mode.

        This function fuses quantizable modules within the model to prepare it
        for quantization.
        """
        if not self.quant_begin_neck:
            # P2
            if self.reader:
                self.reader.fuse_model()

        if self.neck:
            self.neck.fuse_model()

    def set_qconfig(self):
        """Set quantization configurations for the model.

        This function sets quantization configurations for the model and its
        submodules. It configures quantization settings for different parts of
        the model based on the `quant_begin_neck` attribute.
        """
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.reader, self.backbone]:
            if module is not None:
                if self.quant_begin_neck:
                    module.qconfig = None
                else:
                    if hasattr(module, "set_qconfig"):
                        module.set_qconfig()

        for module in [self.neck, self.head]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()

    def set_calibration_qconfig(self):
        """Set calibration quantization configurations for the model.

        This function is deprecated by calibration_v2.
        """
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()

        for module in [self.reader, self.backbone]:
            if module is not None:
                if self.quant_begin_neck:
                    module.qconfig = None
                else:
                    if hasattr(module, "set_calibration_qconfig"):
                        module.set_calibration_qconfig()

        for module in [self.neck, self.head]:
            if module is not None and hasattr(
                module, "set_calibration_qconfig"
            ):
                module.set_calibration_qconfig()


@OBJECT_REGISTRY.register
class PointPillarsDetectorIrInfer(nn.Module):
    """
    The basic structure of PointPillarsDetectorIrInfer.

    Args:
        ir_model: The ir model.
        postprocess: Postprocess module.
        anchor_generator: The anchor generator module.
        max_points: The max of points.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        postprocess: nn.Module,
        anchor_generator: nn.Module,
        max_points: int = 150000,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.postprocess = postprocess
        self.max_points = max_points
        self.anchor_generator = anchor_generator

    def forward(self, example):

        padding_shape = self.max_points - example["points"][0].shape[0]
        inputs = torch.nn.functional.pad(
            example["points"][0],
            [0, 0, 0, padding_shape],
            mode="constant",
            value=-100,
        )

        hbir_outputs = self.ir_model({"points": inputs})

        box_preds, cls_preds, dir_preds = hbir_outputs
        # (B, C, H, W) -> (B, H, W, C)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()

        return self._post_process(example, (box_preds, cls_preds, dir_preds))

    def _post_process(self, data, preds):
        # (1, H, W)
        box_preds, cls_preds, dir_preds = preds
        feature_map_size = [1, box_preds.shape[1], box_preds.shape[2]]

        device = box_preds.device

        anchor_dict = self.anchor_generator(feature_map_size, device=device)
        anchors = torch.cat(anchor_dict["anchors"], dim=-2)

        return self.postprocess(
            box_preds,
            cls_preds,
            dir_preds,
            anchors,
        )
