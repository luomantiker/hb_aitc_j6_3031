from typing import List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

torch.fx.wrap("len")

__all__ = ["CenterPointDetector", "CenterPointDetectorIrInfer"]


@OBJECT_REGISTRY.register
class CenterPointDetector(nn.Module):
    """
    The basic structure of CenterPoint.

    Args:
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        pre_process: pre_process module.
        reader: reader module.
        scatter: scatter module.
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target generator module.
        loss: Loss module.
        postprocess: Postprocess module.
        quant_begin_neck: Whether to quantize beginning from neck.
        is_deploy: Is deploy model or not.
    """

    def __init__(
        self,
        feature_map_shape: List[int],
        pre_process: Optional[nn.Module] = None,
        reader: Optional[nn.Module] = None,
        scatter: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        targets: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        quant_begin_neck: bool = False,
        is_deploy: bool = False,
    ):
        super(CenterPointDetector, self).__init__()

        self.pre_process = pre_process
        self.reader = reader
        self.scatter = scatter
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.targets = targets
        self.loss = loss
        self.postprocess = postprocess
        self.quant_begin_neck = quant_begin_neck

        self.feature_map_shape = feature_map_shape
        self.is_deploy = is_deploy

    def forward(self, example):
        """Perform the forward pass of the model.

        Args:
            example: A dictionary containing the input data,
                including points or extracted features by deploy flag.

        Returns:
            results: Results produced by post_process.
        """

        # train: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> neck && Head -> output -> Loss
        # eval: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> voxel-feature -> neck && Head -> output -> post_process
        # deploy: -------------------------------------------------------
        #    [features, coors] -> FeatureExtractor -> PillarScatter
        #       -> voxel-feature -> neck && Head -> output

        if self.is_deploy:
            data = dict(  # noqa C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.feature_map_shape,
            )
        else:
            # use horizon pre_process
            features, coords = self.pre_process(
                example["points"], not self.training
            )
            data = dict(  # noqa C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.feature_map_shape,
            )

        # only support horizon_preprocess=True in reader for centerpoint
        input_features = self.reader(
            data["features"],
            horizon_preprocess=True,
        )
        if self.scatter is not None:
            x = self.scatter(
                input_features,
                data["coors"],
                data["batch_size"],
                torch.tensor(self.feature_map_shape),
            )
            x = self.backbone(x)
        else:
            x = self.backbone(
                input_features,
                data["coors"],
                data["batch_size"],
                torch.tensor(self.feature_map_shape),
            )
        x = self.neck(x)
        if self.head:
            outs = self.head(x)
        else:
            return x

        if self.is_deploy:
            outs_list = []
            for out in outs:
                for _, v in out.items():
                    outs_list.append(v)
            return outs_list

        return self._post_process(example, outs)

    @fx_wrap()
    def _post_process(self, example, outs):
        """
        Perform post-processing of model outputs.

        Args:
            example: A dictionary containing the input data.
            outs: Model outputs to be processed.

        Returns:
            results: Processed results, which can be losses during training or
                     bounding box results during evaluation.
        """
        if self.training:
            heatmaps, anno_boxes, inds, masks = self.targets(
                example["gt_boxes"], example["gt_classess"]
            )
            losses = self.loss(
                heatmaps,
                anno_boxes,
                inds,
                masks,
                outs,
            )
            return losses

        else:
            if self.postprocess is not None:
                bbox_results = self.postprocess(outs)
                return bbox_results
            else:
                return outs

    def fuse_model(self):
        """
        Fuse quantizable modules in the model.

        This function fuses quantizable modules within the model to prepare it
        for quantization.
        """
        if not self.quant_begin_neck:
            # P2
            if self.reader:
                self.reader.fuse_model()

        for module in [self.neck, self.head]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self):
        """
        Set quantization configurations for the model.

        This function sets quantization configurations for the model and its
        submodules. It configures quantization settings for different parts of
        the model based on the `quant_begin_neck` attribute.
        """
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.quant_begin_neck:
            # P1
            for module in [self.reader, self.backbone]:
                module.qconfig = None
        else:
            for module in [self.reader, self.backbone]:
                if module is not None:
                    if hasattr(module, "set_qconfig"):
                        module.set_qconfig()

        # P2
        for module in [self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        for module in [self.targets, self.loss, self.postprocess]:
            if module is not None:
                module.qconfig = None

    def set_calibration_qconfig(self):
        """
        Set calibration quantization configurations for the model.

        This function is deprecated by calibration_v2.
        """
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()

        if self.quant_begin_neck:
            # P1
            for module in [self.reader, self.backbone]:
                module.qconfig = None
        else:
            for module in [self.reader, self.backbone]:
                if module is not None:
                    if hasattr(module, "set_calibration_qconfig"):
                        module.set_calibration_qconfig()

        # P2
        for module in [self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_calibration_qconfig"):
                    module.set_calibration_qconfig()

        for module in [self.targets, self.loss, self.postprocess]:
            if module is not None:
                module.qconfig = None


@OBJECT_REGISTRY.register
class CenterPointDetectorIrInfer(nn.Module):
    """
    The basic structure of CenterPointIrInfer.

    Args:
        ir_model: The ir model.
        pre_process: pre_process module.
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        postprocess: Postprocess module.
        headkeys: The key of headoutputs.
        tasks: Task information including class number and class names.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        pre_process: nn.Module,
        feature_map_shape: List[int],
        postprocess: nn.Module,
        tasks: Optional[List[dict]],
        headkeys: List[str] = (
            "reg",
            "height",
            "dim",
            "rot",
            "vel",
            "heatmap",
        ),
    ):
        super().__init__()
        self.ir_model = ir_model
        self.pre_process = pre_process
        self.feature_map_shape = feature_map_shape
        self.postprocess = postprocess
        self.headkeys = headkeys
        self.tasks = tasks

    def forward(self, example):  # noqa: C408
        features, coords = self.pre_process(example["points"], True)
        data = dict(  # noqa C408
            features=features,
            coors=coords.to(torch.int32),
            num_points_in_voxel=None,
            batch_size=len(example["points"]),
            input_shape=self.feature_map_shape,
        )
        hbir_outputs = self.ir_model(data)
        results_convert = []
        start = 0
        for _ in range(len(self.tasks)):
            task_dict = {}
            for j, key in enumerate(self.headkeys):
                task_dict.update({key: hbir_outputs[start + j]})
            results_convert.append(task_dict)
            start += len(self.headkeys)
        bbox_results = self.postprocess(results_convert)
        return bbox_results
