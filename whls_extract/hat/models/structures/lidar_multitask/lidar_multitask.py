from typing import List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

torch.fx.wrap("len")

__all__ = ["LidarMultiTask", "LidarMultiTaskIrInfer"]


@OBJECT_REGISTRY.register
class LidarMultiTask(nn.Module):
    """
    The basic structure of LidarMultiTask.

    Args:
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        pre_process: Pre-process module.
        reader: Reader module.
        scatter: Scatter module.
        backbone: Backbone module.
        neck: Neck module.
        lidar_decoders: List of Lidar Decoder modules.
        quant_begin_backbone: Whether to quantize beginning from the backbone.
        is_deploy: Is it a deploy model or not.
    """

    def __init__(
        self,
        feature_map_shape: List[int],
        pre_process: Optional[nn.Module] = None,
        reader: Optional[nn.Module] = None,
        scatter: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        lidar_decoders: List[nn.Module] = None,
        quant_begin_backbone: bool = False,
        is_deploy: bool = False,
    ):
        super(LidarMultiTask, self).__init__()

        self.pre_process = pre_process
        self.reader = reader
        self.scatter = scatter
        self.backbone = backbone
        self.neck = neck
        self.lidar_decoders = nn.ModuleList(lidar_decoders)
        self.quant_begin_backbone = quant_begin_backbone

        self.feature_map_shape = feature_map_shape
        self.is_deploy = is_deploy

    def forward(self, example):
        """
        Forward pass through the LidarMultiTask model.

        Args:
            example: Input data dictionary containing "points" and other
            relevant information.

        Returns:
            preds: Model predictions.
            results: Additional results if available.
        """
        # train: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> pseudo-image -> backbone & neck & Head -> output -> Loss
        # eval: [points] -> PreProcess -> FeatureExtractor -> PillarScatter
        #   -> pseudo-image -> backbone & neck & Head -> output -> post_process
        # deploy: -------------------------------------------------------
        #    [features, coors] -> FeatureExtractor -> PillarScatter
        #       -> pseudo-image -> backbone & neck & Head -> output

        if self.pre_process:
            # use horizon pre_process
            features, coords = self.pre_process(
                example["points"], self.is_deploy or not self.training
            )
            data = dict(  # noqa C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.feature_map_shape,
            )
        else:
            data = dict(  # noqa C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.feature_map_shape,
            )

        input_features = self.reader(
            data["features"],
            data["coors"],
            data["num_points_in_voxel"],
            horizon_preprocess=True,
        )

        x = self.scatter(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.feature_map_shape),
        )
        if self.backbone is not None:
            x = self.backbone(x)
        x = self.neck(x)

        preds = None
        results = {}
        for lidar_decoder in self.lidar_decoders:
            pred, result = lidar_decoder(x, example)
            preds, results = self._update_res(pred, result, preds, results)

        if self.is_deploy:
            return preds

        return preds, results

    @fx_wrap()
    def _update_res(self, pred, result, preds, results):
        """
        Update predictions and results.

        Args:
            pred: Model predictions.
            result: Additional results.
            preds: Accumulated predictions.
            results: Accumulated results.

        Returns:
            preds: Updated predictions.
            results: Updated results.
        """
        if preds is None:
            preds = pred
        else:
            preds.extend(pred)

        if result:
            results.update(result)
        return preds, results

    def fuse_model(self):
        """Fuse model operations for quantization."""
        if not self.quant_begin_backbone:
            # P2
            if self.reader:
                self.reader.fuse_model()

        for module in [
            self.backbone,
            self.neck,
            *self.lidar_decoders,
        ]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self):
        """Set quantization configuration for the model."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.quant_begin_backbone:
            # P1
            for module in [self.reader, self.scatter]:
                module.qconfig = None
        else:
            for module in [self.reader, self.scatter]:
                if module is not None:
                    if hasattr(module, "set_qconfig"):
                        module.set_qconfig()

        # P2
        for module in [
            self.backbone,
            self.neck,
            *self.lidar_decoders,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()


@OBJECT_REGISTRY.register
class LidarMultiTaskIrInfer(nn.Module):
    """
    The basic structure of LidarMultiTaskIrInfer.

    Args:
        ir_model: The ir model.
        pre_process: pre_process module.
        feature_map_shape: Feature map shape, in (W, H, 1) format.
        lidar_decoders: Lidar decoder module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        pre_process: nn.Module,
        feature_map_shape: List[int],
        lidar_decoders: List[nn.Module],
    ):
        super().__init__()
        self.ir_model = ir_model
        self.pre_process = pre_process
        self.feature_map_shape = feature_map_shape
        self.lidar_decoders = nn.ModuleList(lidar_decoders)

    def forward(self, example):
        features, coords = self.pre_process(example["points"], True)
        data = dict(  # noqa C408
            features=features,
            coors=coords.to(torch.int32),
            num_points_in_voxel=None,
            batch_size=len(example["points"]),
            input_shape=self.feature_map_shape,
        )

        hbir_outputs = self.ir_model(data)

        seg_out = self.lidar_decoders[0](hbir_outputs[0])
        preds = [hbir_outputs[0]]
        result = {}
        result["seg"] = seg_out
        outs = []
        for i in range(6):
            tmp = {}
            tmp["reg"] = hbir_outputs[1 + i * 6]
            tmp["height"] = hbir_outputs[1 + i * 6 + 1]
            tmp["dim"] = hbir_outputs[1 + i * 6 + 2]
            tmp["rot"] = hbir_outputs[1 + i * 6 + 3]
            tmp["vel"] = hbir_outputs[1 + i * 6 + 4]
            tmp["heatmap"] = hbir_outputs[1 + i * 6 + 5]

            outs.append(tmp)
        preds.append(outs)
        result["det"] = self.lidar_decoders[1](outs)
        return preds, result
