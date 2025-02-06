# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Any, Dict, List, Optional, Union

import torch
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import nn

from hat.models.task_modules.sparsebevoe.head import SparseBEVOEHead
from hat.models.task_modules.sparsebevoe.instance_bank import InstanceBankOE
from hat.registry import OBJECT_REGISTRY

__all__ = ["SparseBevFusionOE"]


@OBJECT_REGISTRY.register
class SparseBevFusionOE(nn.Module):
    """
    SparseBEVFusionOE is a module for a sparse end-to-end bev model with lidar.

    Args:
        lidar_net: The lidar network for feature extraction.
        backbone : The backbone network for feature extraction.
        head : The module for processing extracted features.
        neck : Optional module for additional feature processing.
        depth_branch : Optional module for depth estimation.
        lidar_level_idx: The indices of lidar features to use.
        compiler_model : Indicates if the model is for compiler purposes.
        only_lidar: if True, only lidar is used for inference.
    """

    def __init__(
        self,
        lidar_net: nn.Module,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        depth_branch: Optional[nn.Module] = None,
        lidar_level_idx: List[int] = None,
        compiler_model: bool = False,
        only_lidar: bool = False,
    ):
        super(SparseBevFusionOE, self).__init__()
        self.lidar_net = lidar_net
        self.lidar_level_idx = lidar_level_idx
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.depth_branch = depth_branch
        self.compiler_model = compiler_model
        self.only_lidar = only_lidar

    def extract_feat(self, img: torch.Tensor) -> List:
        """
        Extract features from an input image.

        Args:
            img : Input image tensor.

        Returns:
            List: List of feature maps extracted.
        """
        feature_maps = self.backbone(img)
        feature_maps = list(self.neck(feature_maps))

        return feature_maps

    def _post_process(
        self,
        feature_maps: List,
        model_outs: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform post-processing on model outputs.

        Args:
            feature_maps : List of feature maps extracted from input image.
            model_outs : Output dictionary from the model.
            data : Input data dictionary.

        Returns:
            Output dictionary containing processed outputs or losses.
        """
        if self.compiler_model is False:
            if self.training is True and Tracer.is_tracing() is False:
                output = self.head.loss(model_outs, data)
                if self.depth_branch is not None and not self.only_lidar:
                    depths = self.depth_branch(feature_maps, data)
                    output["loss_dense_depth"] = self.depth_branch.loss(
                        depths, data
                    )
                return output
            else:
                results = self.head.post_process(model_outs)
                return results
        else:
            return model_outs

    def forward_lidar_feature(self, example: dict):
        if self.lidar_net.pre_process is None:
            data = dict(  # noqa C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.lidar_net.feature_map_shape,
            )
        else:
            features, coords = self.lidar_net.pre_process(
                example["points"], not self.training
            )
            data = dict(  # noqa C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.lidar_net.feature_map_shape,
            )

        input_features = self.lidar_net.reader(
            data["features"],
            horizon_preprocess=True,
        )
        x = self.lidar_net.scatter(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.lidar_net.feature_map_shape),
        )
        if self.lidar_net.backbone:
            x = self.lidar_net.backbone(x)
        if self.lidar_net.neck:
            x = self.lidar_net.neck(x)
        if self.lidar_net.head:
            x = self.lidar_net.head(x)
        if self.lidar_level_idx is not None:
            x = [x[i] for i in self.lidar_level_idx]
        return x

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        """
        Define the forward pass of the model.

        Args:
            data : Input data dictionary containing 'img'.

        Returns:
            Output dictionary or list containing processed outputs.
        """
        if not self.only_lidar:
            feature_maps = self.extract_feat(data["img"])
        else:
            feature_maps = None
        lidar_feature = self.forward_lidar_feature(data)
        model_outs = self.head(
            feature_maps,
            data,
            compiler_model=self.compiler_model,
            lidar_feature=lidar_feature,
        )
        return self._post_process(feature_maps, model_outs, data)

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class SparseBEVFusionOEIrInfer(nn.Module):
    """
    SparseBEVOEIrInfer is a module for performing inference.

    Args:
        ir_model: The ir model.
        lidar_preprocess: The lidar preprocess module.
        projection_mat_key: Key to retrieve projection matrix.
        gobel_mat_key: Key to retrieve Gobel matrix.
        first_frame_input: Input data for the first frame.
        max_time_interval: Maximum time interval allowed. Default is 2.
        decoder: Decoder module for processing model outputs.
        use_memory_bank: If True, use memory bank for storing instances.
        confidence_decay: Confidence decay factor.
        num_temp_instances: Number of temporal instances.
        num_memory_instances: Number of memory instances.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        lidar_preprocess: nn.Module,
        projection_mat_key: str,
        gobel_mat_key: str,
        first_frame_input: Dict[str, Any],
        max_time_interval: int = 2,
        decoder: nn.Module = None,
        use_memory_bank: bool = True,
        confidence_decay: float = 0.6,
        num_temp_instances: int = 128,
        num_memory_instances: int = 384,
    ):
        super(SparseBEVFusionOEIrInfer, self).__init__()

        self.ir_model = ir_model
        self.lidar_preprocess = lidar_preprocess
        self.first_frame_input = first_frame_input
        self.decoder = decoder
        self.metas = None
        self.projection_mat_key = projection_mat_key
        self.gobel_mat_key = gobel_mat_key
        self.max_time_interval = max_time_interval

        self.use_memory_bank = use_memory_bank
        self.confidence_decay = confidence_decay
        self.num_temp_instances = num_temp_instances
        self.num_memory_instances = num_memory_instances

    def cache(self, metas: Dict[str, Any], data: Dict[str, Any]) -> None:
        """
        Cache metadata and input data for future reference.

        Args:
            metas: Metadata associated with current input.
            data: Input data dictionary.
        """
        confidence = data["classification"]
        anchor = data["prediction"]
        instance_feature = data["feature"]
        (
            cached_confidence,
            cached_feature,
            cached_anchor,
        ) = InstanceBankOE.cache_instance(
            self.num_temp_instances,
            instance_feature,
            anchor,
            confidence,
            self.cached_confidence,
            self.confidence_decay,
        )
        self.metas = metas
        if self.use_memory_bank is True:
            self.cached_confidence = torch.cat(
                [cached_confidence, self.cached_confidence], dim=1
            )[:, : self.num_memory_instances]
            self.cached_feature = torch.cat(
                [cached_feature, self.cached_feature], dim=1
            )[:, : self.num_memory_instances]
            self.cached_anchor = torch.cat(
                [cached_anchor, self.cached_anchor], dim=1
            )[:, : self.num_memory_instances]
        else:
            self.cached_anchor = cached_anchor
            self.cached_feature = cached_feature
            self.cached_confidence = cached_confidence

    def forward(self, metas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform forward pass through the model.

        Args:
            metas: Metadata associated with current input.

        Returns:
            Output dictionary containing model predictions
            or tuple of decoded results depending on whether
            a decoder is provided.
        """
        input_data = {"img": metas["img"]}
        projection_mat = SparseBEVOEHead.gen_projection_mat(
            self.projection_mat_key, metas
        ).float()
        input_data["projection_mat"] = projection_mat
        features, coords = self.lidar_preprocess(metas["points"], True)
        input_data["features"] = features
        input_data["coors"] = coords.to(torch.int32)

        device = metas["img"].device
        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v.to(device))
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(device=self.cached_anchor.device)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBankOE.gen_temp2cur(
                self.gobel_mat_key, self.metas, metas
            ).to(device)
            self.cached_anchor = InstanceBankOE.anchor_projection(
                self.cached_anchor,
                T_temp2cur,
                time_interval,
            )
        input_data["cached_anchor"] = self.cached_anchor
        input_data["cached_feature"] = self.cached_feature
        if self.use_memory_bank is False:
            input_data["cached_confidence"] = self.cached_confidence
            input_data["mask"] = self.mask

        outs = self.ir_model(input_data)

        self.cache(metas, outs)
        results = self.decoder(
            outs["classification"], outs["prediction"], outs.get("quality")
        )
        return results
