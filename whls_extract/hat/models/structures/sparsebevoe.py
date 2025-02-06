# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Any, Dict, List, Optional, Union

import torch
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import nn

from hat.models.task_modules.sparsebevoe.head import SparseBEVOEHead
from hat.models.task_modules.sparsebevoe.instance_bank import InstanceBankOE
from hat.registry import OBJECT_REGISTRY

__all__ = ["SparseBEVOE"]


@OBJECT_REGISTRY.register
class SparseBEVOE(nn.Module):
    """
    SparseBEVOE is a module for a sparse end-to-end bev model.

    Args:
        backbone : The backbone network for feature extraction.
        head : The module for processing extracted features.
        neck : Optional module for additional feature processing.
        depth_branch : Optional module for depth estimation.
        compiler_model : Indicates if the model is for compiler purposes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        depth_branch: Optional[nn.Module] = None,
        compiler_model: bool = False,
    ):
        super(SparseBEVOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.depth_branch = depth_branch
        self.compiler_model = compiler_model

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
                if self.depth_branch is not None:
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

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        """
        Define the forward pass of the model.

        Args:
            data : Input data dictionary containing 'img'.

        Returns:
            Output dictionary or list containing processed outputs.
        """
        feature_maps = self.extract_feat(data["img"])

        model_outs = self.head(feature_maps, data, self.compiler_model)
        return self._post_process(feature_maps, model_outs, data)

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class SparseBEVOEIrInfer(nn.Module):
    """
    SparseBEVOEIrInfer is a module for performing inference.

    Args:
        ir_model: The ir model.
        projection_mat_key: Key to retrieve projection matrix.
        gobel_mat_key: Key to retrieve Gobel matrix.
        first_frame_input: Input data for the first frame.
        max_time_interval: Maximum time interval allowed. Default is 2.
        decoder: Decoder module for processing model outputs.
    """

    def __init__(
        self,
        ir_model: nn.Module,
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
        super(SparseBEVOEIrInfer, self).__init__()

        self.ir_model = ir_model
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
        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v)
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(device=self.cached_anchor.device)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBankOE.gen_temp2cur(
                self.gobel_mat_key, self.metas, metas
            )
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
