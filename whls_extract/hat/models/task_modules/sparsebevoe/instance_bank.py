from typing import Any, Dict, Optional, Tuple, Union

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["InstanceBankOE"]


def topk(confidence, k, *inputs):
    """
    Perform top-k selection on confidence and gather corresponding inputs.

    Args:
        confidence: Confidence scores, shape (bs, N)
                    where bs is batch size and N is the number of items.
        k : Number of top-k elements to select.
        *inputs : Additional tensors to gather based on the indices of
                  top-k elements.

    Returns:
        A tuple containing:
            - confidence: Top-k confidence scores, shape (bs, k).
            - outputs : List of gathered tensors,
                        each of shape (bs, k, -1).

    """
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@OBJECT_REGISTRY.register
class InstanceBankOE(nn.Module):
    """
    Module for managing instance embeddings associated with anchors.

    Args:
        num_anchor : Number of anchors.
        embed_dims : Dimensionality of instance embeddings.
        anchor : Initial anchor points or path to anchor points.
        num_temp_instances : Number of temporal instances (default: 0).
        default_time_interval: Default time interval (default: 0.5).
        confidence_decay : Confidence decay rate (default: 0.6).
        anchor_grad : Whether anchor points are trainable
                      (default: True).
        max_time_interval : Maximum time interval (default: 2).
        projection_mat_key : Key for projection matrix
                             (default: "lidar2global").
    """

    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: Union[str, list, tuple, np.ndarray],
        anchor_dims: int = 11,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        max_time_interval: int = 2,
        projection_mat_key: str = "lidar2global",
        feat_grad: bool = False,
    ):
        super(InstanceBankOE, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.projection_mat_key = projection_mat_key
        self.anchor_dims = anchor_dims

        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple, np.ndarray)):
            anchor = np.array(anchor)
        elif anchor is None:
            anchor = np.zeros((num_anchor, self.anchor_dims))
        if len(anchor.shape) == 3:  # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

        self.prepare_quant_stub()

    def prepare_quant_stub(self):
        self.feature_cat = FloatFunctional()
        self.anchor_cat = FloatFunctional()

        self.feat_where = hnn.Where()
        self.anchor_where = hnn.Where()

        self.cls_quant_stub = QuantStub()
        self.anchor_quant_stub = QuantStub()
        self.instance_feature_quant_stub = QuantStub()
        self.mask_feature_quant_stub = QuantStub()
        self.dequant = DeQuantStub()

    def update_temp(self):
        return True

    def reset(self) -> None:
        """Reset instance features to initial values."""
        self.cached_feature = None
        self.cached_anchor = None
        self.cached_confidence = None
        self.mask = None
        self.metas = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def init_weight(self) -> None:
        """Initialize anchor weights with the initial anchor points."""
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    @staticmethod
    def anchor_projection(
        anchor: torch.Tensor,
        T_src2dst: torch.Tensor,
        time_interval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project anchors to a new coordinate frame.

        Args:
            anchor : Anchor points.
            T_src2dst : Transformation matrix from source
                        to destination frame.
            time_interval : Time interval for velocity adjustment
                            (default: None).

        Returns:
            Projected anchor points in the destination frame.
        """
        dst_anchor = anchor.clone()
        vel = anchor[..., 8:]
        T_src2dst = torch.unsqueeze(T_src2dst, dim=1)
        center = dst_anchor[..., 0:3]
        if time_interval is not None:
            translation = vel.transpose(0, -1) * time_interval.to(
                dtype=vel.dtype
            )
            translation = translation.transpose(0, -1)
            center = center + translation
        dst_anchor[..., 0:3] = (
            torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(
                dim=-1
            )
            + T_src2dst[..., :3, 3]
        )
        dst_anchor[..., [7, 6]] = torch.matmul(
            T_src2dst[..., :2, :2],
            dst_anchor[..., [7, 6], None],
        ).squeeze(-1)

        dst_anchor[..., 8:] = torch.matmul(
            T_src2dst[..., :3, :3], vel[..., None]
        ).squeeze(-1)
        return dst_anchor

    @staticmethod
    def gen_temp2cur(
        projection_mat_key: str,
        his_metas: Dict[str, Any],
        metas: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Generate transformation matrix from temporal to current frame.

        Args:
            projection_mat_key : Key for projection matrix in metadata.
            his_metas : Metadata from historical frame.
            metas : Metadata from current frame.

        Returns:
            Transformation matrix from temporal to current frame.
        """
        projection_mat = his_metas[projection_mat_key]
        src_projection_mat = projection_mat.cpu().numpy()
        dst_projection_mat = metas[projection_mat_key].cpu().numpy()
        temp2cur_numpy = np.linalg.inv(dst_projection_mat) @ src_projection_mat
        T_temp2cur = torch.tensor(temp2cur_numpy).float()
        return T_temp2cur

    def gen_cached_data(
        self,
        batch_size: int,
        metas: Dict[str, Any],
        dn_metas: Dict[str, Any],
        num_cached_instances: int,
    ) -> None:
        """
        Generate cached data.

        Args:
            batch_size: Batch size.
            metas: Metadata dictionary (default: None).
            dn_metas: Dense anchor metadata dictionary (default: None).
            num_cached_instances: Number of cached instances.
        """
        if self.metas is None or batch_size != self.cached_anchor.shape[0]:
            self.reset()
            self.cached_feature = torch.zeros(
                (batch_size, num_cached_instances, self.embed_dims)
            ).to(device=self.instance_feature.device)
            self.cached_anchor = torch.zeros(
                (batch_size, num_cached_instances, self.anchor_dims)
            ).to(device=self.instance_feature.device)
            self.mask = (
                torch.zeros((batch_size))
                .bool()
                .to(device=self.instance_feature.device)
            )
            self.cached_confidence = torch.zeros(
                (batch_size, num_cached_instances)
            ).to(device=self.instance_feature.device)

        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=self.instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            with torch.no_grad():
                T_temp2cur = InstanceBankOE.gen_temp2cur(
                    self.projection_mat_key, self.metas, metas
                ).to(device=self.instance_feature.device)
                self.cached_anchor = InstanceBankOE.anchor_projection(
                    self.cached_anchor,
                    T_temp2cur,
                    time_interval,
                )
                if (
                    dn_metas is not None
                    and batch_size == dn_metas["dn_anchor"].shape[0]
                ):
                    dn_anchor = dn_metas["dn_anchor"]
                    num_dn_group, num_dn = dn_anchor.shape[1:3]
                    dn_anchor = InstanceBankOE.anchor_projection(
                        dn_anchor.flatten(1, 2),
                        T_temp2cur,
                        time_interval=time_interval,
                    )
                    dn_metas["dn_anchor"] = dn_anchor.reshape(
                        batch_size, num_dn_group, num_dn, -1
                    )

    def get(
        self,
        batch_size: int,
        metas: Dict[str, Any],
        dn_metas: Dict[str, Any],
        compiler_model: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get instance features and anchors for the current batch.

        Args:
            batch_size: Batch size.
            metas: Metadata dictionary (default: None).
            dn_metas: Dense anchor metadata dictionary (default: None).
            compiler_model: Flag indicating if in compiler mode
                            (default: False).

        Returns:
            Tuple containing instance features and anchors.
        """
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )

        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if self.get_num_temp() <= 0:
            return instance_feature, anchor, None, None
        if compiler_model is True:
            self.cached_anchor = metas["cached_anchor"]
            self.cached_feature = metas["cached_feature"]
            self.mask = metas["mask"]
        else:
            self.gen_cached_data(
                batch_size, metas, dn_metas, self.get_num_temp()
            )

        anchor = self.anchor_quant_stub(anchor)
        instance_feature = self.instance_feature_quant_stub(instance_feature)
        temp_anchor = self.anchor_quant_stub(self.cached_anchor)
        temp_feature = self.instance_feature_quant_stub(self.cached_feature)

        return (
            instance_feature,
            anchor,
            temp_feature,
            temp_anchor,
        )

    def get_num_temp(self) -> int:
        """Get number of temporal instances."""
        return self.num_temp_instances

    def update(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update instance features and anchors based on the given inputs.

        Args:
            instance_feature : Instance features tensor.
            anchor : Anchor tensor.
            confidence : Confidence scores tensor.
            temp_instance_feature : Temporary instance features tensor.
            temp_anchor: Temporary anchor tensor.
        """
        num_temp_instances = self.get_num_temp()
        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - num_temp_instances
        confidence = confidence.max(dim=-1)[0]
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = self.feature_cat.cat(
            [temp_instance_feature, selected_feature], dim=1
        )
        selected_anchor = self.anchor_cat.cat(
            [temp_anchor, selected_anchor], dim=1
        )
        instance_feature = self.feat_where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = self.anchor_where(
            self.mask[:, None, None], selected_anchor, anchor
        )

        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )
        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        return instance_feature, anchor

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        self.anchor_cat.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )
        self.anchor_where.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )
        self.temp_anchor_quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )

    @staticmethod
    def cache_instance(
        num_temp_instances: int,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        cached_confidence,
        confidence_decay,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        confidence = confidence.max(dim=-1)[0]
        confidence = confidence.sigmoid()
        confidence[:, :num_temp_instances] = torch.maximum(
            cached_confidence[:, :num_temp_instances] * confidence_decay,
            confidence[:, :num_temp_instances],
        )

        (
            cached_confidence,
            (cached_feature, cached_anchor),
        ) = topk(confidence, num_temp_instances, instance_feature, anchor)
        return cached_confidence, cached_feature, cached_anchor

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        metas: Dict[str, Any],
    ) -> None:
        if isinstance(instance_feature, QTensor):
            self.instance_feature_quant_stub.activation_post_process.reset_dtype(  # noqa
                instance_feature.dtype, False
            )
            self.instance_feature_quant_stub.activation_post_process.set_qparams(  # noqa
                instance_feature.q_scale()
            )
        anchor = self.dequant(anchor)
        instance_feature = self.dequant(instance_feature)
        num_temp_instances = self.get_num_temp()
        (
            cached_confidence,
            cached_feature,
            cached_anchor,
        ) = InstanceBankOE.cache_instance(
            num_temp_instances,
            instance_feature,
            anchor,
            confidence,
            self.cached_confidence,
            self.confidence_decay,
        )
        self.metas = metas
        self.cached_confidence = cached_confidence
        self.cached_feature = cached_feature
        self.cached_anchor = cached_anchor

    def get_instance_id(
        self, confidence: torch.Tensor, anchor: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(
        self,
        instance_id: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
