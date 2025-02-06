# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)
from torch.cuda.amp import autocast
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import reduce_mean
from .blocks import QINT16_MAX, DecoupledMultiheadAttention

__all__ = ["SparseBEVOEHead"]


@OBJECT_REGISTRY.register
class SparseBEVOEHead(nn.Module):
    """
    Module for handling object detection and 3D object processing.

    Args:
        instance_bank : Instance bank module for managing instance features.
        anchor_encoder : Anchor encoder module for encoding anchor features.
        ffn : Feedforward network module for processing encoded features.
        deformable_model : Deformable model module for
                            modeling deformable aspects.
        refine_layer : Refinement layer module for refining output predictions.
        num_decoder : Number of decoders. Defaults to 6.
        num_single_frame_decoder : Number of single frame decoders.
                                   Defaults to -1.
        embed_dims : Dimension of embeddings. Defaults to 256.
        num_heads : Number of attention heads. Defaults to 8.
        num_views : Number of views. Defaults to 6.
        level_index : List of level indices. Defaults to None.
        loss_cls : Classification loss module. Defaults to None.
        loss_reg : Regression loss module. Defaults to None.
        loss_cns : CNS loss module. Defaults to None.
        loss_yns : YNS loss module. Defaults to None.
        decoder : Decoder function. Defaults to None.
        target : Target function. Defaults to None.
        gt_key : Key for ground truth data.
                 Defaults to "lidar_bboxes_labels".
        reg_weights : List of regression weights. Defaults to None.
        operation_order: List of operation orders. Defaults to None.
        cls_threshold_to_reg: Classification threshold for regression.
                              Defaults to -1.0.
        enable_dn : Flag to enable dn. Defaults to False.
        enable_temp_dn : Flag to enable temporal dn. Defaults to False.
        cls_allow_reverse : Allow reverse classification flag.
                            Defaults to None.
        projection_mat_key : Key for projection matrix.
                             Defaults to "lidar2img".
    """

    def __init__(
        self,
        instance_bank: nn.Module,
        anchor_encoder: nn.Module,
        ffn: nn.Module,
        deformable_model: nn.Module,
        refine_layer: nn.Module,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_views: int = 6,
        level_index: List = None,
        loss_cls: Optional[nn.Module] = None,
        loss_reg: Optional[nn.Module] = None,
        loss_cns: Optional[nn.Module] = None,
        loss_yns: Optional[nn.Module] = None,
        decoder: Optional[Callable] = None,
        target: Optional[Callable] = None,
        gt_key: str = "lidar_bboxes_labels",
        reg_weights: Optional[List[float]] = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1.0,
        enable_dn: bool = False,
        enable_temp_dn: bool = False,
        cls_allow_reverse: int = None,
        projection_mat_key: str = "lidar2img",
        lidar_only: str = False,
    ):
        super(SparseBEVOEHead, self).__init__()
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_key = gt_key
        self.projection_mat_key = projection_mat_key

        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.enable_dn = enable_dn
        self.enable_temp_dn = enable_temp_dn
        self.level_index = level_index

        self.reg_weights = [1.0] * 10 if reg_weights is None else reg_weights
        if operation_order is None:
            operation_order = [
                "interaction",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        self.dropout = 0.1
        self.fc_before = nn.Linear(embed_dims, embed_dims * 2, bias=False)
        self.fc_after = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        temp_instance_interaction = DecoupledMultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        instance_interaction = DecoupledMultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        norm_layer = nn.LayerNorm(embed_dims)
        self.op_map = {
            "temp_interaction": [temp_instance_interaction, 0],
            "interaction": [instance_interaction, 0],
            "norm": [norm_layer, 0],
            "ffn": [ffn, 0],
            "deformable": [deformable_model, 0],
            "lidar_deformable": [deformable_model, 0],
            "refine": [refine_layer, 0],
        }
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op not in self.op_map:
                self.layers.append(None)
            elif self.op_map[op][1] == 0:
                self.layers.append(self.op_map[op][0])
            else:
                self.layers.append(copy.deepcopy(self.op_map[op][0]))
            self.op_map[op][1] += 1
        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.target = target
        self.decoder = decoder
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_cns = loss_cns
        self.loss_yns = loss_yns
        self.num_views = num_views
        self.init_weights()
        self.cls_allow_reverse = cls_allow_reverse
        self.mat_quant_stub = QuantStub()

        self.dequant = DeQuantStub()

    def init_weights(self) -> None:
        """Initialize weights."""
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    @staticmethod
    def gen_projection_mat(
        projection_mat_key: str, metas: Dict[str, Any]
    ) -> torch.Tensor:
        """Modify the given homography matrix based on the feature and image shapes.

        Args:
            projection_mat_key: key in the metas for projection_mat.
            feats: Feature tensor.
            meta: Metadata dictionary.

        Returns:
            homopgrahpy: Modified homography matrix.
        """
        scales = [np.array(img.shape[::-1][:2]) for img in metas["img"]]
        projection_mat = metas[projection_mat_key]

        views = []
        for scale in scales:
            view = np.eye(4)
            view[0, 0] = 2 / scale[0]
            view[1, 1] = 2 / scale[1]
            view[0, 2] = -1
            view[1, 2] = -1
            views.append(torch.tensor(view).to(device=projection_mat.device))
        views = torch.stack(views)
        projection_mat = torch.matmul(views.double(), projection_mat.double())
        return projection_mat

    @autocast(enabled=False)
    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: Dict[str, Any],
        compiler_model: str = False,
        lidar_feature=None,
    ) -> torch.Tensor:
        """
        Forward pass of the SparseBEVOEHead module.

        Args:
            feature_maps: Input feature maps.If a single torch.Tensor,
                          it's converted to a List[torch.Tensor].
            metas : Metadata dictionary containing additional information.
            compiler_model : Flag indicating whether to compile the model
                             or not.
        """

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]

        if self.level_index is not None:
            feature_maps_head = []
            for index in self.level_index:
                if compiler_model is False:
                    feature_maps_head.append(feature_maps[index].float())
                else:
                    feature_maps_head.append(feature_maps[index])
            feature_maps = feature_maps_head
        if feature_maps is not None:
            batch_size = feature_maps[0].shape[0] // self.num_views
        else:
            batch_size = lidar_feature.shape[0]

        dn_metas = None
        if (
            self.target is not None
            and self.target.dn_metas is not None
            and self.target.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.target.dn_metas = None

        if compiler_model is False:
            dn_metas = self.target.dn_metas

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
        ) = self.instance_bank.get(batch_size, metas, dn_metas, compiler_model)
        dn_metas = None
        attn_mask = None
        temp_dn_reg_target = None
        # prepare for denosing training
        if self.enable_dn:
            if self.training and hasattr(self.target, "get_dn_anchors"):
                if "instance_ids" in metas:
                    gt_instance_id = []
                    for instance_id in metas["instance_ids"]:
                        instance_id = np.array(instance_id, dtype=np.int64)
                        gt_instance_id.append(
                            torch.from_numpy(instance_id).cuda()
                        )
                else:
                    gt_instance_id = None
                dn_metas = self.target.get_dn_anchors(
                    *self.parse_labels(metas), gt_instance_id
                )
            if dn_metas is not None:
                (
                    dn_anchor,
                    dn_reg_target,
                    dn_cls_target,
                    dn_attn_mask,
                    valid_mask,
                    dn_id_target,
                ) = dn_metas
                num_dn_anchor = dn_anchor.shape[1]
                if dn_anchor.shape[-1] != anchor.shape[-1]:
                    remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]

                    dn_anchor = torch.cat(
                        [
                            dn_anchor,
                            dn_anchor.new_zeros(
                                batch_size, num_dn_anchor, remain_state_dims
                            ),
                        ],
                        dim=-1,
                    )
                anchor = torch.cat([anchor, dn_anchor], dim=1)
                instance_feature = torch.cat(
                    [
                        instance_feature,
                        instance_feature.new_zeros(
                            batch_size,
                            num_dn_anchor,
                            instance_feature.shape[-1],
                        ),
                    ],
                    dim=1,
                )
                num_instance = instance_feature.shape[1]
                num_free_instance = num_instance - num_dn_anchor
                attn_mask = anchor.new_ones(
                    (num_instance, num_instance), dtype=torch.bool
                )
                attn_mask[:num_free_instance, :num_free_instance] = False
                attn_mask[
                    num_free_instance:, num_free_instance:
                ] = dn_attn_mask

        # generate anchor embed
        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None
        bs, _, _ = anchor.shape
        if compiler_model is True:
            projection_mat = metas["projection_mat"]
        else:
            projection_mat = SparseBEVOEHead.gen_projection_mat(
                self.projection_mat_key, metas
            )
        projection_mat = self.mat_quant_stub(
            projection_mat.view(bs, -1, 4, 4).float()
        )

        # forward layers
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_interaction":
                value = temp_instance_feature
                if temp_instance_feature is not None:
                    value = self.fc_before(value)
                instance_feature = self.layers[i](
                    instance_feature,
                    temp_instance_feature,
                    value,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
                instance_feature = self.fc_after(instance_feature)
            elif op == "interaction":
                value = instance_feature
                value = self.fc_before(value)
                instance_feature = self.layers[i](
                    instance_feature,
                    instance_feature,
                    value=value,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
                instance_feature = self.fc_after(instance_feature)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    projection_mat,
                )
            elif op == "lidar_deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    lidar_feature,
                    projection_mat,
                )

            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(self.dequant(anchor))
                classification.append(self.dequant(cls))
                quality.append(self.dequant(qt))
                if (
                    len(prediction) == self.num_single_frame_decoder
                    and self.num_single_frame_decoder > 0
                ):
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature,
                        anchor,
                        cls,
                        temp_instance_feature,
                        temp_anchor,
                    )
                    if (
                        dn_metas is not None
                        and self.target.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.target.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)

                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                    and self.instance_bank.update_temp()
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.get_num_temp()
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}
        cls = self.dequant(cls)
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
            self.target.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        if compiler_model is True:
            output.update(
                {
                    "classification": classification[-1],
                    "prediction": prediction[-1],
                    "quality": quality[-1],
                    "feature": instance_feature,
                }
            )
        else:
            output.update(
                {
                    "classification": classification,
                    "prediction": prediction,
                    "quality": quality,
                }
            )
            self.instance_bank.cache(
                instance_feature,
                anchor,
                cls,
                metas,
            )
        if Tracer.is_tracing() is True:
            self.instance_bank.reset()
        return output

    def parse_labels(self, data):
        device = data["img"].device
        gt_cls = []
        gt_reg = []
        for _, labels in enumerate(data[self.gt_key]):
            if len(labels) == 0:
                cls = []
                reg = torch.zeros((1, 9), device=device).float()
            else:
                cls = labels[:, 9]
                reg = torch.tensor(labels[:, :9], device=device).float()
            gt_cls.append(torch.tensor(cls, device=device).long())
            gt_reg.append(reg)
        return gt_cls, gt_reg

    @autocast(enabled=False)
    def loss(
        self,
        model_outs: Dict[str, torch.Tensor],
        data: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for the SparseBEVOEHead module.

        Args:
            model_outs: Dictionary containing model outputs,
                including 'classification', 'prediction', and 'quality'.
            data : Dictionary containing ground truth data.

        Returns:
            Dictionary containing computed loss values.

        """
        gt_cls, gt_reg = self.parse_labels(data)

        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, cls_weights, reg_weights = self.target(
                cls,
                reg,
                gt_cls,
                gt_reg,
                None,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask = torch.logical_and(mask, cls_weights != 0)
            if mask.is_cuda:
                num_pos = max(
                    reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
                )
            else:
                num_pos = max(torch.sum(mask).to(dtype=reg.dtype), 1.0)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weights = cls_weights.flatten(end_dim=1)
            cls_loss = self.loss_cls(
                cls, cls_target, avg_factor=num_pos, weight=cls_weights
            )
            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            cls = cls[mask]

            if self.cls_allow_reverse is not None and cls_target.shape[0] != 0:
                if_reverse = (
                    torch.nn.functional.cosine_similarity(
                        reg_target[..., 6:8],
                        reg[..., 6:8],
                        dim=-1,
                    )
                    < 0
                )
                if_reverse = (
                    torch.isin(
                        cls_target,
                        cls_target.new_tensor(self.cls_allow_reverse),
                    )
                    & if_reverse
                )
                reg_target[..., 6:8] = torch.where(
                    if_reverse[..., None],
                    -reg_target[..., 6:8],
                    reg_target[..., 6:8],
                )
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            )
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            )
            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
                cns = qt[..., 0]
                yns = qt[..., 1].sigmoid()
                cns_target = torch.norm(
                    reg_target[..., :3] - reg[..., :3], p=2, dim=-1
                )
                cns_target = torch.exp(-cns_target)
                cns_loss = self.loss_cns(cns, cns_target, avg_factor=num_pos)
                output[f"loss_cns_{decoder_idx}"] = cns_loss
                yns_target = (
                    torch.nn.functional.cosine_similarity(
                        reg_target[..., 6:8],
                        reg[..., 6:8],
                        dim=-1,
                    )
                    > 0
                )
                yns_target = yns_target.float()
                yns_loss = self.loss_yns(yns, yns_target)
                output[f"loss_yns_{decoder_idx}"] = yns_loss

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)

        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
            )
            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            )
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            )
            output.update(
                {
                    f"dn_loss_cls_{decoder_idx}": cls_loss,
                    f"dn_loss_reg_{decoder_idx}": reg_loss,
                }
            )
        return output

    def prepare_for_dn_loss(
        self, model_outs: Dict[str, torch.Tensor], prefix: str = ""
    ) -> Tuple[torch.Tensor]:
        """
        Prepare inputs for domain-specific loss calculation.

        Args:
            model_outs : Dictionary containing model outputs
                         with keys including '{prefix}dn_valid_mask',
                         '{prefix}dn_cls_target', and '{prefix}dn_reg_target'.
            prefix : Prefix used to retrieve keys from model_outs.
                     Defaults to "".

        Returns:
            Tuple containing noise classification and regression target.
        """
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]

        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )

        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        self.fc_after.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 50 / QINT16_MAX,
            },
        )
        self.fc_before.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 5 / QINT16_MAX,
            },
        )
        self.mat_quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )
        self.instance_bank.set_qconfig()
        for mod in self.layers:
            if hasattr(mod, "set_qconfig"):
                mod.set_qconfig()

    def post_process(
        self, model_outs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform post-processing on model outputs.

        Args:
            model_outs (dict): Dictionary containing model outputs with keys:
                - 'classification': Tensor of classification scores.
                - 'prediction': Tensor of prediction outputs.
                - 'quality' (optional): Tensor of quality estimation outputs.

        Returns:
            Processed outputs, typically decoded predictions.

        """
        return self.decoder(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("quality"),
        )
