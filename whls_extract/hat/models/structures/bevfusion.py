from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.apply_func import _as_list

__all__ = ["BevFusion", "BevFusionHbirInfer"]


@OBJECT_REGISTRY.register
class BevFusion(nn.Module):
    """BevFusion fuses LiDAR and camera data for BEV tasks.

    BevFusion is a module that fuses features from LiDAR and camera inputs to
    produce a Bird's Eye View (BEV) representation.
    The fused features are then passed through a series of decoders to generate
    the final outputs, which can be used for tasks like object detection,
    semantic segmentation, etc.

    Args:
        lidar_network: The neural network module for processing LiDAR data.
        camera_network: The neural network module for processing camera data.
        bev_decoders: A list of decoder modules that process
                the fused BEV features.
        fuse_module: Fusion module that combines LiDAR and camera features.
        bev_h: The height of the BEV feature map.
        bev_w: The width of the BEV feature map.
    """

    def __init__(
        self,
        lidar_network: Optional[nn.Module] = None,
        camera_network: Optional[nn.Module] = None,
        bev_decoders: Optional[List[nn.Module]] = None,
        fuse_module: Optional[nn.Module] = None,
        bev_h: int = 128,
        bev_w: int = 128,
    ):
        super().__init__()
        self.lidar_net = lidar_network
        self.camera_net = camera_network
        self.fuse_module = fuse_module
        self.bev_h = bev_h
        self.bev_w = bev_w
        assert lidar_network or camera_network, "at least has one input"

        self.bev_decoders = nn.ModuleList(bev_decoders)

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
        x = self.lidar_net.backbone(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.lidar_net.feature_map_shape),
        )
        x = self.lidar_net.neck(x)
        return x

    def forward_camera_feature(self, data: dict):
        imgs = data["img"]
        feats = self.camera_net.extract_feat(imgs)
        bev_feat = self.camera_net.view_transformer(feats, data)
        return bev_feat

    def forward(self, data: dict):
        if self.lidar_net:
            lidar_feature = self.forward_lidar_feature(data)

        if self.camera_net:
            camera_feature = self.forward_camera_feature(data)
            B, _, C = camera_feature.shape
            cx = camera_feature.permute(0, 2, 1).reshape(
                B, C, self.bev_h, self.bev_w
            )
        if self.camera_net and self.lidar_net:
            if cx.shape[-1] != self.bev_w:
                cx = F.interpolate(cx, (self.bev_h, self.bev_w))
            cat_bev = torch.cat([cx, lidar_feature], dim=1)
            pts_feats = self.fuse_module(cat_bev)
            B, C = pts_feats.shape[:2]
        elif self.camera_net:
            pts_feats = cx
        else:
            pts_feats = lidar_feature
            B, C = pts_feats.shape[:2]

        results = self.post_process(data, pts_feats)
        return results

    def post_process(self, data: dict, pts_feats: torch.Tensor):
        if self.training:
            loss = {}
            for bev_decoder in self.bev_decoders:
                result = bev_decoder(pts_feats, data)
                loss.update(result)
            return loss
        else:
            results = []
            for bev_decoder in self.bev_decoders:
                result = bev_decoder(pts_feats, data)
                results.append(result)
            return results

    def set_qconfig(self):
        all_modules = []
        if self.lidar_net:
            all_modules += [
                self.lidar_net.reader,
                self.lidar_net.backbone,
                self.lidar_net.neck,
                self.lidar_net.head,
                self.lidar_net.reader,
            ]
        if self.camera_net:
            all_modules += [
                self.camera_net.backbone,
                self.camera_net.neck,
                self.camera_net.view_transformer,
            ]
        for module in [
            *all_modules,
            *self.bev_decoders,
        ]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()

        int16_modules = [
            self.bev_decoders[0].quant_object_query_embed,
            self.bev_decoders[0].reference_points,
        ]

        for m in int16_modules:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )


@OBJECT_REGISTRY.register
class BevFusionHbirInfer(nn.Module):
    """Inference module for BEV fusion with HBIR.

    Args:
        ir_model: ir model.
        deploy_model: The deployed model for inference.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.deploy_model = deploy_model
        self.deploy_model.eval()
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "ego2global": None,
        }

    def export_reference_points(self, data: Dict) -> Dict:
        return self.deploy_model.camera_net.export_reference_points(data)

    def process_input(self, data):
        image = data["img"]
        features, coords = self.deploy_model.lidar_net.pre_process(
            data["points"], True
        )

        (
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        ) = self.export_reference_points(data)

        return {
            "features": features,
            "coors": coords.to(torch.int32),
            "img": image,
            "queries_rebatch_grid": queries_rebatch_grid,
            "restore_bev_grid": restore_bev_grid,
            "reference_points_rebatch": reference_points_rebatch,
            "bev_pillar_counts": bev_pillar_counts,
        }
        # return (features, coords.to(torch.int32)) + (image,) + ref_inputs

    def forward(self, data):
        inputs = self.process_input(data)
        outputs = self.ir_model(inputs)

        bev_embed, outputs_classes, reference_out, outputs_coords = outputs[0]
        bev_outputs = self.deploy_model.bev_decoders[0].get_outputs(
            [outputs_classes], [reference_out], [outputs_coords], bev_embed
        )
        results = None
        result = self.deploy_model.bev_decoders[0].post_process(bev_outputs)
        bev_results = self._update_res(result, results)

        occ_outputs = outputs[1]
        occ_score = occ_outputs[0].softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        return bev_results, [occ_res]

    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results
