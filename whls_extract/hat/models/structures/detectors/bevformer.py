# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional, Sequence

from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["BevFormer"]


@OBJECT_REGISTRY.register
class BevFormer(nn.Module):
    """The basic structure of BevFormer.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        view_transformer:
            View transformer module for transforming from img view to bev view.
        out_indices: Out indices for backbone.
        bev_decoders: Decoder for bev feature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        view_transformer: nn.Module = None,
        out_indices: Sequence[int] = (5,),
        bev_decoders: List[nn.Module] = None,
    ):
        super(BevFormer, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.view_transformer = view_transformer
        self.out_indices = out_indices
        self.bev_decoders = nn.ModuleList(bev_decoders)

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        x = [x[i] for i in self.out_indices]
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict) -> List:
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        bev_feat = self.view_transformer(feats, data)

        results = None
        for bev_decoder in self.bev_decoders:
            result = bev_decoder(bev_feat, data)
            results = self._update_res(result, results)
        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
        ]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()
        for m in self.bev_decoders:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        for m in self.bev_decoders:
            if hasattr(m, "set_qconfig"):
                m.set_qconfig()

    def export_reference_points(self, data: Dict) -> Dict:
        """Export refrence points.

        Args:
            data: A dictionary containing the input data.
            feat_wh: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """

        return self.view_transformer.get_reference_points_cam(data)


@OBJECT_REGISTRY.register
class BevFormerIrInfer(nn.Module):
    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module,
        model_convert_pipeline: List[callable],
        post_process: nn.Module = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.deploy_model = model_convert_pipeline(deploy_model)
        self.deploy_model.eval()
        self.bev_decoder_postprocess = post_process
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "ego2global": None,
        }

    def export_reference_points(self, data: Dict) -> Dict:

        return self.deploy_model.export_reference_points(data)

    def process_input(self, data):
        image = data["img"]
        ref_inputs = self.deploy_model.export_reference_points(data)

        temporal_inputs = self.deploy_model.view_transformer.get_deploy_prev(
            data, self.prev_frame_info
        )

        outputs = {
            "img": image,
            "prev_bev": temporal_inputs[0],
            "prev_bev_ref": temporal_inputs[1],
            "queries_rebatch_grid": ref_inputs[0],
            "restore_bev_grid": ref_inputs[1],
            "reference_points_rebatch": ref_inputs[2],
            "bev_pillar_counts": ref_inputs[3],
        }

        return outputs

    def forward(self, data):
        inputs = self.process_input(data)
        outputs = self.ir_model(inputs)

        bev_embed, outputs_classes, reference_out, outputs_coords = outputs
        outputs = self.deploy_model.bev_decoders[0].get_outputs(
            [outputs_classes], [reference_out], [outputs_coords], bev_embed
        )
        results = None
        result = self.deploy_model.bev_decoders[0].post_process(outputs)
        results = self._update_res(result, results)
        self.update_pre_info(data, bev_embed)
        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results

    def update_pre_info(
        self,
        data,
        bev_embed,
    ) -> List[List[Tensor]]:
        """Update prev bev info."""
        self.prev_frame_info["scene_token"] = data["seq_meta"][0]["meta"][0][
            "scene"
        ]
        self.prev_frame_info["ego2global"] = data["seq_meta"][0]["ego2global"]
        self.prev_frame_info["prev_bev"] = bev_embed
