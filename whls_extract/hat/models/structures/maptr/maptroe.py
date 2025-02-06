import logging
from typing import Dict, List, Optional, Sequence

from torch import Tensor, nn

from hat.models.task_modules.maptr.sparse_head import SparseOMOEHead
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["MapTROE"]


@OBJECT_REGISTRY.register
class MapTROE(nn.Module):
    """The general structure of the MapTR series (w or w/o sd map fusion).

    Args:
        backbone: Backbone module.
        out_indices: Out indices for backbone.
        neck: Neck module.
        view_transformer:
            View transformer module for transforming from img view to bev view.
        sd_map_fusion: Fuse sd map. Default is False.
        osm_encoder:
            Module for extracting osm(OpenStreetMap) features from osm mask.
            It can't be None if sd map fusion is performed. Default is None.
        bev_fusion: Module for fusing bev features and osm features.
            It can't be None if sd map fusion is performed. Default is None.
        bev_decoders: Decoder for fused bev features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_indices: Sequence[int] = (5,),
        neck: Optional[nn.Module] = None,
        view_transformer: nn.Module = None,
        sd_map_fusion: bool = False,
        osm_encoder: Optional[nn.Module] = None,
        bev_fusion: Optional[nn.Module] = None,
        bev_decoders: List[nn.Module] = None,
    ):
        super(MapTROE, self).__init__()
        self.backbone = backbone
        self.out_indices = out_indices
        self.neck = neck
        self.view_transformer = view_transformer
        self.sd_map_fusion = sd_map_fusion
        self.osm_encoder = osm_encoder
        self.bev_fusion = bev_fusion
        self.bev_decoders = nn.ModuleList(bev_decoders)

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        x = [x[i] for i in self.out_indices]
        if self.neck is not None:
            x = self.neck(x)
        return x

    def extract_osm_feat(self, osm: Tensor) -> List[Tensor]:
        """Directly extract features from osm."""
        osm_feat = self.osm_encoder(osm)
        return osm_feat

    def forward(self, data: Dict) -> List:
        imgs = data["img"]
        feats = self.extract_feat(imgs)

        if self.view_transformer is not None:
            bev_feat = self.view_transformer(feats, data)
        else:
            bev_feat = None

        if self.sd_map_fusion:
            osms = data["osm_mask"]
            assert osms is not None
            osms_feat = self.extract_osm_feat(osms)
            bev_fusion = self.bev_fusion(bev_feat, osms_feat)
        else:
            bev_fusion = bev_feat
        results = None
        for bev_decoder in self.bev_decoders:
            result = bev_decoder(bev_fusion, feats, data)
            results = self._update_res(result, results)
        return results

    @fx_wrap()
    def _update_res(self, result, results):
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results

    def export_reference_points(self, data: Dict) -> Dict:
        """Export refrence points.

        Args:
            data: A dictionary containing the input data.

        Returns:
            The Reference points.
        """

        return self.view_transformer.get_reference_points_cam(data)

    def set_qconfig(self):
        for m in self.bev_decoders:
            if hasattr(m, "set_qconfig"):
                m.set_qconfig()


@OBJECT_REGISTRY.register
class MapTROEIrInfer(nn.Module):
    """
    The basic structure of MapTROEIrInfer.

    Args:
        ir_model: The ir model.
        test_model: The model to process input and get outputs.
        model_convert_pipeline: Define the process of test model convert.
        sd_map_fusion: Fuse sd map. Default is False.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        test_model: nn.Module,
        model_convert_pipeline: List[callable],
        sd_map_fusion: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.test_model = model_convert_pipeline(test_model)
        self.test_model.eval()
        self.sd_map_fusion = sd_map_fusion

    def process_input(self, data):
        image = data["img"]
        (
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        ) = self.test_model.export_reference_points(data)
        if self.sd_map_fusion:
            return {
                "img": image,
                "osm_mask": data["osm_mask"],
                "queries_rebatch_grid": queries_rebatch_grid,
                "restore_bev_grid": restore_bev_grid,
                "reference_points_rebatch": reference_points_rebatch,
                "bev_pillar_counts": bev_pillar_counts,
            }
        else:
            return {
                "img": image,
                "queries_rebatch_grid": queries_rebatch_grid,
                "restore_bev_grid": restore_bev_grid,
                "reference_points_rebatch": reference_points_rebatch,
                "bev_pillar_counts": bev_pillar_counts,
            }

    def forward(self, data):
        inputs = self.process_input(data)
        device = inputs["img"].device
        outputs = self.ir_model(inputs)

        if isinstance(outputs[0], Tensor):
            outputs_classes, reference_out = outputs
        else:
            assert isinstance(outputs[0], dict)
            outputs_classes = outputs[0]["classification"]
            reference_out = outputs[0]["prediction"]
        outputs_classes = outputs_classes.to(device)
        reference_out = reference_out.to(device)

        outputs = self.test_model.bev_decoders[0].get_outputs(
            outputs_classes=[outputs_classes],
            reference_out=[reference_out],
        )
        results = None
        result = self.test_model.bev_decoders[0].post_process(outputs, data)
        results = self._update_res(result, results)

        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results


@OBJECT_REGISTRY.register
class SparseMapIrInfer(MapTROEIrInfer):
    def __init__(
        self,
        projection_mat_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projection_mat_key = projection_mat_key

    def process_input(self, data):
        image = data["img"]
        projection_mat = SparseOMOEHead.gen_projection_mat(
            self.projection_mat_key, data
        ).float()

        return {
            "img": image,
            "projection_mat": projection_mat,
        }
