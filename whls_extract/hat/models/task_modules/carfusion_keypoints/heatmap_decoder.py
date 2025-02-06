import torch

from hat.core.heatmap_decoder import decode_heatmap
from hat.models.base_modules.postprocess import PostProcessorBase
from hat.registry import OBJECT_REGISTRY

__all__ = ["HeatmapDecoder"]


@OBJECT_REGISTRY.register
class HeatmapDecoder(PostProcessorBase):
    """Decode heatmap prediction to landmark coordinates.

    Args:
        scale: Same as feat stride, the Scale of heatmap coordinates
                relative to the original image.
        mode: The decoder method, currently support "diff_sign" and "averaged"
            In the 'averaged' mode, the coordinates and heatmap values of the
            area surrounding the maximum point on the heatmap, with a size
            of k_size x k_size, are weighted to obtain the coordinates
            of the key point.
        k_size: kernel size used for "averaged" decoder.
    """

    def __init__(
        self,
        scale: int,
        mode: str = "diff_sign",
        k_size: int = 5,
    ):
        super(HeatmapDecoder, self).__init__()
        self.scale = scale
        self.mode = mode
        self.k_size = k_size

    def forward(self, heatmap: torch.Tensor):
        results = decode_heatmap(heatmap, self.scale, self.mode, self.k_size)
        return results
