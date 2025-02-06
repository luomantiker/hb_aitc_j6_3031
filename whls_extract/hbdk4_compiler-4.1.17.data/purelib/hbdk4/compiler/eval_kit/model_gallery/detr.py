import torch
from hbdk4.compiler.eval_kit.model_gallery.utils import Detector
from hbdk4.compiler.eval_kit import trace


def retrieve_detr(backbone, sizes, pretrained=False, post_process=False):
    """
    get detr with coco detection head, with pretrained weights with post process graph
      backbone: specify backbone name. choosing from detr_resnet50, detr_resnet50_dc5, detr_resnet101 and detr_resnet101_dc5
    """
    assert post_process is False
    detr = torch.hub.load(
        "facebookresearch/detr:main",
        backbone,
        pretrained=pretrained,
        return_postprocessor=post_process,
    )

    ei = torch.rand(*sizes)
    return (
        trace(Detector(detr, post_process, False), ei, splat=not pretrained),
        ei,
    )
