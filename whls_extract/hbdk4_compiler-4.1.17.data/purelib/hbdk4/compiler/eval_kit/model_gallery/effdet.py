import torch
from hbdk4.compiler.eval_kit.model_gallery.utils import Detector
from hbdk4.compiler.eval_kit import trace


def retrieve_efficientdet(backbone, pretrained=False, post_process=False):
    """
    get efficientdet with coco detection head, with pretrained weights with post process graph
      backbone: specify backbone name. choosing from efficientdet_d0 ~ d7
    """
    from effdet import get_efficientdet_config, create_model_from_config

    config = get_efficientdet_config("tf_" + backbone)
    config.backbone_args.drop_path_rate = 0.0  # NOTE: drop path for traning.

    model = create_model_from_config(
        config, bench_task="predict" if post_process else "", pretrained=pretrained
    )

    isize = config["image_size"]
    ei = torch.rand(1, *isize, 3)
    return trace(Detector(model, post_process), ei, splat=not pretrained), ei
