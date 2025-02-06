import torch
from hbdk4.compiler.eval_kit import trace

from hbdk4.compiler.eval_kit.model_gallery.utils import Backbone, Classifer


def retrieve_timm_backbone(backbone, ei, pretrained=False):
    """
    get backbone with pretrained weights
      backbone: specify backbone name. availables are
        resnet series: resnet50, resnet101, resnet152
        resnext series: resnext50_32x4d, resnext101_32x8d
        resnetrs series: resnetrs50, resnetrs101, resnetrs152, resnetrs200, resnetrs270, resnetrs350, resnetrs420
        nfnet series: nfnet_f0, nfnet_f1, nfnet_f2, nfnet_f3, nfnet_f4, nfnet_f5, nfnet_f6
        efficientnetV2 series: efficientnetv2_s, efficientnetv2_m, efficientnetv2_l, efficientnetv2_xl
        efficientnet-lite series: efficientnet_lite0, efficientnet_lite1, efficientnet_lite2, efficientnet_lite3, efficientnet_lite4
        swin transformer series: swin_base_patch4_window7_224, swin_base_patch4_window12_384, swin_large_patch4_window7_224, swin_large_patch4_window12_384
    """
    prefix = ""
    suffix = ""
    permute_return = True
    if "resnet" in backbone or "resnext" in backbone:
        from timm.models import resnet as models
    elif "efficientnet" in backbone:
        prefix = "tf_"
        if "v2" in backbone:
            suffix = "_in21ft1k"
        from timm.models import efficientnet as models
    elif "nfnet" in backbone:
        prefix = "dm_"
        from timm.models import nfnet as models
    elif "swin" in backbone:
        from timm.models import swin_transformer as models

        suffix = "_in22k"
        # swin transformer need modify input size in cfg
        models.default_cfgs[prefix + backbone + suffix]["input_size"] = tuple(
            [ei.shape[3], *ei.shape[1:3]]
        )
        permute_return = False

    net = Backbone(
        getattr(models, prefix + backbone + suffix)(pretrained=pretrained),
        permute_return=permute_return,
    )
    return trace(net, ei, splat=not pretrained)


def retrieve_timm_imagenet(backbone, batch=1, pretrained=False):
    """
    get backbone with imagenet1k classification head
      backbone: specify backbone name. availables are
        resnet series: resnet50, resnet101, resnet152
        resnext series: resnext50_32x4d, resnext101_32x8d
        resnetrs series: resnetrs50, resnetrs101, resnetrs152, resnetrs200, resnetrs270, resnetrs350, resnetrs420
        nfnet series: nfnet_f0, nfnet_f1, nfnet_f2, nfnet_f3, nfnet_f4, nfnet_f5, nfnet_f6
        efficientnetV2 series: efficientnetv2_s, efficientnetv2_m, efficientnetv2_l, efficientnetv2_xl
        efficientnet-lite series: efficientnet_lite0, efficientnet_lite1, efficientnet_lite2, efficientnet_lite3, efficientnet_lite4
        swin transformer series: swin_base_patch4_window7_224, swin_base_patch4_window12_384, swin_large_patch4_window7_224, swin_large_patch4_window12_384
      batch: speicify batch size
      pretrained: with real parameters or not
    """
    prefix = ""
    suffix = ""
    if "resnet" in backbone or "resnext" in backbone:
        from timm.models import resnet as models
    elif "efficientnet" in backbone:
        prefix = "tf_"
        if "v2" in backbone:
            suffix = "_in21ft1k"
        from timm.models import efficientnet as models
    elif "nfnet" in backbone:
        prefix = "dm_"
        from timm.models import nfnet as models
    elif "swin" in backbone:
        from timm.models import swin_transformer as models

        suffix = "_in22k"

    backbone_name = prefix + backbone + suffix

    cfg = getattr(models, "default_cfgs")[backbone_name]
    if "test_input_size" in cfg.keys():
        isize = cfg["test_input_size"]
    else:
        isize = cfg["input_size"]

    net = Classifer(getattr(models, backbone_name)(pretrained=pretrained))
    ei = torch.rand(batch, *isize[1:], isize[0])
    return trace(net, ei, splat=not pretrained), ei
