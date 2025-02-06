import torch

from hbdk4.compiler.eval_kit.model_gallery.utils import Detector
from hbdk4.compiler.eval_kit import trace

from hbdk4.compiler.eval_kit.darknet.darknet import Darknet

import os


def retrieve_yolo(backbone, sizes=None, pretrained=False, post_process=False):
    """
    get yolo with coco detection head, with pretrained weights with post process graph
      backbone: specify backbone name. choosing from yolov3 and yolov4
    """

    cfg_dir = os.environ["TORCH_HOME"] + "/darknet/cfg"

    weight_dir = os.environ["TORCH_HOME"] + "/darknet/weight"
    yolo = Darknet(cfg_dir + "/" + backbone + ".cfg", post_process)
    if pretrained:
        yolo.load_weights(weight_dir + "/" + backbone + ".weights")

    net_info = yolo.blocks[0]
    if not sizes:
        isize = [
            1,
            int(net_info["height"]),
            int(net_info["width"]),
            int(net_info["channels"]),
        ]
    else:
        isize = sizes
    ei = torch.rand(*isize)
    return trace(Detector(yolo, post_process), ei, splat=not pretrained), ei
