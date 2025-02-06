import copy

import torch
import torch.nn as nn
from easydict import EasyDict


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def combine_instances(init_instances, track_instances):
    new_instances = EasyDict()
    for key in list(init_instances.keys()):
        new_instances[key] = torch.concat(
            [init_instances[key], track_instances[key]], dim=0
        )
    return new_instances


def select_instances(ori_instances, ids):
    new_instances = EasyDict()
    for key in list(ori_instances.keys()):
        new_instances[key] = ori_instances[key][ids]
    return new_instances


def random_drop_tracks(track_instances, drop_probability: float):
    if drop_probability > 0 and len(track_instances.query_pos) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = select_instances(track_instances, keep_idxes)
    return track_instances


def padding_tracks(track_instances, fake_instances):
    pad_len = 0
    for key in list(track_instances.keys()):
        pad_len = len(track_instances[key])
        tmp_data = fake_instances[key].clone()
        tmp_data[:pad_len] = track_instances[key]
        track_instances[key] = tmp_data
    return track_instances, pad_len


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
