# Copyright (c) 2022 by Contributors
# file: compatible_ops.py
# date: 2022-06-01
# author: Yushu Gao (yushu.gao@horizon.ai)
# brief: compatible ops
# =============================================================================
from torch.nn import functional as F  # noqa N812

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.memory_opt import MemoryOptManager, MemoryOptSwitch
from horizon_plugin_pytorch.nn import functional as horizon_functional

horizon_relu_switch = MemoryOptSwitch(
    "ReLUMaskedBackward", 0, [0, 1], levels=[0, 1]
)
MemoryOptManager.register_switch(horizon_relu_switch)


def relu(out, use_relu6=False):
    if not use_relu6 or hz.get_march() == hz.March.BERNOULLI:
        return (
            horizon_functional.relu(out)
            if horizon_relu_switch.value == 1
            else F.relu(out)
        )
    else:
        return (
            horizon_functional.relu6(out)
            if horizon_relu_switch.value == 1
            else F.relu6(out)
        )


def relu6(out):
    return relu(out, use_relu6=True)
