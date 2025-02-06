from typing import List

import torch
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ChannelScale2d(nn.Module):
    """Do linear scale on output feature of Conv2d.

    The weight is trainable, and this operation must be fused into Conv2d
    before QAT.

    Args:
        num_features (int):  :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

        self.weight = Parameter(torch.empty(num_features))
        # following attrs is for mimic Bn
        # reduce buffer size to 1 to save memory
        self.eps = 1e-5
        self.register_buffer("bias", torch.zeros(1))
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = (
            self.weight if keep_vars else self.weight.detach()
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys: List[str],
        unexpected_keys,
        error_msgs,
    ):
        ret = super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        ignore_buffer_names = (
            "bias",
            "running_mean",
            "running_var",
        )
        for n in ignore_buffer_names:
            qualified_name = prefix + n
            if qualified_name in missing_keys:
                missing_keys.remove(qualified_name)

        return ret

    def forward(self, input: Tensor) -> Tensor:
        return input * self.weight.reshape(self.num_features, 1, 1)
