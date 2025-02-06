from torch import Tensor, nn

from horizon_plugin_pytorch.utils.checkpoint import CheckpointState


class BatchNorm2d(nn.BatchNorm2d):
    """Customized BN for using checkpoint.

    Supress the update of running statics during the second forward
    in CheckpointFunction.backward.
    """

    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype,
        )
        self.ori_momentum = momentum

    def forward(self, input: Tensor) -> Tensor:
        if CheckpointState.supress_update():
            self.momentum = 0.0
            ret = super().forward(input)
            self.momentum = self.ori_momentum
        else:
            ret = super().forward(input)

        return ret
