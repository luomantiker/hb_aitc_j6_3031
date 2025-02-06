import logging

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.fx.fx_helper import wrap

logger = logging.getLogger(__name__)


@wrap()
class PytreePacker(nn.Module):
    """
    Flatten and reconstruct python object struct.

    Please refer to the doc of `torch.utils._pytree` for more info.
    """

    def __init__(self):
        super(PytreePacker, self).__init__()
        self.tree_spec = None

    def forward(self, input):
        output, tree_spec = tree_flatten(input)
        self.tree_spec = tree_spec
        logger.debug(
            "PytreePacker {} packed with spec {}".format(
                id(self), str(self.tree_spec)
            )
        )
        return output

    def reconstruct(self, input):
        logger.debug(
            "PytreePacker {} reconstruct with spec {}".format(
                id(self), str(self.tree_spec)
            )
        )
        if self.tree_spec is None:
            raise RuntimeError(
                "forward must be called first to run reconstruct"
            )
        return tree_unflatten(input, self.tree_spec)
