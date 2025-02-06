import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from torch.types import _dtype as DType  # noqa N812
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op

logger = logging.getLogger(__name__)


# torch.nn.functional.softmax has parameters '_stacklevel' and 'dtype', while
# torch.nn.Softmax does not have these two parameters, drop here.
@replace_torch_op("softmax", is_nn_op=True)
class Softmax(torch.nn.Softmax):
    def __init__(
        self,
        dim: Optional[int] = None,
        _stacklevel: int = 3,
        dtype: Optional[DType] = None,
    ):
        super(Softmax, self).__init__(dim)
        logger.warning("Parameters `_stacklevel` and `dtype` will be dropped.")
