from torch import nn

from horizon_plugin_pytorch.qtensor import QTensor
from .qat_meta import QATModuleMeta


class AdaptiveMaxPool1d(nn.AdaptiveMaxPool1d, metaclass=QATModuleMeta):
    def forward(self, input: QTensor):
        return super().forward(input)
