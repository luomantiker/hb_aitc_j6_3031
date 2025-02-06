from torch import nn

from horizon_plugin_pytorch.qtensor import QTensor
from .qat_meta import QATModuleMeta


class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d, metaclass=QATModuleMeta):
    def forward(self, input: QTensor):
        return super().forward(input)
