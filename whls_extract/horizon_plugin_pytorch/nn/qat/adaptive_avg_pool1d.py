from torch import nn

from .qat_meta import QATModuleMeta


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, metaclass=QATModuleMeta):
    pass
