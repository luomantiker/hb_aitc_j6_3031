from torch import nn

from .qat_meta import QATModuleMeta


class Embedding(nn.Embedding, metaclass=QATModuleMeta):
    pass
