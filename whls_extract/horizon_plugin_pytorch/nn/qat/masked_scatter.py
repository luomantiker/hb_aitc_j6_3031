from ..masked_scatter import MaskedScatter as FloatMaskedScatter
from .qat_meta import QATModuleMeta


class MaskedScatter(FloatMaskedScatter, metaclass=QATModuleMeta, input_num=3):
    pass
