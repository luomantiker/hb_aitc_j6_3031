from ..slice_scatter import SliceScatter as FloatSliceScatter
from .qat_meta import QATModuleMeta


class SliceScatter(FloatSliceScatter, metaclass=QATModuleMeta, input_num=2):
    pass
