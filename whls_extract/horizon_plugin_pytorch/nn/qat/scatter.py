from ..scatter import Scatter as FloatScatter
from ..scatter import ScatterAdd as FloatScatterAdd
from ..scatter import ScatterReduce as FloatScatterReduce
from .qat_meta import QATModuleMeta


class Scatter(FloatScatter, metaclass=QATModuleMeta, input_num=4):
    pass


class ScatterAdd(FloatScatterAdd, metaclass=QATModuleMeta, input_num=4):
    pass


class ScatterReduce(FloatScatterReduce, metaclass=QATModuleMeta, input_num=5):
    pass
