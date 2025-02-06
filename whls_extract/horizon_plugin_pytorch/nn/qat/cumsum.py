from ..cumsum import CumSum as FloatCumSum
from .qat_meta import QATModuleMeta


class CumSum(FloatCumSum, metaclass=QATModuleMeta, input_num=1):
    pass
