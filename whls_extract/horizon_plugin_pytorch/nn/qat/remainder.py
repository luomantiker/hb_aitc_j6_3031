from ..remainder import Remainder as FloatRemainder
from .qat_meta import QATModuleMeta


class Remainder(FloatRemainder, metaclass=QATModuleMeta, input_num=2):
    pass
