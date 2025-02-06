from ..einsum import EinSum as FloatEinSum
from .qat_meta import QATModuleMeta, init_activation_preprocesses


class EinSum(FloatEinSum, metaclass=QATModuleMeta, input_num=None):
    def __init__(self):
        self._input_num = len(self.equation.split("->")[0].split(","))
        init_activation_preprocesses(self)
