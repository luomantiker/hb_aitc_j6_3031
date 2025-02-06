from ..fmod import FMod as FloatFMod
from .qat_meta import QATModuleMeta


class FMod(FloatFMod, metaclass=QATModuleMeta, input_num=2):
    pass
