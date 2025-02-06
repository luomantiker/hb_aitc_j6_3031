from ..channel_scale import ChannelScale2d as FloatChannelScale2d
from .qat_meta import QATModuleMeta


class ChannelScale2d(
    FloatChannelScale2d, metaclass=QATModuleMeta, input_num=1
):
    def __init__(self) -> None:
        raise RuntimeError(
            "ChannelScale2d is not supported in QAT, "
            "it must be fused into Conv2d"
        )
