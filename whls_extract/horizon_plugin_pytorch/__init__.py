try:
    from .version import __version__  # noqa: F401

except ImportError:
    pass

from . import extension, jit, nn, quantization  # noqa: F401
from .dtype import *  # noqa: F403,F401
from .functional import *  # noqa: F403,F401
from .march import *  # noqa: F403,F401
from .qtensor import *  # noqa: F403,F401
from .torch_patch import *  # noqa: F403,F401
from .utils.logger import set_logger

set_logger("horizon_plugin_pytorch", file_dir=".horizon_plugin_pytorch_logs")
