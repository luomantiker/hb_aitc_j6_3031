import numpy as np
import torch
from packaging import version


# torch tensorboard is incompatible with high version numpy
# see https://github.com/pytorch/pytorch/issues/91516
def check_torch_numpy_version():
    if version.parse(torch.__version__.split("+")[0]) <= version.parse(
        "1.13.0"
    ) and version.parse(np.__version__) >= version.parse("1.24.0"):
        raise RuntimeError(
            "torch.utils.tensorboard.SummaryWriter.add_histogram "
            + f"in torch {torch.__version__} is not compatible with "
            + f"numpy {np.__version__}. Please down grade numpy "
            + "to < 1.24.0."
        )
