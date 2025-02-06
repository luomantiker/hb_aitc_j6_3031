import logging

from hat.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

__all__ = ["FreezeBNStatistics"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FreezeBNStatistics(CallbackMixin):  # noqa: D400
    """
    Freeze BatchNorm module every epoch while training.

    Different from `FreezeModule` with `step_or_epoch` being [0]
    and `only_batchnorm` being True when resume training.
    """

    def on_epoch_begin(self, model, **kwargs):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        model.apply(fix_bn)
