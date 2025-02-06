import logging
from typing import Sequence

import torch

from horizon_plugin_pytorch import march, qat_mode

logger = logging.getLogger(__name__)


def pow_quantization():
    return march.get_march() == march.March.BERNOULLI


def set_qparam(input_qparam, out_qparam, qparam_name):
    if isinstance(input_qparam, Sequence):
        out_qparam = torch.tensor(input_qparam)
    if isinstance(input_qparam, torch.Tensor):
        assert out_qparam.shape == input_qparam.shape, (
            "mismatched shape when set "
            + qparam_name
            + " {} vs {}".format(input_qparam.shape, out_qparam.shape)
            + ". Please set or check channel_len param in qconfig"
        )
        out_qparam.copy_(input_qparam)
    else:
        out_qparam.fill_(input_qparam)


class DeprecatedQATMode:
    def warn(self):
        logger.warning(
            "horizon_plugin_pytorch.quantization.misc.QATMode is deprecated, "
            "please use horizon_plugin_pytorch.qat_mode.QATMode",
            extra={"call_times_context": ("message")},
        )

    @property
    def WithBN(self):  # noqa:N802
        self.warn()
        return qat_mode.QATMode.WithBN

    @property
    def FuseBN(self):  # noqa:N802
        self.warn()
        return qat_mode.QATMode.FuseBN

    @property
    def WithBNReverseFold(self):  # noqa: # noqa: N812
        self.warn()
        return qat_mode.QATMode.WithBNReverseFold


QATMode = DeprecatedQATMode()


def get_qat_mode(*args, **kwargs):
    logger.warning(
        "horizon_plugin_pytorch.quantization.misc.get_qat_mode is deprecated, "
        "please use horizon_plugin_pytorch.qat_mode.get_qat_mode",
        extra={"call_times_context": ("message")},
    )

    return qat_mode.get_qat_mode(*args, **kwargs)


def set_qat_mode(*args, **kwargs):
    logger.warning(
        "horizon_plugin_pytorch.quantization.misc.set_qat_mode is deprecated, "
        "please use horizon_plugin_pytorch.qat_mode.set_qat_mode",
        extra={"call_times_context": ("message")},
    )
    return qat_mode.set_qat_mode(*args, **kwargs)
