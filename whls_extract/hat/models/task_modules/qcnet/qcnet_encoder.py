from typing import Dict

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["QCNetOEEncoder"]


@OBJECT_REGISTRY.register
class QCNetOEEncoder(nn.Module):
    """
    QCNetOEEncoder module for encoding map and agent information.

    Args:
        map_encoder: Encoder module for encoding map information.
        agent_encoder: Encoder module for encoding agent information.
        stream_deploy: Flag indicating stream deployment mode.
    """

    def __init__(
        self,
        map_encoder: nn.Module,
        agent_encoder: nn.Module,
        stream_deploy: bool = False,
    ) -> None:
        super(QCNetOEEncoder, self).__init__()
        self.map_encoder = map_encoder
        self.agent_encoder = agent_encoder
        self.stream_deploy = stream_deploy

    def forward(self, data: dict) -> Dict[str, torch.Tensor]:

        quant_infer_cold_start = self.agent_encoder.quant_infer_cold_start
        deploy = self.agent_encoder.deploy
        if quant_infer_cold_start and not deploy:
            #  predict and cold_start
            if data["map_enc"] is not None:
                map_enc = data["map_enc"]
            else:
                map_enc = self.map_encoder(data)
        else:
            map_enc = self.map_encoder(data)
            if map_enc == {}:
                return {}
        agent_enc = self.agent_encoder(data, map_enc)

        return {**map_enc, **agent_enc}

    def set_qconfig(self):
        self.map_encoder.set_qconfig()
        self.agent_encoder.set_qconfig()
