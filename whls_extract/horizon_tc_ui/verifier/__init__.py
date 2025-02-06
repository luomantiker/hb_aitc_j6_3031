# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union

import numpy as np

from horizon_tc_ui.config.config_info import ConfigBase
from horizon_tc_ui.hb_runtime import (HB_HBIRRuntime, HB_HBMRuntime,
                                      HB_ONNXRuntime, HBRuntime)


@dataclass
class ModelInfo(ConfigBase):
    path: str  # model path
    name: str  # model name
    model_type: Literal["onnx", "bc", "hbm"]  # model type (onnx, bc, hbm)
    sess: Union[HB_ONNXRuntime, HB_HBIRRuntime, HB_HBMRuntime, HBRuntime]
    desc: dict = field(default_factory=dict)
    inputs: Dict[str, np.ndarray] = field(default_factory=dict)
    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    intermediate_node_info: Dict[str, str] = field(default_factory=dict)
    output_names: List[str] = field(default_factory=list)
    raw_input_names: List[str] = field(default_factory=list)
    raw_output_names: List[str] = field(default_factory=list)


@dataclass
class InputInfo(ConfigBase):
    path: str  # input data path
    name: str  # input data name
    data: np.ndarray = None  # input data
    batch: int = 1  # batch size
    preprocessed: bool = False


@dataclass
class BoardInfo(ConfigBase):
    ip: str = None
    username: str = "root"
    password: str = ""
    port: int = 22


@dataclass
class VerifierParams(ConfigBase):
    skip_sim: bool = False
    skip_arm: bool = False
    digits: int = 5
    mode: Literal["consistency", "cosine"] = None
    board_info: BoardInfo = field(default_factory=BoardInfo)
    models_info: List[ModelInfo] = field(default_factory=list)
    graph: Dict[str, str] = field(default_factory=dict)
    inputs_info: List[List[InputInfo]] = field(default_factory=list)
    consistency_info: Dict[str, Any] = field(default_factory=dict)
    cosine_info: Dict[str, float] = field(default_factory=dict)
    output_names: List[str] = field(default_factory=list)
