# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from typing import Any, Dict, List, Union

import numpy as np

from horizon_tc_ui.hb_hbirruntime import HB_HBIRRuntime
from horizon_tc_ui.hb_hbmruntime import HB_HBMRuntime
from horizon_tc_ui.hb_onnxruntime import HB_ONNXRuntime


class HBRuntime:
    def __init__(self, model) -> None:
        self.model_path = model
        self.model_type = None
        if self.model_path.endswith(".onnx"):
            self.model_type = 'onnx'
            self.sess = HB_ONNXRuntime(self.model_path)
        elif self.model_path.endswith(".bc"):
            self.model_type = 'bc'
            self.sess = HB_HBIRRuntime(self.model_path)
        elif self.model_path.endswith(".hbm"):
            self.model_type = 'hbm'
            self.sess = HB_HBMRuntime(self.model_path)
        else:
            raise ValueError(
                f"model {model} is invalid. Only models with .onnx .bc or .hbm"
                " suffixes are supported")

    @property
    def model(self):
        return self.sess.model

    @property
    def input_num(self) -> int:
        return self.sess.input_num

    @property
    def output_num(self) -> int:
        return self.sess.output_num

    @property
    def input_names(self) -> List[str]:
        return self.sess.input_names

    @property
    def output_names(self) -> List[str]:
        return self.sess.output_names

    @property
    def input_types(self) -> List[Union[np.dtype, int]]:
        return self.sess.input_types

    @property
    def output_types(self) -> List[Union[np.dtype, int]]:
        return self.sess.output_types

    @property
    def input_shapes(self) -> List[list]:
        return self.sess.input_shapes

    @property
    def output_shapes(self) -> List[list]:
        return self.sess.output_shapes

    @property
    def input_layouts(self) -> List[str]:
        return self.sess.layout

    @property
    def desc(self) -> dict:
        return self.sess.desc

    def run(self, output_names: list, input_feed: Dict[str, np.ndarray],
            **extra_args: dict) -> Any:
        return self.sess.run(output_names, input_feed, **extra_args)

    def run_direct(self, output_names: list, input_feed: Dict[str, np.ndarray],
                   **extra_args: dict) -> Any:
        return self.sess.run_direct(output_names, input_feed, **extra_args)
