# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
import subprocess

from hbdk4.compiler import load

from horizon_tc_ui.hbir_handle import HBIRHandle
from horizon_tc_ui.hbm_handle import HBMHandle
from horizon_tc_ui.utils.tool_utils import get_ip


class Visualize:
    def __init__(self, model_path: str, save_path: str) -> None:
        self.model_path = model_path
        self.save_path = save_path
        self.check()

    def check(self) -> None:
        model_suffix = ('.onnx', '.bc', '.hbm')
        if not self.model_path.endswith(model_suffix):
            raise ValueError(f'Input model {self.model_path} '
                             'is not supported now')

        if not os.path.exists(self.model_path):
            raise ValueError(f'Input model {self.model_path} ' 'is not exist')

    def start_server(self) -> None:
        ip = get_ip()
        cmd = f"netron {self.save_path} --host {ip}"
        try:
            subprocess.run(cmd, shell=True, check=False)
        except KeyboardInterrupt:
            logging.info("Visualization webserver closed")

    def convert_model(self) -> None:
        if self.model_path.endswith('.bc'):
            model = load(self.model_path)
            handle = HBIRHandle(model=model)
            handle.visualize(save_path=self.save_path)
        elif self.model_path.endswith('.hbm'):
            handle = HBMHandle(hbm_path=self.model_path)
            handle.visualize(save_path=self.save_path)
        else:
            return None

    def visualize(self) -> None:
        self.convert_model()
        self.start_server()
