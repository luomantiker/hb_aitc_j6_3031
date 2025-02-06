#  Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#  #
#  The material in this file is confidential and contains trade secrets
#  of Horizon Robotics Inc. This is proprietary information owned by
#  Horizon Robotics Inc. No part of this work may be disclosed,
#  reproduced, copied, transmitted, or used in any way for any purpose,
#  without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
from typing import List

from hbdk4.compiler import hbm_extract_desc, hbm_update_desc
from hbdk4.compiler._mlir_libs import __file__ as _mlir_libs_path
from hbdk4.compiler.hbm import Graph, Hbm

from horizon_tc_ui.utils.shell_wrapper import (subprocess_run,
                                               subprocess_run_full)
from horizon_tc_ui.utils.tool_utils import validate_json_str
from horizon_tc_ui.utils.wrap_utils import try_except_wrapper


class HBMHandle:
    """
    For hbm model file analysis, disassembly, and description maintenance
    """
    def __init__(self, hbm_path: str) -> None:
        self.hbm_path = os.path.abspath(hbm_path)
        if not self.hbm_path.endswith('.hbm'):
            raise ValueError(
                f"file invalid: {hbm_path}. It should be a '.hbm' file")
        if not os.path.exists(self.hbm_path):
            raise ValueError(f"{self.hbm_path} does not exist !!!")

        logging.info(f"hbm_path: {self.hbm_path}")
        self.hbm_name = os.path.basename(self.hbm_path)[:-4]
        self.workspace = os.path.dirname(hbm_path)
        self.desc_info = None
        self.disas_info = None
        self.hbdk_tool_base = os.path.dirname(_mlir_libs_path)
        self.hbdk_perf = os.path.join(self.hbdk_tool_base, "hbdk-perf")
        self.hbdk_disas = os.path.join(self.hbdk_tool_base, "hbrt4-disas")
        self.hbdk_hbm_desc = os.path.join(self.hbdk_tool_base, "hbdk-hbm-desc")
        self.model = Hbm(hbm_path)

    @property
    def name(self) -> str:
        return self.hbm_name

    @try_except_wrapper(module_info="hbdk.hbdk-perf")
    def perf(self, workspace: str = ".perf/") -> str:
        # TODO(ruxin.song): reusable
        workspace = os.path.abspath(workspace)
        logging.info(f"workspace: {workspace}")
        if not os.path.exists(workspace):
            os.makedirs(workspace)

        command = [self.hbdk_perf, self.hbm_path, '-o', workspace]
        returncode, _, stderr = subprocess_run_full(command)
        if returncode != 0:
            raise ValueError(str(stderr))
        logging.info(f"Successfully perf model {self.hbm_path}")
        # TODO(ruxin.song): check dir
        return workspace

    @try_except_wrapper(module_info="hbdk.hbm_update_desc")
    def update_desc(self, model_name: str, desc: str = "") -> None:
        desc_info = self.desc()
        if model_name not in desc_info["models"]:
            raise ValueError(f"Invalid model_name {model_name}")
        if not validate_json_str(desc):
            raise ValueError("Convert desc to json failed")

        desc_info["models"][model_name]["desc"] = desc
        hbm_update_desc(self.hbm_path, desc_info)

        return None

    @try_except_wrapper(module_info="hbdk.hbm_extract_desc")
    def desc(self) -> dict:
        hbm_desc_dict = hbm_extract_desc(self.hbm_path)
        return hbm_desc_dict

    @try_except_wrapper(module_info="hbdk.hbrt4-disas")
    def disas(self, workspace: str = ".disas/") -> dict:
        if self.disas_info:
            return self.disas_info

        # TODO(ruxin.song): reusable
        workspace = os.path.abspath(workspace)
        logging.info(f"workspace: {workspace}")
        if not os.path.exists(workspace):
            os.makedirs(workspace)

        filename = os.path.basename(self.hbm_path)[:-4]
        json_path = os.path.join(workspace, filename + ".json")
        logging.info(f"json_path: {json_path}")
        command = [self.hbdk_disas, self.hbm_path, "--json", "-o", json_path]
        subprocess_run(command, workspace=workspace)
        # TODO(ruxin.song): check file
        with open(json_path, 'r') as f:
            self.disas_info = json.load(f)

        # clear env
        os.remove(json_path)
        return self.disas_info

    @try_except_wrapper(module_info="hbdk.compiler.hbm")
    def get_graphs(self) -> List[Graph]:
        return self.model.graphs

    @try_except_wrapper(module_info="hbdk.hbm.visualize")
    def visualize(self, save_path) -> None:
        """Use hbrt-disas to generate a prototxt file

        Args:
            save_path (str): prototxt file save path
        """
        command = [self.hbdk_disas, self.hbm_path, "--netron", "-o", save_path]
        subprocess_run(command)
        logging.info(f"Successfully generated prototxt file to {save_path}")
