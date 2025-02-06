# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
import subprocess
from typing import Tuple


def subprocess_run(cmd_list, workspace=None, timeout=None, retry=3) -> int:
    if not workspace:
        workspace = os.getcwd()
    cp = subprocess.run(cmd_list,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        cwd=workspace,
                        timeout=timeout,
                        check=False)

    if cp.returncode != 0:
        logging.error("subprocess_run failed!")
        logging.error("workspace: %s", workspace)
        logging.error("return code: %s", cp.returncode)
        logging.error("stdout: %s", cp.stdout)
        logging.error("stderr: %s", cp.stderr)
    if cp.returncode == 137 and retry != 0:
        cp.returncode = subprocess_run(cmd_list, workspace, timeout, retry - 1)
    return cp.returncode


def subprocess_run_full(cmd_list,
                        workspace=None,
                        timeout=None,
                        retry=3) -> Tuple[int, str, str]:
    if not workspace:
        workspace = os.getcwd()
    cp = subprocess.run(cmd_list,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        cwd=workspace,
                        timeout=timeout,
                        check=False)
    if cp.returncode != 0:
        logging.error("subprocess_run failed!")
        logging.error("workspace: %s", workspace)
        logging.error("return code: %s", cp.returncode)
        logging.error("stdout: %s", cp.stdout)
        logging.error("stderr: %s", cp.stderr)
    if cp.returncode == 137 and retry != 0:
        cp.returncode = subprocess_run(cmd_list, workspace, timeout, retry - 1)
    return cp.returncode, cp.stdout, cp.stderr
