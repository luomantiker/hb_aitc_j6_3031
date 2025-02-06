# -*- coding: utf-8 -*-
# pylint: disable=W,C

"""
Horizon Robotics Development Kit.
All rights reserved.

Entry point functions for console programs
"""

import os
import sys


def main():
    abspath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../bin/hbrt4-run-model-nash")
    )
    if len(sys.argv) == 2 and sys.argv[1] == "--where":
        print(abspath, flush=True)
    else:
        os.execv(abspath, [abspath] + sys.argv[1:])
