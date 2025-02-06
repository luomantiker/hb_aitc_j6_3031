# Copyright (c) Horizon Robotics. All rights reserved.
# This file is used to manage all global variables in HAT

from typing import Any

global global_dict
global_dict = {}


def set_value(key: str, value: Any):
    global_dict[key] = value


def get_value(key: str):
    try:
        return global_dict[key]
    except KeyError:
        return None
