# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from __future__ import print_function  # 兼容print

import logging


class colour:
    """
    打印颜色
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def __print(header, msg, tee):
    print(header + msg + colour.ENDC)
    if tee:
        tee(msg + '\n')


def print_green(msg, tee=None):
    __print(colour.GREEN, msg, tee)


def print_blue(msg, tee=None):
    __print(colour.BLUE, msg, tee)


def print_red(msg, tee=None):
    __print(colour.RED, msg, tee)


def print_yellow(msg, tee=None):
    __print(colour.YELLOW, msg, tee)


class ColouredFormatter(logging.Formatter):
    COLOR_TO_LEVEL = {
        'DEBUG': colour.BLUE,
        'INFO': colour.GREEN,
        'WARNING': colour.YELLOW,
        'CRITICAL': colour.YELLOW,
        'ERROR': colour.RED,
    }

    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        log_msg = str(record.msg).strip()
        if log_msg.lower().startswith("warning:"):
            log_msg = log_msg[8:].strip()
        if log_msg.lower().startswith("error:"):
            log_msg = log_msg[6:].strip()
        record.msg = log_msg
        levelname = record.levelname
        log_msg = str(record.msg).strip()
        if log_msg.lower().startswith("warning:"):
            log_msg = log_msg[8:].strip()
            levelname = 'WARNING'
        if log_msg.lower().startswith("error:"):
            log_msg = log_msg[6:].strip()
            levelname = 'ERROR'
        record.msg = log_msg
        levelname_color = ColouredFormatter.COLOR_TO_LEVEL[
            levelname.upper()] + levelname + colour.ENDC
        record.levelname = levelname_color
        ret = logging.Formatter.format(self, record)
        record.levelname = levelname
        return ret
