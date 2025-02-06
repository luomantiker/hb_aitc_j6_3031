# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
# flake8: noqa

try:
    import hmct
except ImportError as e:
    raise EnvironmentError("Please install hmct or hmct-gpu first.") from e

import os

import onnx  # noqa

from horizon_tc_ui.helper import ModelProtoBase  # noqa
from horizon_tc_ui.parser.caffe_parser import CaffeProto  # noqa
from horizon_tc_ui.parser.onnx_parser import OnnxModel  # noqa

from .hb_hbirruntime import HB_HBIRRuntime  # noqa
from .hb_onnxruntime import HB_ONNXRuntime  # noqa
from .hb_runtime import HBRuntime  # noqa
from .version import __version__

tool_path, _ = os.path.split(__file__)
