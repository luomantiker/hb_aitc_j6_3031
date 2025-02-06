# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os

__test_path__ = os.path.dirname(os.path.abspath(__file__))
__deps_path__ = os.path.dirname(os.path.abspath(__file__)) + "/../test_deps/"
__deps_path__ = os.path.realpath(__deps_path__)

__board_ip__ = "10.103.43.19"

print(__test_path__)
print(__deps_path__)
