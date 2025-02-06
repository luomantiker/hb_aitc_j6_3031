# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import numpy as np


# this function only support 3 dim
def raw_data_read(file_name, shape, data_type=np.uint8):
    logging.debug("Read raw file: {}".format(file_name))
    data = np.fromfile(file_name, dtype=data_type).reshape(shape)
    return data
