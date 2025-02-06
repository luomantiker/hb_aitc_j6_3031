# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
from horizon_tc_ui.data.dataset import SingleImageDataset
from horizon_tc_ui.data.dataset import CifarDataset


def DataLoader(dataset, transformers, batch_size=1):
    dataset = dataset.transform(transformers)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def SingleImageDataLoader(transformers, image_path, imread_mode='skimage'):
    dataset = SingleImageDataset(image_path, imread_mode)
    dataset = dataset.transform(transformers)
    dataset = dataset.batch(1)
    return next(dataset)
