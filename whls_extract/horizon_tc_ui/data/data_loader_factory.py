# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from horizon_tc_ui.data.dataloader import DataLoader
from horizon_tc_ui.data.dataset_loader import (COCOValidLoader, ImageNetLoader,
                                               SingleImageLoader,
                                               VOCValidLoader)
from horizon_tc_ui.data.loader import (ImageDirLoader, MultiLoaderWrapper,
                                       RawImagesDirLoader,
                                       SingleInferImageLoader, SingleRawLoader)


def _get_loader(loader, transformers, batch_size=None):
    loader = loader.transform(transformers)
    if batch_size:
        loader = loader.batch(batch_size)
    return loader


def get_image_dir_loader(transformers,
                         image_path,
                         batch_size=None,
                         gray=False):
    return _get_loader(ImageDirLoader(image_path, gray=gray), transformers,
                       batch_size)


def get_single_image_loader(transformers, image_file, gray=False):
    loader = SingleInferImageLoader(image_file, gray)
    return _get_loader(loader, transformers)


def get_raw_image_dir_loader(transformer,
                             image_path,
                             shape,
                             dtype=np.uint8,
                             batch_size=None):
    loader = RawImagesDirLoader(image_path, shape, dtype)
    return _get_loader(loader, transformer, batch_size)


def get_raw_single_loader(transformer, image_file, shape, dtype=np.uint8):
    loader = SingleRawLoader(image_file, shape, dtype)
    return _get_loader(loader, transformer)


def get_multi_loader(*loader):
    return MultiLoaderWrapper(*loader)


def ImageNetDataLoader(transformers,
                       image_path,
                       label_path=None,
                       imread_mode='skimage',
                       batch_size=None):
    loader = ImageNetLoader(image_path, label_path, imread_mode)
    return DataLoader(loader, transformers=transformers, batch_size=batch_size)


def ImageLoader(transformers, image_path, imread_mode='skimage'):
    loader = SingleImageLoader(image_path, imread_mode)
    loader = DataLoader(loader, transformers=transformers, batch_size=1)
    return next(loader)


def RawImageLoader(transformers, image_path, shape_c_h_w, im_type):
    loader = SingleRawLoader(image_path, shape_c_h_w)
    loader = DataLoader(loader, transformers=transformers, batch_size=1)
    return next(loader)


def DetectionLoader(transformers, image_path, imread_mode='opencv'):
    origin_image_loader = SingleImageLoader(image_path, imread_mode)
    origin_image_loader = DataLoader(origin_image_loader,
                                     transformers=[],
                                     batch_size=1)
    process_image_loader = SingleImageLoader(image_path, imread_mode)
    process_image_loader = DataLoader(process_image_loader,
                                      transformers=transformers,
                                      batch_size=1)
    return [next(origin_image_loader), next(process_image_loader)]


def COCOValidDataLoader(transformers,
                        imageset_path,
                        annotations_path=None,
                        batch_size=1,
                        imread_mode='opencv'):
    loader = COCOValidLoader(imageset_path, annotations_path, imread_mode)
    return DataLoader(loader, transformers=transformers, batch_size=batch_size)


def VOCValidDataLoader(transformers,
                       imageset_path,
                       annotations_path=None,
                       val_txt_path=None,
                       batch_size=1,
                       imread_mode='opencv'):
    loader = VOCValidLoader(imageset_path, annotations_path, val_txt_path,
                            imread_mode)
    return DataLoader(loader, transformers=transformers, batch_size=batch_size)
