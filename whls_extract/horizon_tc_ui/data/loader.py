# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import hashlib
import logging
import os
import warnings
from collections import defaultdict
from itertools import chain
from typing import Any, Iterable, List

import numpy as np
import skimage.io

from horizon_tc_ui.data.transformer import TransposeTransformer

warnings.filterwarnings("ignore", "(Possibly )?Corrupt EXIF data", UserWarning)


class DummyLoader(object):
    def perform(self):
        raise AssertionError("Shouldn't reach here.")


class Loader(object):
    def __init__(self, parant_loader=DummyLoader):
        self._parant = parant_loader

    def __iter__(self):
        return self

    def __next__(self):
        return self.perform()

    def perform(self):
        data = self._parant.perform()
        return self.process(data)

    def process(self, data):
        return data

    def sequential(self, loader_list=[]):  # noqa
        return SequentialLoader(self, loader_list)

    def concat(self, loader_list=[]):  # noqa
        return ConcatLoader(self, loader_list)

    def batch(self, batch_size, batch_loader=None):
        if batch_loader:
            return batch_loader(self, batch_size)
        return BatchLoader(self, batch_size)

    def transform(self, transformers):
        return TransformLoader(self, transformers)

    def size(self, size):
        return SizeLoader(self, size)

    def split(self, size):
        return SplitLoader(self, size).split()


class SequentialLoader(Loader):
    def __init__(self, loader, loader_list=[]):  # noqa
        self._loader_list = loader_list
        super(SequentialLoader, self).__init__(loader)

    def process(self, data):
        for loader in self._loader_list:
            data = loader.process(data)
        return data


class ConcatLoader(Loader):
    def __init__(self, loader, loader_list=[]):  # noqa
        self._loader_list = loader_list
        super(ConcatLoader, self).__init__(loader)

    def __iter__(self):
        return chain(*self._loader_list)


class FunctionLoader(Loader):
    def __init__(self, loader, iter_func, kwargs):
        self.iter_func = iter_func
        self.kwargs = kwargs
        super(FunctionLoader, self).__init__(loader)

    def perform(self):
        return self.iter_func(**self.kwargs)


class SplitLoader(Loader):
    def __init__(self, loader, size):
        self.size = size
        self.buff = dict()  # noqa
        super(SplitLoader, self).__init__(loader)

    def split(self):
        ret_loader = list()
        for i in range(self.size):
            kwargs = {"index": i}

            def func(index):
                return self.get(index)

            loader = FunctionLoader(self, func, kwargs)
            ret_loader.append(loader)
        return ret_loader

    def get(self, index):
        if index in self.buff:
            ret_data = self.buff[index]
            del self.buff[index]
            return ret_data
        data = super(SplitLoader, self).perform()
        if not isinstance(data, list):
            raise TypeError("data must be a list")

        if len(data) < self.size:
            raise ValueError("size of data {} must equal size {}".format(
                len(data), self.size))
        for i, d in zip(range(self.size), data):
            if i != index:
                self.buff[i] = d
        return data[index]


class TransformLoader(Loader):
    def __init__(self, loader, transformers):
        super(TransformLoader, self).__init__(loader)
        self._trans = transformers

    def process(self, data):
        for tran in self._trans:
            data = tran(data)
        return data


class BatchLoader(Loader):
    def __init__(self, loader, batch_size):
        self._batch_size = batch_size
        super(BatchLoader, self).__init__(loader)

    def perform(self):
        batch_map = defaultdict(list)
        for _ in range(self._batch_size):
            try:
                data = super(BatchLoader, self).perform()
                for i in range(len(data)):
                    batch_map['data%d' % (i)].append(data[i])
            except StopIteration:
                if len(batch_map) > 0:
                    break
                else:
                    raise StopIteration

        data = list(batch_map.values())
        for i in range(len(data)):
            data[i] = np.array(data[i])
        if len(data) == 1:
            return data[0]
        return data


class SizeLoader(Loader):
    def __init__(self, loader, size):
        self._size = size
        self._total = 0
        super(SizeLoader, self).__init__(loader)

    def perform(self):
        if self._total > self._size:
            raise StopIteration
        data = super(SizeLoader, self).perform()
        self._total += 1
        return data


class DirDataLoader(Loader):
    def __init__(self, image_path):
        super(DirDataLoader, self).__init__()
        file_list = self.build_image_list(image_path)

        def _gen():
            for f in file_list:
                yield f

        self.file_gen = _gen()

    def image_read_method(self, file):
        raise NotImplementedError

    @staticmethod
    def build_image_list(data_dir):
        if not os.listdir(data_dir):
            raise ValueError(
                f"Directory {data_dir} is empty, please check calibration pics"
            )
        image_name_list = []
        for image in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, image)
            if os.path.isdir(full_path):
                logging.warning("Skip folder %s in %s", image, data_dir)
                continue
            image_name_list.append(full_path)

        first_img = sorted(image_name_list)[0]
        logging.info("*******************************************")
        logging.info("First calibration picture name: %s", first_img)
        try:
            # use python compute md5 for the first image
            with open(first_img, 'rb') as f:
                first_img_md5 = hashlib.md5(f.read()).hexdigest()
            logging.info(f"First calibration picture md5: {first_img_md5}")
        except Exception:
            logging.info("Get md5 info failed.")
        logging.info("*******************************************")
        return image_name_list

    def perform(self):
        return self.image_read_method(next(self.file_gen))


class ImageDirLoader(DirDataLoader):
    def __init__(self, image_path, gray=False):
        super(ImageDirLoader, self).__init__(image_path)
        self.gray = gray

    def image_read_method(self, file):
        if not file.lower().endswith(".jpg") and not file.lower().endswith(
                ".jpeg") and not file.lower().endswith(
                    ".png") and not file.lower().endswith(
                        ".gif") and not file.lower().endswith(".bmp"):
            logging.warning(f"File {file} may not be a regular image file. "
                            "Please double check the file input.")
        image = skimage.img_as_float(skimage.io.imread(file)).astype(
            np.float32)
        # expect gray but receive full color image
        if image.ndim == 3 and self.gray:
            logging.warning(f"Input image {file} is not gray image, "
                            "will be converted to gray image")
        # expend gray scale image to three channels
        if image.ndim != 3:
            image = image[..., np.newaxis]
            image = np.concatenate([image, image, image], axis=-1)
        return [image]


class SingleInferImageLoader(Loader):
    """
    A basic image loader, it will imread one single image for model inference
    """
    def __init__(self, image_file, gray=False):
        super(SingleInferImageLoader, self).__init__()
        self.image_file = image_file
        self.gray = gray

    def image_read_method(self, file):
        try:
            image = skimage.img_as_float(skimage.io.imread(file)).astype(
                np.float32)
        except Exception:
            logging.error(f"failed to read file {file}")
            raise ValueError(f"Failed to open {file} with skimage")
        # expect gray but receive full color image
        if image.ndim == 3 and self.gray:
            logging.warning(f"Input image {file} is not gray image, "
                            "will be converted to gray image")

        # expend gray scale image to three channels
        if image.ndim != 3:
            image = image[..., np.newaxis]
            image = np.concatenate([image, image, image], axis=-1)
        return [image]

    def perform(self):
        return self.image_read_method(self.image_file)


class RawImagesDirLoader(DirDataLoader):
    """
    a data loader for ImageNet valid dataset
    """
    def __init__(self,
                 image_path: str,
                 shape: Iterable,
                 dtype=np.uint8) -> None:
        super(RawImagesDirLoader, self).__init__(image_path)
        logging.debug('created RawImageDirLoader of shape:%s', shape)
        self.shape = shape
        self.dtype = dtype

    def image_read_method(self, file: str) -> List[Any]:
        logging.debug("Read raw file: %s", file)
        if not file.endswith('.npy'):
            data = np.fromfile(file, dtype=self.dtype) \
                .reshape(self.shape).astype(np.float32)
            return list(data)
        try:
            data = np.load(file=file).reshape(self.shape)
        except ValueError as error:
            logging.error(error)
            raise ValueError(f'File {file} load failed, '
                             'please double check your input.') from error
        return list(data)

    def _show_image(self, data, file) -> None:
        tmpdata = TransposeTransformer((1, 2, 0)).run_transform([data])[0]
        fname = file.split('/')[-1].split('.')[0]
        skimage.io.imsave(f"./{fname}_processed.jpg", tmpdata)
        print(f"./{fname}_processed.jpg generated")


class SingleRawLoader(Loader):
    """
    A basic image loader, it will imread one single image for model inference
    """
    def __init__(self,
                 image_path: str,
                 shape: Iterable,
                 dtype=np.uint8) -> None:
        super(SingleRawLoader, self).__init__()
        self.shape = shape
        self.dtype = dtype

        def raw_data_read(file: str) -> Any:
            logging.debug("Read raw file: %s", file)
            if not file.endswith('.npy'):
                data = np.fromfile(file, dtype=self.dtype).reshape(
                    self.shape).astype(np.float32)
                return data
            try:
                data = np.load(file=file).reshape(self.shape)
            except ValueError as error:
                logging.error(error)
                raise ValueError(f'File {file} load failed, '
                                 'please double check your input.') from error
            return data

        self.image_read_method = raw_data_read
        self.image_path = image_path

    def perform(self):
        return [self.image_read_method(self.image_path)]


class MultiLoaderWrapper(Loader):
    def __init__(self, *loader):
        super(MultiLoaderWrapper, self).__init__()
        self.loaders = loader

    def perform(self):
        return [next(loader)[0] for loader in self.loaders]
