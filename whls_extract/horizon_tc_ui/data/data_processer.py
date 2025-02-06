# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
from typing import Generator, Iterable, Literal, Union, List

import cv2
import numpy as np
import skimage.io


class DataProcesser:
    def __init__(
            self,
            transformers: list,
            read_mode: Literal["opencv", "skimage", "numpy", "random"],
            dtype: Union[np.dtype, None] = None,
            shape: Iterable = [],
            gray: bool = False,
            dataset: str = "",
    ) -> None:
        """Data Processer

        Processing the data according to the input parameters

        Args:
            transformers (list): data transformers
            dataset (str): dataset path, single input or folder is ok
            read_mode (str): choose one from opencv/skimage/numpy
            dtype (str, optional): data type. Default to None.
            shape (Iterable, optional): data shape. Default to [].
        """
        self.transformers = transformers
        self.read_mode = read_mode
        self.dtype = dtype
        self.shape = shape
        self.gray = gray
        if os.path.isfile(dataset):
            self.dataset = [dataset]
        if os.path.isdir(dataset):
            self.dataset = [
                os.path.join(dataset, _data) for _data in os.listdir(dataset)
            ]
            if not self.dataset:
                raise ValueError(f"Directory {dataset} is empty")
        if read_mode in ["numpy", "random"]:
            if dtype is None or shape == []:
                raise ValueError(f"The read_mode is {read_mode}, "
                                 "dtype and shape must be specified")

    def read_data(self, data_path: str) -> np.ndarray:
        """Read data by different readmode

        Args:
            data_path (str): data path

        Returns:
            np.ndarray: numpy ndarray
        """
        if self.read_mode == "skimage":
            # skimage return rgb with float32
            if self.dtype and self.dtype != np.float32:
                raise ValueError("Invalid data type! "
                                 "When read mode is skimage, "
                                 "only valid data type is float32")
            data = skimage.img_as_float(skimage.io.imread(data_path)).astype(
                self.dtype)
        elif self.read_mode == "opencv":
            # opencv return bgr/gray with uint8
            if not self.dtype:
                self.dtype = np.uint8
            color_mode = (cv2.IMREAD_GRAYSCALE
                          if self.gray else cv2.IMREAD_COLOR)
            data = cv2.imread(data_path, color_mode).astype(self.dtype)
        elif self.read_mode == "numpy":
            # bin file need dtype to read and shape info to reshape
            data = (np.fromfile(data_path,
                                dtype=self.dtype).reshape(self.shape))
        elif self.read_mode == "random":
            data = np.random.random(self.shape).astype(self.dtype)
        else:
            raise ValueError(f"Invalid read mode {self.read_mode}")
        # expend gray scale image to three channels
        # if data.ndim != 3 and self.read_mode != "numpy":
        #     data = data[..., np.newaxis]
        #     data = np.concatenate([data, data, data], axis=-1)
        return data

    def process(self, data_path: str) -> np.ndarray:
        """Processing the data according to the input transformers

        Args:
            data_path (str): data path

        Returns:
            np.ndarray: processed data
        """
        processed_data = self.read_data(data_path)
        if not self.transformers:
            return processed_data
        for transformer in self.transformers:
            processed_data = transformer.run_transform(processed_data)
        return processed_data

    def perform(self) -> Generator[np.ndarray, None, None]:
        """Processing the input data in sequence

        Yields:
            Generator[np.ndarray, None, None]: processed data
        """
        for data_path in self.dataset:
            processed_data = self.process(data_path)
            yield processed_data

    def perform_all(self) -> List[np.ndarray]:
        """Processing all the input data at once and return a list

        Returns:
            List[np.ndarray]: all processed data in list
        """
        processed_data_list = []
        for processed_data in self.perform():
            processed_data_list.append(processed_data)
        return processed_data_list
