#!/usr/bin/env python
from typing import List
import numpy as np
import sys
import os
import logging


class InputInfo:
    def __init__(
        self,
        *,
        input_idx: int,
        feature_name: str,
        filename: str,
        dims: List[int],
        input_semantic: str,
        element_type: str,
        image_shape: List[int] = None,
        image_roi: List[int] = None,
        image_stride: int = None,
        image_mode: str,
    ):
        self.input_idx = input_idx
        self.feature_name = feature_name
        self.filename = filename
        self.dims = dims
        self.input_semantic = input_semantic.lower()
        self.element_type = element_type
        self.image_shape = image_shape
        self.image_roi = image_roi
        self.image_stride = image_stride
        self.image_mode = image_mode
        self.data = None
        self.bpu_input_file = None

        if self.filename is not None:
            self.load_data_from_file()

        self.stride_speicified_manually = False
        if self.input_semantic in ("pyramid", "resizer"):
            if image_stride is None and self.filename is not None:
                self.image_stride = (self.image_shape[1] + 15) // 16 * 16
            elif image_stride is not None:
                self.image_stride = image_stride
                self.stride_speicified_manually = True

        if self.input_semantic in ("pyramid", "resizer") and self.filename is not None:
            self.check_image_size()

    def check_image_size(self):
        h = self.image_shape[0]
        w = self.image_shape[1]
        batch_size = self.dims[0]
        if self.image_shape[1] > self.image_stride:
            logging.critical(
                "The stride %d of the input image %s should be greater or equal to the image width %s"
                % (self.image_stride, self.filename, w)
            )
            sys.exit(1)

        img_size = np.size(self.data)
        if not self.stride_speicified_manually:
            if self.image_mode == "gray":
                expect_size = batch_size * h * w
            else:
                expect_size = batch_size * (h * w + ((h + 1) // 2) * ((w + 1) // 2) * 2)
            if img_size != expect_size:
                logging.critical(
                    "%s image %s with shape %dx%d expect file size of %d bytes,"
                    " but %d bytes"
                    % (self.image_mode, self.filename, h, w, expect_size, img_size)
                )
                sys.exit(1)
        else:
            if self.image_mode == "gray":
                expect_size = batch_size * h * self.image_stride
            else:
                expect_size = batch_size * (
                    h * self.image_stride
                    + ((h + 1) // 2) * ((self.image_stride + 1) // 2) * 2
                )
            if img_size != expect_size:
                logging.critical(
                    "%s image %s with shape %dx%d and stride %d expect file size of %d bytes, but %d bytes"
                    % (
                        self.image_mode,
                        self.filename,
                        h,
                        w,
                        self.image_stride,
                        expect_size,
                        img_size,
                    )
                )
                sys.exit(1)

    @property
    def ext(self):
        return self.filename.split(".")[-1]

    def is_y_or_nv12(self):
        return self.ext in ("y", "yuv")

    def load_data_from_file(self):
        data_dtype = np.int8
        if self.element_type == "uint8":
            data_dtype = np.uint8
        elif self.element_type == "int16":
            data_dtype = np.int16
        elif self.element_type == "int32":
            data_dtype = np.int32
        elif self.element_type == "float32":
            data_dtype = np.float32

        if self.ext == "txt":
            self.data = np.loadtxt(self.filename, dtype=data_dtype)
        elif self.input_semantic == "normal":
            self.data = np.frombuffer(
                open(self.filename, "rb").read(), dtype=data_dtype
            )
        elif self.is_y_or_nv12():
            self.data = np.frombuffer(
                open(self.filename, "rb").read(), dtype=data_dtype
            )
            assert self.image_shape, "YUV must specify image shape"
        else:  # jpg, png, jpeg
            import cv2

            image = cv2.imread(self.filename)
            is_y_only = False
            if self.image_shape is None:
                if self.dims[-1] == 1:
                    is_y_only = True
            self.data = convert_any_img_to_nv12(image)[0]
            if is_y_only:
                self.data = self.data[: image.shape[0] * image.shape[1]]
                self.image_shape[-1] = 1

    def generate_random_input_file(
        self, local_work_path, converted_module, input_data_list
    ):
        if self.filename:
            self.bpu_input_file = self.filename
        else:
            filename = os.path.join(local_work_path, "bpu_input_" + str(self.input_idx))
            if self.input_semantic == "normal":
                filename += ".bin"
            elif self.input_semantic == "pyramid" or self.input_semantic == "resizer":
                if self.image_mode == "gray":
                    filename += ".y"
                else:
                    filename += ".yuv"
            logging.info("======> Generate Input Data")
            data = None
            if self.input_semantic == "normal":
                if self.element_type == "float32":
                    data = np.random.uniform(-1, 1, size=self.dims).astype(np.float32)
                else:
                    data_dtype = np.int8
                    data_low = -128
                    data_high = 127
                    if self.element_type == "uint8":
                        data_dtype = np.uint8
                        data_low = 0
                        data_high = 255
                    elif self.element_type == "int16":
                        data_dtype = np.int16
                        data_low = -32768
                        data_high = 32767
                    elif self.element_type == "int32":
                        data_dtype = np.int32
                        data_low = np.iinfo(data_dtype).min
                        data_high = np.iinfo(data_dtype).max
                    data = np.random.randint(
                        low=data_low, high=data_high, size=self.dims, dtype=data_dtype
                    )

            elif self.input_semantic == "pyramid":
                line_num = self.dims[1]
                if self.image_mode == "nv12":
                    line_num += int((self.dims[1] + 1) // 2)
                line_num = line_num * self.dims[0]

                if self.image_shape is None:
                    self.image_shape = self.dims[1:]

                if self.image_stride is None:
                    self.image_stride = self.image_shape[1]

                data = np.random.randint(
                    low=0, high=255, size=[line_num, self.image_stride], dtype=np.uint8
                )

            elif self.input_semantic == "resizer":
                if self.image_shape is not None:
                    if self.image_stride is None:
                        self.image_stride = self.image_shape[1]
                else:
                    self.image_shape = [
                        1024,
                        1024,
                        1 if self.image_mode == "gray" else 3,
                    ]
                    if self.image_stride is not None:
                        logging.warning(
                            "image stride will be override for random input "
                            + self.feature_name
                            + " as random image width"
                        )
                    self.image_stride = 1024

                line_num = self.image_shape[0]
                if self.image_mode == "nv12":
                    line_num += int((self.image_shape[0] + 1) // 2)
                line_num = line_num * self.dims[0]

                data = np.random.randint(
                    low=0, high=255, size=[line_num, self.image_stride], dtype=np.uint8
                )

            self.bpu_input_file = filename
            self.data = data.flatten()
            if self.input_semantic in {"pyramid", "resizer"}:
                self.check_image_size()
            self.data.tofile(self.bpu_input_file)

        if converted_module is not None:
            if self.input_semantic == "pyramid":
                data = self.data.flatten()
                if self.image_mode == "nv12":
                    split_idx = self.dims[0] * self.dims[1] * self.image_stride
                    dim_y = (self.dims[0], self.dims[1], self.image_stride, 1)
                    dim_uv = (
                        self.dims[0],
                        int((self.dims[1] + 1) // 2),
                        self.image_stride // 2,
                        2,
                    )
                    data_y = data[0:split_idx].reshape(dim_y)
                    data_uv = data[split_idx:].reshape(dim_uv)
                    input_data_list.append(data_y)
                    input_data_list.append(data_uv)
                else:
                    dim_y = (self.dims[0], self.dims[1], self.image_stride, 1)
                    input_data_list.append(data.reshape(dim_y))
            elif self.input_semantic == "resizer":
                data = self.data.flatten()
                h0, w0, h1, w1 = self.image_roi
                data_roi = np.array([h0, w0, h1, w1]).astype(np.int32).reshape([1, 4])
                if self.image_mode == "nv12":
                    split_idx = self.dims[0] * self.image_shape[0] * self.image_stride
                    dim_y = (self.dims[0], self.image_shape[0], self.image_stride, 1)
                    dim_uv = (
                        self.dims[0],
                        int((self.image_shape[0] + 1) // 2),
                        self.image_stride // 2,
                        2,
                    )
                    data_y = data[0:split_idx].reshape(dim_y)
                    data_uv = data[split_idx:].reshape(dim_uv)
                    input_data_list.append(data_y)
                    input_data_list.append(data_uv)
                    input_data_list.append(data_roi)
                else:
                    dim_y = (self.dims[0], self.image_shape[0], self.image_stride, 1)
                    input_data_list.append(data.reshape(dim_y))
                    input_data_list.append(data_roi)
            else:
                data = self.data.reshape(self.dims)
                input_data_list.append(data)

        logging.info("======> Generate Input Data Done")


def convert_any_img_to_nv12(img_data):
    import cv2

    img_shape = img_data.shape
    # convert image format to YUV_I420
    nv12_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2YUV_I420)
    uv_start_idx = img_shape[0] * img_shape[1]
    u_or_v_size = ((img_shape[0] + 1) // 2) * ((img_shape[1] + 1) // 2)
    nv12_y_data = nv12_data.flatten()[0:uv_start_idx]
    nv12_u_data = nv12_data.flatten()[uv_start_idx : uv_start_idx + u_or_v_size]
    nv12_v_data = nv12_data.flatten()[
        uv_start_idx + u_or_v_size : uv_start_idx + 2 * u_or_v_size
    ]
    # truncate YUV data as int8
    nv12_y_data = nv12_y_data.astype(np.uint8)
    nv12_u_data = nv12_u_data.astype(np.uint8)
    nv12_v_data = nv12_v_data.astype(np.uint8)
    # reformat data as nv12
    nv12_res = nv12_y_data
    nv12_res = np.resize(
        nv12_res,
        [
            uv_start_idx + u_or_v_size * 2,
        ],
    )
    for i in range(u_or_v_size):
        nv12_res[uv_start_idx + 2 * i] = nv12_u_data[i]
        nv12_res[uv_start_idx + 2 * i + 1] = nv12_v_data[i]
    return nv12_res, img_shape
