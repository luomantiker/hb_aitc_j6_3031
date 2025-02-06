# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Any, Optional

import cv2
import numpy as np

from hat.utils.timer import AlarmTimerDecorator

__all__ = ["pil_loader"]

logger = logging.getLogger(__name__)


def pil_loader(data_path, mode="RGB", size=None, timeout=60):
    """
    Load image using PIL, open path as file to avoid ResourceWarning.

    # (https://github.com/python-pillow/Pillow/issues/835)

    """
    import timeout_decorator
    from PIL import Image

    @timeout_decorator.timeout(timeout)
    def _pil_loader(data_path, io_mode):
        fid = open(data_path, io_mode)
        with Image.open(fid) as img:
            if size is None:
                img = img.convert(mode)
            else:
                img.draft(mode, size)
                img.load()
        fid.close()
        return img

    try:
        img = _pil_loader(data_path, "rb")
        return img
    except (timeout_decorator.TimeoutError, FileNotFoundError) as e:
        if isinstance(e, timeout_decorator.TimeoutError):
            logger.info(f"read {data_path} timeout > {timeout}sec")
        elif isinstance(e, FileNotFoundError):
            logger.info(f"{data_path} FileNotFoundError")
        raise FileNotFoundError


@AlarmTimerDecorator(20)
def decode_img(
    s: bytes,
    iscolor: int = -1,
    img_type=np.uint8,
    from_buffer=True,
    backend="cv2",
) -> np.ndarray:
    """decode_img.

    Decode a image from a buffer or nd.ndarray.

    Args:
        s : buffer or np.ndarray
        iscolor : is color image
        img_type : image type, default is np.uint8
        from_buffer : weather from buffer
        backend: backend to decode image, default is cv2

    Returns:
        np.ndarray : image
    """
    if backend == "cv2":
        if from_buffer:
            img = np.frombuffer(s, dtype=img_type)
        else:
            img = s
        img = cv2.imdecode(img, iscolor)
    else:
        raise NotImplementedError(f"backend {backend} is not supported!")
    return img


@AlarmTimerDecorator(20)
def encode_decode_img(
    data,
    ext: str = ".jpg",
    encode_params: Optional[Any] = None,
    is_coler: int = -1,
) -> np.ndarray:
    """
    Encode a image to a buffer and decode it to a image.

    Args:
        data : image data
        ext : image format
        encode_params : encode parameters
        is_coler : is color image
    """
    _, encimg = cv2.imencode(ext, data, encode_params)
    decimg = cv2.imdecode(encimg, is_coler)
    return decimg
