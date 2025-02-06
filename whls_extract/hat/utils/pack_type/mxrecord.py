# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import numbers
import os
import queue as Queue
import struct
import sys
from collections import namedtuple
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from timeout_decorator import TimeoutError, timeout

from hat.utils.package_helper import require_packages
from hat.utils.timer import AlarmTimerContextManager
from .base import PackType

try:
    from horizon_plugin_pytorch.utils.mxrecordio import (
        MXIndexedRecordIO,
        MXRecordIO,
    )
except ImportError:
    MXIndexedRecordIO = object
    MXRecordIO = object

try:
    import imageio
except ImportError:
    imageio = None

try:
    import PIL
except ImportError:
    PIL = None

try:
    import turbojpeg

    turbojpeg_backend = turbojpeg.TurboJPEG()
except (ImportError, RuntimeError, AttributeError):
    turbojpeg = None
    turbojpeg_backend = None


__all__ = ["MXRecord", "MXRecordIO", "MXIndexedRecordIO"]

logger = logging.getLogger(__name__)


class MXRecordIO(MXRecordIO):  # noqa: D205,D400
    """Reads/writes `RecordIO` data format,
    supporting sequential read and write.

    Args:
        uri: Path to the record file.
        flag: 'w' for write or 'r' for read.
    """

    @require_packages("horizon_plugin_pytorch>=0.17.1")
    def __init__(self, uri: str, flag: str):
        super().__init__(uri, flag)

    @require_packages("horizon_plugin_pytorch>=1.0.0")
    def create_idx_file(self, idx_path: str = None, key_type: type = int):
        """Create idx file.

        Args:
            idx_path: The path to the index file.
            key_type: data type for keys.
        """

        assert self.record is not None, "Please open mxrecord before read."
        idx_path = self.uri + ".idx" if idx_path is None else idx_path
        idxs = {}
        keys = []
        cnt = 0
        pos = self.tell()
        while self.read():
            key = key_type(cnt)
            idxs[key] = pos
            keys.append(key)
            pos = self.tell()
            cnt += 1
            if cnt % 1000 == 0:
                sys.stdout.write("indexing %d recs.    \r" % (cnt))
                sys.stdout.flush()
        with open(idx_path, "w") as fidx:
            for key in keys:
                fidx.write("%s\t%d\n" % (str(key), idxs[key]))
        logger.info("Total rec number: %d           " % (len(keys)))
        self.reset()


class MXIndexedRecordIO(MXIndexedRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Args:
        idx_path : path to the index file.
        uri: path to the record file. Only supports seekable file types.
        flag: 'w' for write or 'r' for read.
        key_type: data type for keys.
    """

    @require_packages("horizon_plugin_pytorch>=0.17.1")
    def __init__(
        self, idx_path: str, uri: str, flag: str, key_type: type = int
    ):
        super().__init__(idx_path, uri, flag, key_type)


class MXRecord(PackType):
    """
    Abstract class of RecordIO, include all operators.

    While write_part_size > 1, multi recs will be got,
    names like: *.rec.part0; *.rec.part1.

    Args:
        uri (str): Path to record file.
        idx_path (str): Path to idx file.
        writable (bool): Writable flag for opening MXRecord.
        write_part_size (int): The size(MB) for each part
            if you want to split rec into parts.
            Non positive value means we do not do partition.
        key_type (type): Data type for record keys.
    """

    def __init__(
        self,
        uri: str,
        idx_path: Optional[str] = None,
        writable: bool = True,
        write_part_size: int = -1,
        key_type: type = int,
    ):
        self.uri = uri + ".rec" if os.path.isdir(uri) else uri
        self.idx_path = uri + ".idx" if idx_path is None else idx_path
        self.flag = "w" if writable else "r"
        self.write_part_size = write_part_size
        self.first_reopen_in_process = True
        self.part_id = 0
        self.key_type = key_type

        self.record = None
        self.open()

    def read(self, idx: int) -> bytes:
        """Read mxrecord file."""
        assert self.record is not None, "Please open mxrecord before read."
        try:
            return self._read_idx(idx)
        except TimeoutError as exception:
            logger.error(
                f"Time out when reading data with index of "
                f"{idx} from {self.uri}"
            )
            raise exception

    @timeout(seconds=600)
    def _read_idx(self, idx: int) -> bytes:
        return self.record.read_idx(idx)

    def write(self, idx: int, record: bytes):
        """Write record data into mxrecord file."""
        assert self.record is not None, "Please open mxrecord before write."

        # mxrecord should open and write in same process.
        if self.first_reopen_in_process:
            self.close()
            if self.write_part_size > 0:
                self.open(self.part_id)
            else:
                self.open()
            self.first_reopen_in_process = False

        if (
            self.write_part_size > 0
            and self.record.tell() > self.write_part_size * 1024 * 1024
        ):
            self.part_id += 1
            self.close()
            self.open(self.part_id)

        self.record.write_idx(idx, record)

    @timeout(seconds=600)
    def _open_record(self, idx_path, uri, flag, key_type):
        return MXIndexedRecordIO(
            idx_path=idx_path,
            uri=uri,
            flag=flag,
            key_type=key_type,
        )

    def open(self, part_id: int = -1):
        """Open mxrecord file."""
        if self.record is not None:
            return
        try:
            if part_id < 0:
                self.record = self._open_record(
                    idx_path=self.idx_path,
                    uri=self.uri,
                    flag=self.flag,
                    key_type=self.key_type,
                )
            else:
                self.record = self._open_record(
                    idx_path=self.idx_path + ".part{}".format(part_id),
                    uri=self.uri + ".part{}".format(part_id),
                    flag=self.flag,
                    key_type=self.key_type,
                )
        except TimeoutError as exception:
            logger.error(f"Time out when opening {self.idx_path}")
            raise exception

    def close(self):
        """Close mxrecord file."""
        if self.record:
            self.record.close()
            self.record = None

    def reset(self):
        """Reset the pointer to first item."""
        if self.record is not None:
            self.record.reset()
        else:
            self.open()

    def get_keys(self):
        """Get all keys."""
        try:
            keys = self.record.keys
            assert len(keys) > 0
            return range(len(keys))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            cnt = 0
            while self.record.read():
                keys.append(cnt)
                cnt += 1
            return keys

    def __len__(self):
        """Get the length."""
        return len(self.get_keys())

    def __getstate__(self):
        state = self.__dict__
        self.close()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.open()


IRHeader = namedtuple("HEADER", ["flag", "label", "id", "id2"])
_IR_FORMAT = "IfQQ"
_IR_SIZE = struct.calcsize(_IR_FORMAT)


def pack(header: IRHeader, s: str) -> str:
    """Pack a string into MXImageRecord.

    Args:
        header: Header of the image record,
                `header.label` can be a number or an array.
        s: Raw image string to be packed.

    Returns:
        s: The packed string.
    """
    header = IRHeader(*header)
    if isinstance(header.label, numbers.Number):
        header = header._replace(flag=0)
    else:
        label = np.asarray(header.label, dtype=np.float32)
        header = header._replace(flag=label.size, label=0)
        s = label.tostring() + s
    s = struct.pack(_IR_FORMAT, *header) + s
    return s


def pack_multi_record(header: IRHeader, s: str) -> str:  # noqa: D205,D400
    """Pack multi-records in one rec
    To use multi-record format recordio data in training, you should set
      recordio_format='multi_record' in your io iter config
    """
    header = IRHeader(
        flag=len(s), label=header.label, id=header.id, id2=header.id2
    )
    s = struct.pack(_IR_FORMAT, *header) + s
    return s


def unpack(s: str, multi_record: bool = False) -> Tuple[IRHeader, str]:
    """Unpack a MXImageRecord to string.

    Args:
        s: String buffer from ``MXRecordIO.read``.
        multi_record: Whether unpack image in format "multi_record".

    Returns:
        header : IRHeaderHeader of the image record.
        s : Unpacked string.
    """
    if multi_record:
        q = Queue.Queue()
        begin = 0
        while begin < len(s):
            header_t = IRHeader(*struct.unpack(_IR_FORMAT, s[begin:_IR_SIZE]))
            end = begin + _IR_SIZE + header_t.flag
            q.put((header_t, s[begin + _IR_SIZE : end]))
            begin = end
        return q

    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(
            label=np.frombuffer(s, np.float32, header.flag)
        )
        s = s[header.flag * 4 :]
    return header, s


def unpack_img(
    s: str,
    iscolor: int = -1,
    with_img_buf: bool = False,
    backend: str = "opencv",
) -> Tuple[IRHeader, np.ndarray, Optional[str]]:
    """Unpack a MXImageRecord to image.

    Args:
        s: String buffer from ``MXRecordIO.read``.
        iscolor: Image format option for ``cv2.imdecode``.
        with_img_buf: If return img buf.

    Returns:
        header: Header of the image record.
        img: Unpacked image.
        img_buf: Img buf
    """
    header, img_buf = unpack(s)

    if backend == "opencv":
        # order bgr
        img = np.frombuffer(img_buf, dtype=np.uint8)
        assert cv2 is not None
        with AlarmTimerContextManager(20, "decode image"):
            img = cv2.imdecode(img, iscolor)
    elif backend == "imageio":
        assert imageio is not None
        # order rgb
        img = imageio.v3.imread(BytesIO(img_buf))
    elif backend == "PIL":
        assert PIL is not None
        # order rgb
        img = np.asarray(PIL.Image.open(BytesIO(img_buf)))
    elif backend == "turbojpeg":
        assert turbojpeg is not None
        # order rgb
        img = turbojpeg_backend.decode(img_buf, pixel_format=0)
    else:
        raise ValueError(
            f"'{backend}' image processing backend unsport1. "
            "Please choose one of [opencv, imageio, PIL, turbojpeg] as backend!"  # noqa
        )
    if with_img_buf:
        return header, img, img_buf
    else:
        return header, img


def pack_img(
    header: IRHeader,
    img: np.ndarray,
    quality: int = 95,
    img_fmt: str = ".jpg",
    multi_record: bool = False,
) -> str:
    """Pack an image into ``MXImageRecord``.

    Args:
        header: Header of the image record,
                `header.label` can be a number or an array.
        img: Image to be packed.
        quality: Quality for JPEG encoding in range 1-100,
                 or compression for PNG encoding in range 1-9.
        img_fmt: Encoding of the image (.jpg for JPEG, .png for PNG).
        multi_record: whether pack image in format "multi_record".

    Returns:
        s: The packed string.
    """
    assert cv2 is not None
    jpg_formats = [".JPG", ".JPEG"]
    png_formats = [".PNG"]
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]

    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, "failed to encode image"
    if multi_record:
        return pack_multi_record(header, buf.tostring())
    else:
        return pack(header, buf.tostring())
