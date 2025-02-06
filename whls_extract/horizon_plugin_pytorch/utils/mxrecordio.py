"""Read and write for the RecordIO data format."""
from contextlib import contextmanager
from multiprocessing import current_process

from horizon_plugin_pytorch._C import MXRecordReader, MXRecordWriter


@contextmanager
def reader(path, text=False, encoding=None):
    r_mode = "r"
    if not text:
        r_mode = "rb"
    r = open(path, r_mode, encoding=encoding)
    try:
        yield r
    finally:
        r.close()


class MXRecordIO(object):
    """Reads/writes `RecordIO` data format.

    supporting sequential read and write.

    Args:
        uri: Path to the record file.
        flag: 'w' for write or 'r' for read.
    """

    def __init__(self, uri: str, flag: str):
        self.uri = uri
        self.flag = flag
        self.pid = None
        self.is_open = False
        self.record = None
        self.open()

    def open(self):
        """Open the record file."""
        if self.flag == "w":
            self.record = MXRecordWriter(self.uri)
            self.writable = True
        elif self.flag == "r":
            self.record = MXRecordReader(self.uri)
            self.writable = False
        else:
            raise ValueError("Invalid flag %s" % self.flag)
        self.pid = current_process().pid
        self.is_open = True

    def __del__(self):
        self.close()

    def __getstate__(self):
        """Override pickling behavior."""
        # pickling pointer is not allowed
        is_open = self.is_open
        self.close()
        d = dict(self.__dict__)
        d["is_open"] = is_open
        return d

    def __setstate__(self, d):
        """Restore from pickled."""
        self.__dict__ = d
        is_open = d["is_open"]
        self.is_open = False
        if is_open:
            self.open()

    def _check_pid(self, allow_reset=False):
        """Check process id to ensure integrity, reset if in new process."""
        # pylint: disable=not-callable
        # It's bug from pylint(astroid).
        # See https://github.com/PyCQA/pylint/issues/1699.
        if not self.pid == current_process().pid:
            if allow_reset:
                self.reset()
            else:
                raise RuntimeError("Forbidden operation in multiple processes")

    def close(self):
        """Close the record file."""
        if not self.is_open:
            return
        del self.record
        self.record = None
        self.is_open = False
        self.pid = None

    def reset(self):
        """Reset the pointer to first item.

        If the record is opened with 'w',
        this function will truncate the file to empty.
        """
        self.close()
        self.open()

    def write(self, buf: str):
        """Insert a string buffer as a record.

        Args:
            buf: bytes buffer to write.
        """
        assert self.writable
        self._check_pid(allow_reset=False)
        self.record.WriteRecord(buf, len(buf))

    def read(self):
        """Read record as a string."""
        assert not self.writable
        # trying to implicitly read from multiple processes is forbidden,
        # there's no elegant way to handle unless lock is introduced
        self._check_pid(allow_reset=False)

        self.record.NextRecord()
        if len(self.record.read_buff):
            return self.record.read_buff
        else:
            return None

    def tell(self):
        """Get the current position of record head."""
        pos = self.record.Tell()
        return pos


class MXIndexedRecordIO(MXRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Args:
        idx_path : path to the index file.
        uri: path to the record file. Only supports seekable file types.
        flag: 'w' for write or 'r' for read.
        key_type: data type for keys.
    """

    def __init__(
        self, idx_path: str, uri: str, flag: str, key_type: type = int
    ):
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self.key_type = key_type
        self.fidx = None
        super().__init__(uri, flag)

    def open(self):
        super().open()
        self.idx = {}
        self.keys = []
        if self.writable:
            self.fidx = open(self.idx_path, self.flag)
        else:
            with reader(self.idx_path, True) as r:
                for line in iter(r.readline, ""):
                    line = line.strip().split("\t")
                    key = self.key_type(line[0])
                    self.idx[key] = int(line[1])
                    self.keys.append(key)

    def close(self):
        """Close the record file."""
        if not self.is_open:
            return
        super().close()
        if self.writable:
            self.fidx.close()

    def __getstate__(self):
        """Override pickling behavior."""
        d = super().__getstate__()
        d["fidx"] = None
        return d

    def seek(self, idx):
        """Set the current read pointer position.

        This function is internally called
        by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything.
        """
        assert not self.writable
        self._check_pid(allow_reset=True)
        pos = self.idx[idx]
        self.record.Seek(pos)

    def tell(self):
        """Get the current position of record head."""
        pos = self.record.Tell()
        return pos

    def read_idx(self, idx):
        """Read the record at given index."""
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Insert input record at given index."""
        assert self.writable
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write("%s\t%d\n" % (str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)
