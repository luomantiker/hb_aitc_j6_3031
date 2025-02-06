import atexit
import os
import shutil
import tempfile


class _TempDirectory:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    def relpath(self, name):
        return os.path.join(self.temp_dir, name)

    def mkdtemp(self):
        return tempfile.mkdtemp(dir=self.temp_dir)


TEMP_DIR = _TempDirectory()
