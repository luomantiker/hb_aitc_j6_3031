import csv
import os

from tabulate import tabulate

from horizon_plugin_pytorch.utils.location_info import TorchLocationInfo


class CheckInfoItem:
    _default_msg = "passed"

    def __init__(self, location: TorchLocationInfo, msg=None) -> None:
        self._location = location
        if msg is None:
            self._passed = True
            self._msg = self._default_msg
        else:
            self._passed = False
            self._msg = msg

    def passed(self):
        return self._passed

    def get_contents(self):
        return self._location.get_contents() + (self._msg,)

    @classmethod
    def get_headers(cls):
        return TorchLocationInfo.get_headers() + ("Message",)


class DynamicInfo:
    def __init__(self, location: TorchLocationInfo, msg) -> None:
        self._location = location
        self._msg = msg

    def get_contents(self):
        return self._location.get_contents()[-1:] + (self._msg,)

    @classmethod
    def get_headers(cls):
        return TorchLocationInfo.get_headers()[-1:] + ("Message",)


class ModelChecker:
    _current = None

    def __init__(self, enabled=True, only_record_errors=True) -> None:
        self._check_results: list[CheckInfoItem] = []
        self._dynamic_infos: list[DynamicInfo] = []
        self._enabled = enabled
        self._only_record_errors = only_record_errors
        self._passed = True

    def __enter__(self, *args, **kwargs):
        self.old_instance = ModelChecker._current
        ModelChecker._current = self

    def __exit__(self, *args, **kwargs):
        ModelChecker._current = self.old_instance

    @classmethod
    def _get_instance(cls):
        if cls._current is None:
            raise RuntimeError("No ModelChecker instance is activated now")
        return cls._current

    @classmethod
    def enabled(cls):
        if cls._current is None:
            return False
        else:
            self = cls._get_instance()
            return self._enabled

    @classmethod
    def add_op_item(cls, item: CheckInfoItem):
        self = cls._get_instance()
        if not self._only_record_errors or not item.passed():
            self._check_results.append(item)
        self._passed = self._passed and item.passed()

    @classmethod
    def add_dynamic_info(cls, item: DynamicInfo):
        self = cls._get_instance()
        self._dynamic_infos.append(item)

    def passed(self):
        return self._passed

    def summary(self):
        headers = CheckInfoItem.get_headers()
        contents = (info.get_contents() for info in self._check_results)

        op_ret = tabulate(contents, headers)

        headers = DynamicInfo.get_headers()
        contents = (info.get_contents() for info in self._dynamic_infos)

        dynamic_ret = tabulate(contents, headers)

        return "\nOP Check Results:\n{}\n\nDynamic Info:\n{}".format(
            op_ret, dynamic_ret
        )

    def _write_tabulate_to_csv(self, output_path, contents, headers):
        with open(output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(contents)

    def save_to(self, dir_name):
        os.makedirs(dir_name, exist_ok=True)

        headers = CheckInfoItem.get_headers()
        contents = (info.get_contents() for info in self._check_results)

        self._write_tabulate_to_csv(
            os.path.join(dir_name, "op_check_results.csv"), contents, headers
        )

        headers = DynamicInfo.get_headers()
        contents = (info.get_contents() for info in self._dynamic_infos)

        self._write_tabulate_to_csv(
            os.path.join(dir_name, "dynamic_in_code.csv"), contents, headers
        )
