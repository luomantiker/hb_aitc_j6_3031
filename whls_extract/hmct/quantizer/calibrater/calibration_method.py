from typing import Dict, Optional


class CalibrationMethod:
    def __init__(self):
        self._methods = []
        self._percentiles = []
        self._max_num_bins = []
        self._num_bins = []
        self._per_channels = []
        self._per_channel_masks = []
        self._asymmetries = []
        self._block_sizes = []
        self._axis = []
        self.index = -1

    def set(
        self,
        method: str,
        percentile: float = 1.0,
        max_num_bin: int = 16384,
        num_bin: int = 1024,
        block_size: int = 0,
        axis: Dict[str, int] = {},
        per_channel: bool = False,
        per_channel_mask: bool = False,
        asymmetric: bool = False,
    ) -> "CalibrationMethod":
        self._methods.append(method)
        self._percentiles.append(percentile)
        self._max_num_bins.append(max_num_bin)
        self._num_bins.append(num_bin)
        self._block_sizes.append(block_size)
        self._axis.append(axis)
        self._per_channels.append(per_channel)
        self._per_channel_masks.append(per_channel_mask)
        self._asymmetries.append(asymmetric)
        return self

    def has(self, method: str) -> bool:
        return method in self._methods

    def subset(
        self,
        per_channel: Optional[bool] = None,
        asymmetric: Optional[bool] = None,
    ) -> "CalibrationMethod":
        _method = CalibrationMethod()
        for index in range(len(self._methods)):
            if per_channel is not None and per_channel != self._per_channels[index]:
                continue
            if asymmetric is not None and asymmetric != self._asymmetries[index]:
                continue
            _method.set(
                method=self._methods[index],
                percentile=self._percentiles[index],
                max_num_bin=self._max_num_bins[index],
                num_bin=self._num_bins[index],
                per_channel=self._per_channels[index],
                per_channel_mask=self._per_channel_masks[index],
                asymmetric=self._asymmetries[index],
            )
        return _method

    def __str__(self) -> str:
        cal_param = []
        if self._methods[self.index] == "kl":
            cal_param.extend(
                (
                    f"num_bin={self._num_bins[self.index]}",
                    f"max_num_bin={self._max_num_bins[self.index]}",
                )
            )
        if self._methods[self.index] == "max-percentile":
            cal_param.append(f"percentile={self._percentiles[self.index]}")
        if self._per_channels[self.index]:
            cal_param.append("per_channel")
        if self._asymmetries[self.index]:
            cal_param.append("asymmetric")
        if self._block_sizes[self.index]:
            cal_param.append(f"block_size={self._block_sizes[self.index]}")
        if cal_param:
            return "{}:{}".format(self._methods[self.index], ",".join(cal_param))
        return self._methods[self.index]

    def __len__(self) -> int:
        return len(self._methods)

    @property
    def type(self) -> str:
        return self._methods[self.index]

    @property
    def max_num_bin(self) -> int:
        return self._max_num_bins[self.index]

    @property
    def num_bin(self) -> int:
        return self._num_bins[self.index]

    @property
    def percentile(self) -> float:
        return self._percentiles[self.index]

    @property
    def per_channel(self) -> bool:
        """是否将per_channel方法写入到校准节点的cali_attrs属性."""
        if self._per_channel_masks[self.index]:
            return self._per_channels[self.index]
        return False

    @property
    def asymmetric(self) -> bool:
        return self._asymmetries[self.index]

    @asymmetric.setter
    def asymmetric(self, asymmetric: bool) -> None:
        self._asymmetries[self.index] = asymmetric

    @property
    def block_size(self) -> int:
        return self._block_sizes[self.index]

    @property
    def axis(self) -> Dict:
        return self._axis[self.index]

    def __iter__(self) -> "CalibrationMethod":
        return self

    def __next__(self) -> "CalibrationMethod":
        if self.index + 1 < len(self._methods):
            self.index += 1
            return self

        self.index = -1
        raise StopIteration

    def __add__(self, other: "CalibrationMethod") -> "CalibrationMethod":
        _method = CalibrationMethod()
        _method._methods = self._methods + other._methods
        _method._percentiles = self._percentiles + other._percentiles
        _method._max_num_bins = self._max_num_bins + other._max_num_bins
        _method._num_bins = self._num_bins + other._num_bins
        _method._per_channels = self._per_channels + other._per_channels
        _method._per_channel_masks = self._per_channel_masks + other._per_channel_masks
        _method._asymmetries = self._asymmetries + other._asymmetries
        _method._block_sizes = self._block_sizes + other._block_sizes
        _method._axis = self._axis + other._axis
        return _method
