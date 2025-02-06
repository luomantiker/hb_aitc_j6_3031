from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np

from .onnx_node import OnnxNode
from .pyquant import CalibrationAttrs, QuantizationAttrs

if TYPE_CHECKING:
    from .onnx_graph import OnnxGraph
    from .onnx_utils import NodeProto
    from .onnx_variable import OnnxVariable


class CalibrationNode(OnnxNode):
    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: "NodeProto",
        variable_mappings: Mapping[str, "OnnxVariable"],
    ) -> None: ...

    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        *,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        outputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        num_outputs: Optional[int] = None,
    ) -> None: ...

    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: Optional["NodeProto"] = None,
        variable_mappings: Optional[Mapping[str, "OnnxVariable"]] = None,
        op_type: Optional[str] = None,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        outputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        num_outputs: Optional[int] = None,
    ):
        if proto is None:
            assert op_type is not None, "op_type should be given if init_from_config."
            super().__init__(
                owning_graph=owning_graph,
                op_type=op_type,
                domain=domain,
                name=name,
                attributes=attributes,
                inputs=inputs,
                outputs=outputs,
                num_outputs=num_outputs,
            )
        else:
            assert (
                variable_mappings is not None
            ), "variable_mappings should be given if init_from_proto."
            super().__init__(
                owning_graph=owning_graph,
                proto=proto,
                variable_mappings=variable_mappings,
            )
        self._calibration_thresholds: Dict[str, List[float]] = {}
        self._sensitivities: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        self._support_asymq: bool = False
        self._parse_bits()
        self._parse_asymq()
        self._parse_info_strings()

    def _parse_bits(self) -> None:
        """支持低版本生成模型, 解析bits属性到qtype并且删除bits属性."""
        if "bits" in self._attributes:
            if "qtype" not in self._attributes:
                self._attributes["qtype"] = "int" + str(self._attributes["bits"])
            del self._attributes["bits"]

    def _parse_asymq(self) -> None:
        """支持低版本生成模型, 解析asymq属性."""
        _info_strings = self._attributes.get("info_strings", [])
        if "asymq" in _info_strings:
            if self._attributes.get("scales", None) is None:
                raise RuntimeError(
                    "Low version model, asymq calibration node missing ",
                    "scales, please generate the latest version model",
                )
            if self._attributes.get("thresholds", None) is None:
                raise RuntimeError(
                    "Low version model, asymq calibration node missing ",
                    "thresholds, please generate the latest version model",
                )
            if self._attributes.get("zero_point", None) is None:
                raise RuntimeError(
                    "Low version model, asymq calibration node missing ",
                    "zero points, please generate the latest version model",
                )
            # update thresholds
            _scales = self._attributes["scales"]
            _thresholds = self._attributes["thresholds"]
            _zero_point = self._attributes["zero_point"]
            for i in range(len(_thresholds)):
                _thresholds[i] -= _zero_point[i] * _scales[i]
            _thresholds = list(map(float, list(map(np.float32, _thresholds))))
            # update qtype
            _qtype = self._attributes.get("qtype", None)
            if _qtype and _qtype.startswith("int"):
                self._attributes["qtype"] = "u" + _qtype
            # del asymq and zero point attr.
            self._attributes["thresholds"] = _thresholds
            del self._attributes["zero_point"]
            _info_strings.remove("asymq")
            self._attributes["info_strings"] = _info_strings
        if "pre_op" in self._attributes:
            # TODO(zsq): insert pre_op before calibration node.
            del self._attributes["pre_op"]

    def _parse_info_strings(self) -> None:
        """支持低版本生成模型, 解析info_strings属性."""
        if "info_strings" in self._attributes:
            _info_strings = self._attributes["info_strings"]
            _quant_attr = QuantizationAttrs(self._attributes.get("quant_attrs", 0))
            for info in _info_strings:
                if info == "scale":
                    _quant_attr.set_scale  # noqa: B018
                if info == "shift":
                    _quant_attr.set_shift  # noqa: B018
                if info == "per-tensor":
                    _quant_attr.set_per_tensor  # noqa: B018
                if info == "per-channel":
                    _quant_attr.set_per_channel  # noqa: B018
                if info == "feature":
                    _quant_attr.set_feature  # noqa: B018
                if info == "weight":
                    _quant_attr.set_weight  # noqa: B018
            del self._attributes["info_strings"]
            self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def tensor_type(self) -> str:
        """返回校准节点tensor类型.

        校准节点类型包括"feature"和"weight".

        属性类型:
            读写属性, 允许赋值修改
        """
        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if _quant_attr.is_feature:
            return "feature"
        if _quant_attr.is_weight:
            return "weight"
        return ""

    @tensor_type.setter
    def tensor_type(self, tensor_type: str) -> None:
        if not isinstance(tensor_type, str):
            raise TypeError(
                f"type(tensor_type) should be str, but got {type(tensor_type)}.",
            )
        assert tensor_type in [
            "feature",
            "weight",
            "convt_weight",
        ], (
            f"tensor_type should be feature, weight or convt_weight,"
            f"but got {tensor_type}"
        )

        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if tensor_type == "feature":
            _quant_attr.set_feature  # noqa: B018
        if tensor_type == "weight":
            _quant_attr.set_weight  # noqa: B018
        if tensor_type == "convt_weight":
            _quant_attr.set_convt_weight  # noqa: B018
        self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def quantize_type(self) -> str:
        """返回校准节点量化类型.

        校准节点量化类型包括"scale"和"shift".

        属性类型:
            读写属性, 允许赋值修改
        """
        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if _quant_attr.is_scale:
            return "scale"
        if _quant_attr.is_shift:
            return "shift"
        return ""

    @quantize_type.setter
    def quantize_type(self, quantize_type: str) -> None:
        assert quantize_type in [
            "scale",
            "shift",
        ], f"quantize_type should be scale or shift, but got {quantize_type}"

        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if quantize_type == "scale":
            _quant_attr.set_scale  # noqa: B018
        if quantize_type == "shift":
            _quant_attr.set_shift  # noqa: B018
        self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def strategy(self) -> str:
        """返回校准节点量化策略.

        校准节点量化策略包括"static"和"dynamic".

        属性类型:
            读写属性, 允许赋值修改
        """
        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if _quant_attr.is_static:
            return "static"
        if _quant_attr.is_dynamic:
            return "dynamic"
        return "static"

    @strategy.setter
    def strategy(self, strategy: str) -> None:
        assert strategy in [
            "static",
            "dynamic",
        ], f"strategy should be static or dynamic, but got {strategy}"

        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if strategy == "static":
            _quant_attr.set_static  # noqa: B018
        if strategy == "dynamic":
            _quant_attr.set_dynamic  # noqa: B018
        self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def axis(self) -> int:
        """返回校准节点量化参数对应的axis.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._attributes.get("axis", 1)

    @axis.setter
    def axis(self, axis: int) -> None:
        if not isinstance(axis, int):
            raise TypeError(f"type(axis) should be int, but got {type(axis)}.")
        self._attributes["axis"] = axis

    @property
    def group(self) -> int:
        """返回校准节点输入采用的group.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._attributes.get("group", 1)

    @group.setter
    def group(self, group: int) -> None:
        if not isinstance(group, int):
            raise TypeError(f"type(group) should be int, but got {type(group)}.")
        self._attributes["group"] = group

    @property
    def constant(self) -> int:
        """返回校准节点constant属性取值.

        校准节点constant属性取值为0或者1.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._attributes.get("constant", 0)

    @constant.setter
    def constant(self, constant: int) -> None:
        if not isinstance(constant, int):
            raise TypeError(f"type(constant) should be int, but got {type(constant)}.")
        self._attributes["constant"] = constant

    @property
    def switch(self) -> str:
        """返回校准节点是否起作用.

        "ON"表示执行校准和伪量化, "OFF"表示不起作用.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._attributes.get("switch", "ON")

    @switch.setter
    def switch(self, switch: str) -> None:
        assert switch in ["ON", "OFF"], f"switch should be ON or OFF, but got {switch}"
        self._attributes["switch"] = switch

    @property
    def asymmetric(self) -> bool:
        """返回校准节点的非对称量化属性.

        属性类型:
            读写属性, 允许赋值修改
        """
        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        return _quant_attr.is_asymmetric

    @asymmetric.setter
    def asymmetric(self, asymmetric: bool) -> None:
        if not isinstance(asymmetric, bool):
            raise TypeError(
                f"type(asymmetric) should be bool, but got {type(asymmetric)}.",
            )

        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if asymmetric:
            _quant_attr.set_asymmetric  # noqa: B018
        else:
            _quant_attr.set_symmetric  # noqa: B018
        self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def granularity(self) -> str:
        """返回校准节点的量化粒度.

        校准节点的量化粒度包括"per-tensor", "per-channel"和"per-block".

        属性类型:
            读写属性, 允许赋值修改
        """
        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if _quant_attr.is_per_tensor:
            return "per-tensor"
        if _quant_attr.is_per_channel:
            return "per-channel"
        if _quant_attr.is_per_block:
            return "per-block"
        return ""

    @granularity.setter
    def granularity(self, granularity: str) -> None:
        assert granularity in ["per-tensor", "per-channel", "per-block"], (
            f"granularity should be per-tensor, per-channel or per-block, "
            f"but got {granularity}."
        )

        _quant_attr = QuantizationAttrs(self._attributes["quant_attrs"])
        if granularity == "per-tensor":
            _quant_attr.set_per_tensor  # noqa: B018
        if granularity == "per-channel":
            _quant_attr.set_per_channel  # noqa: B018
        if granularity == "per-block":
            _quant_attr.set_per_block  # noqa: B018
        self._attributes["quant_attrs"] = _quant_attr.get_value

    @property
    def qtype(self) -> str:
        """返回校准节点的数据类型.

        校准节点的数据类型包括"int8", "int16"和"float16".

        属性类型:
            读写属性, 允许赋值修改
        """
        if "qtype" in self._attributes:
            return self._attributes["qtype"]
        return "int8"

    @qtype.setter
    def qtype(self, qtype: str) -> None:
        if not isinstance(qtype, str):
            raise TypeError(f"type(qtype) should be str, but got {type(qtype)}.")
        self._attributes["qtype"] = qtype

    @property
    def scales(self) -> Union[Tuple[float, ...], None]:
        """返回校准节点的量化scales.

        属性类型:
            只读属性
        """
        _scales = self._attributes.get("scales", None)
        if self.constant == 2 or self.thresholds is None:
            return tuple(_scales) if _scales else None

        if self.asymmetric:
            assert len(self.thresholds) == 2, (
                f"only support thresholds: [min, max], "
                f"but got thresholds length {len(self.thresholds)}."
            )
            _fmin, _fmax = self.thresholds[0], self.thresholds[1]
            _min, _max = self.min_max_value
            _scales = [(_fmax - _fmin) / (_max - _min)]
        else:
            _max = self.min_max_value[1]
            _scales = np.array(self.thresholds) / _max
            if self.quantize_type == "shift":
                _scales = np.power(2, -np.round(-np.log2(_scales)))
            _scales = list(_scales)

        _scales = list(map(float, list(map(np.float32, _scales))))
        self._attributes["scales"] = _scales
        return tuple(_scales)

    @property
    def thresholds(self) -> Union[Tuple[float, ...], None]:
        """返回校准节点的量化thresholds.

        属性类型:
            读写属性, 允许赋值修改
        """
        _thresholds = self._attributes.get("thresholds", None)
        if self.constant == 2 and self.scales is not None:
            _max = self.min_max_value[1]
            _thresholds = [s * _max for s in self.scales]
            _thresholds = list(map(float, list(map(np.float32, _thresholds))))
            return tuple(_thresholds)

        return tuple(_thresholds) if _thresholds else None

    @thresholds.setter
    def thresholds(
        self,
        thresholds: Union[np.ndarray, List[float], Tuple[float, ...]],
    ) -> None:
        if isinstance(thresholds, np.ndarray):
            _thresholds = thresholds.tolist()
        elif isinstance(thresholds, (list, tuple)):
            _thresholds = list(thresholds)
        else:
            raise TypeError(
                f"type(thresholds) should be np.ndarray, list(float) "
                f"or tuple(float), but got {type(thresholds)}.",
            )
        _thresholds = list(map(float, list(map(np.float32, _thresholds))))
        self._attributes["thresholds"] = _thresholds

    @property
    def zero_point(self) -> Union[Tuple[int, ...], None]:
        """返回校准节点的量化zero_point.

        属性类型:
            只读属性
        """
        _zero_point = self._attributes.get("zero_point", None)
        if self.thresholds is None or not self.asymmetric:
            return tuple(_zero_point) if _zero_point else None

        if self.asymmetric:
            assert len(self.thresholds) == 2, (
                f"only support thresholds: [min, max], "
                f"but got thresholds length {len(self.thresholds)}."
            )
            assert self.scales is not None
            _fmax = self.thresholds[1]
            _max = self.min_max_value[1]
            _zero_point = [int(_max - np.round(_fmax / self.scales[0]))]

        self._attributes["zero_point"] = _zero_point
        return tuple(_zero_point)

    @property
    def block_sizes(self) -> Union[Tuple[float], None]:
        """返回校准节点的量化block_sizes.

        属性类型:
            读写属性, 允许赋值修改
        """
        _block_sizes = self._attributes.get("block_sizes", None)
        return tuple(_block_sizes) if _block_sizes else None

    @block_sizes.setter
    def block_sizes(
        self, block_sizes: Union[np.ndarray, List[int], Tuple[int]]
    ) -> None:
        if isinstance(block_sizes, np.ndarray):
            _block_sizes = block_sizes.tolist()
        elif isinstance(block_sizes, (list, tuple)):
            _block_sizes = list(block_sizes)
        else:
            raise TypeError(
                f"type(block_sizes) should be np.ndarray, list(int) "
                f"or tuple(int), but got {type(block_sizes)}."
            )
        _block_sizes = list(map(int, list(map(np.int32, _block_sizes))))
        self._attributes["block_sizes"] = _block_sizes

    @property
    def min_max_value(self) -> Tuple[float, float]:
        """返回校准节点量化的最小和最大值."""
        _min_value = float(-(1 << (8 - 1)))
        _max_value = float((1 << (8 - 1)) - 1)
        if self.qtype.startswith("int"):
            _bit = int(self.qtype[3:])
            _min_value = float(-(1 << (_bit - 1)))
            _max_value = float((1 << (_bit - 1)) - 1)
        elif self.qtype.startswith("uint"):
            _bit = int(self.qtype[4:])
            _min_value = 0.0
            _max_value = float((1 << _bit) - 1)
        elif self.qtype == "float8e4m3fn":
            _min_value = -448.0
            _max_value = 448.0
        elif self.qtype == "float8e5m2":
            _min_value = -57344.0
            _max_value = 57344.0
        elif self.qtype == "float8e3m4fn":
            _min_value = -30.0
            _max_value = 30.0
        elif self.qtype == "float8e2m5fn":
            _min_value = -7.75
            _max_value = 7.75
        elif self.qtype == "mxint8":
            _min_value = -1.984375
            _max_value = 1.984375

        return (_min_value, _max_value)

    @property
    def support_asymq(self) -> bool:
        return self._support_asymq

    @support_asymq.setter
    def support_asymq(self, support_asymq: bool) -> None:
        self._support_asymq = support_asymq

    @property
    def calibration_thresholds(self) -> Dict[str, List[float]]:
        """返回在特定校准方法下的校准阈值.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._calibration_thresholds

    @calibration_thresholds.setter
    def calibration_thresholds(
        self, calibration_thresholds: Mapping[str, Union[np.ndarray, Iterable[float]]]
    ) -> None:
        self._calibration_thresholds.clear()
        for k, v in calibration_thresholds.items():
            if isinstance(v, np.ndarray):
                self._calibration_thresholds[k] = v.tolist()
            elif isinstance(v, Iterable):
                self._calibration_thresholds[k] = list(v)
            else:
                raise TypeError(f"Unsupported calibration thresholds type: {type(v)}")

    @property
    def sensitivities(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """返回特定校准方法和数据类型下的校准节点量化敏感度字典.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._sensitivities

    def add_calibration_method(
        self,
        method: str,
        max_num_bin: int = 16384,
        num_bin: int = 1024,
        percentile: float = 1.0,
        per_channel: bool = False,
        axis: int = 1,
        block_size: int = 0,
    ) -> None:
        """基于校准方法设置cali_attrs,max_num_bins,num_bins以及percentile属性."""
        _cali_attrs = CalibrationAttrs(self._attributes.get("cali_attrs", []))
        _max_num_bins = self._attributes.get("max_num_bins", [])
        _num_bins = self._attributes.get("num_bins", [])
        _percentiles = self._attributes.get("percentiles", [])
        _block_sizes = self._attributes.get("block_sizes", [])
        if method == "max" and not per_channel and block_size == 0:
            _cali_attrs.set_max  # noqa: B018
        if method == "kl":
            _cali_attrs.set_kl  # noqa: B018
        if method == "max-percentile" and not per_channel:
            _cali_attrs.set_percentile  # noqa: B018
        if method == "min-max":
            _cali_attrs.set_min_max  # noqa: B018
        if method == "max" and per_channel:
            _cali_attrs.set_max_per_channel  # noqa: B018
        elif method == "max" and block_size > 0:
            _cali_attrs.set_max_per_block  # noqa: B018
        if method == "max-percentile" and per_channel:
            _cali_attrs.set_percentile_per_channel  # noqa: B018
        _max_num_bins.append(max_num_bin)
        _num_bins.append(num_bin)
        _percentiles.append(percentile)
        _block_sizes.append(block_size)
        self._attributes["cali_attrs"] = _cali_attrs.get_value
        self._attributes["max_num_bins"] = _max_num_bins
        self._attributes["num_bins"] = _num_bins
        self._attributes["percentiles"] = _percentiles
        self._attributes["axis"] = axis
        self._attributes["block_sizes"] = _block_sizes

    def set_calibration_method(self, method: str) -> None:
        """基于校准方法更新校准节点的thresholds,qtype量化属性."""
        _thresholds = self._calibration_thresholds.get(method)
        if _thresholds is not None:
            self.thresholds = tuple(_thresholds)
        _asymq = "asymmetric" in method
        if self.support_asymq and _asymq is False and self.qtype in ["uint8"]:
            self.qtype = self.qtype[1:]
        if self.support_asymq and _asymq is True and self.qtype in ["int8"]:
            self.qtype = "u" + self.qtype

    def update_node(self, calibration_node: "CalibrationNode") -> None:
        """拷贝模型时,更新CalibrationNode的特有属性到当前模型."""
        self._calibration_thresholds = calibration_node.calibration_thresholds
        self._support_asymq = calibration_node.support_asymq

    def _sync(self) -> None:
        # update calibration node attributes
        self._attributes["constant"] = self.constant
        self._attributes["switch"] = self.switch
        self._attributes["qtype"] = self.qtype
        if self.scales is not None:
            self._attributes["scales"] = list(self.scales)
        if self.thresholds is not None:
            self._attributes["thresholds"] = list(self.thresholds)
        if self.zero_point is not None:
            self._attributes["zero_point"] = list(self.zero_point)
        if self.block_sizes is not None:
            self._attributes["block_sizes"] = list(self.block_sizes)
            self._attributes["axis"] = self.axis
        super()._sync()
