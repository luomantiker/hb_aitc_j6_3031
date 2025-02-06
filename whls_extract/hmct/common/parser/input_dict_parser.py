import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..data.color_convert import ColorConvert

if TYPE_CHECKING:
    import numpy as np


class InputDictParser:
    """Model input handler gathers user input_dict, parse user input info from it."""

    def __init__(
        self,
        march: str,
        input_dict: Optional[Dict[str, Dict[str, Any]]],
        input_sources: Optional[Dict[str, str]] = None,
    ) -> None:
        self.march = march
        self.input_dict = {} if input_dict is None else input_dict
        self.input_sources = {} if input_sources is None else input_sources

        self.input_names: List[str] = []
        self.input_shapes: Dict[str, List[int]] = {}
        self.input_batches: Dict[str, int] = {}
        self.input_layouts: Dict[str, Dict[str, str]] = {}
        self.mean: Dict[str, Union[np.ndarray, None]] = {}
        self.scale: Dict[str, Union[np.ndarray, None]] = {}
        self.preprocess_info_dict: Dict[str, Dict[str, Any]] = {}

        self.float_input_type: Dict[str, str] = {}
        self.float_input_format: Dict[str, str] = {}
        self.float_input_range: Dict[str, str] = {}
        self.fixed_input_type: Dict[str, str] = {}
        self.fixed_input_format: Dict[str, str] = {}
        self.fixed_input_range: Dict[str, str] = {}

        self._parse_input_dict()
        self.update_input_layout_base_on_source(self.input_sources)

    # layout and type parser should be refined together with integration
    def _parse_input_dict(self) -> None:
        self._parse_input_names()
        self._parse_input_shapes()
        self._parse_input_batch()
        self._parse_input_type()
        self._parse_layout()
        self._parse_mean_and_scale()
        self._parse_preprocess_info()

    def _parse_input_names(self) -> None:
        self.input_names = list(self.input_dict.keys())

    def _parse_input_shapes(self) -> None:
        for input_name, input_params in self.input_dict.items():
            if "input_shape" in input_params:
                self.input_shapes[input_name] = input_params["input_shape"]

    def _parse_input_batch(self) -> None:
        for input_name, input_params in self.input_dict.items():
            if "input_batch" in input_params:
                self.input_batches[input_name] = input_params["input_batch"]

    def _parse_input_type(self) -> None:
        for input_name, input_params in self.input_dict.items():
            color_convert = input_params.get("color_convert", ColorConvert.NULL)
            if color_convert != ColorConvert.NULL:
                float_input_type, fixed_input_type = ColorConvert.split_color_convert(
                    color_convert,
                )
            elif (
                "original_input_type" in input_params
                and "expected_input_type" in input_params
            ):
                float_input_type = input_params.get("original_input_type")
                fixed_input_type = input_params.get("expected_input_type")
                color_convert = ColorConvert.get_convert_type(
                    float_input_type,
                    fixed_input_type,
                )
            else:
                float_input_type = "RGB"
                fixed_input_type = "RGB"

            self.float_input_type[input_name] = float_input_type.upper()
            self.fixed_input_type[input_name] = fixed_input_type.upper()
            self._parse_color_format(input_name)
            self._parse_color_range(input_name)

    # TODO: 后续与集成拉通, 将YUV_BT601_VIDEO_RANGE, YUV_BT601_FULL_RANGE
    # 这两个统一成{input_type}_{input_range}这种形式
    def _parse_color_format(self, input_name) -> None:
        input_format_range = self.float_input_type[input_name].split("_")
        self.float_input_format[input_name] = input_format_range[0]
        if self.float_input_format[input_name] == "YUV444":
            self.float_input_format[input_name] = "YUV_BT601_FULL_RANGE"

        if self.fixed_input_type[input_name] in [
            "YUV_BT601_VIDEO_RANGE",
            "YUV_BT601_FULL_RANGE",
        ]:
            self.fixed_input_format[input_name] = self.fixed_input_type[input_name]
        else:
            self.fixed_input_format[input_name] = self.fixed_input_type[
                input_name
            ].split("_")[0]
            if self.fixed_input_format[input_name] == "YUV444":
                # YUV_BT601_Full_Range
                self.fixed_input_format[input_name] = "YUV_BT601_FULL_RANGE"

    def _parse_color_range(self, input_name) -> None:
        float_input_format_range = self.float_input_type[input_name].split("_")
        fixed_input_format_range = self.fixed_input_type[input_name].split("_")
        self.float_input_range[input_name] = (
            float_input_format_range[1] if len(float_input_format_range) == 2 else "255"
        )

        if self.fixed_input_type[input_name] in [
            "YUV_BT601_VIDEO_RANGE",
            "YUV_BT601_FULL_RANGE",
        ]:
            self.fixed_input_range[input_name] = "128"
        else:
            self.fixed_input_range[input_name] = (
                self.fixed_input_type[input_name].split("_")[1]
                if len(fixed_input_format_range) == 2
                else "255"
            )

    def _parse_layout(self) -> None:
        for input_name, input_params in self.input_dict.items():
            float_input_layout = input_params.get("original_input_layout", "NCHW")
            if "expected_input_layout" in input_params:
                fixed_input_layout = input_params.get("expected_input_layout")
            elif self.fixed_input_range[input_name] == "128":
                if self.march in ["bernoulli", "bernoulli2"]:
                    fixed_input_layout = "NHWC"
                elif self.march in ["bayes", "bayes-e"]:
                    fixed_input_layout = "NOTSET"
                else:
                    # nash 架构下, 不应该有该参数传入, 对该参数的解析不支持.
                    raise ValueError(f"Unsupported march: {self.march}")
            else:
                fixed_input_layout = float_input_layout

            self.input_layouts[input_name] = {
                "original_input_layout": float_input_layout,
                "expected_input_layout": fixed_input_layout,
            }

    def _parse_mean_and_scale(self) -> None:
        for input_name, input_params in self.input_dict.items():
            self.mean[input_name] = input_params.get("means", None)
            self.scale[input_name] = input_params.get("scales", None)

    def _parse_preprocess_info(self) -> None:
        for input_name in self.input_dict:
            if (
                self.float_input_type[input_name] != self.fixed_input_type[input_name]
                or self.mean[input_name] is not None
                or self.scale[input_name] is not None
            ):
                preprocess_info = {
                    "mean": self.mean[input_name],
                    "scale": self.scale[input_name],
                    "from_color": self.float_input_format[input_name],
                    "from_color_input_range": self.float_input_range[input_name],
                    "input_layout_train": self.input_layouts[input_name][
                        "original_input_layout"
                    ],
                    "to_color": self.fixed_input_format[input_name],
                    "to_color_input_range": self.fixed_input_range[input_name],
                }
                self.preprocess_info_dict[input_name] = preprocess_info

    def get_input_names(self) -> List[str]:
        """Get the input names.

        Returns:
            A list containing the input names.
        """
        return self.input_names

    def get_input_shapes(self) -> Dict[str, List[int]]:
        """Get the input shapes.

        Returns:
            A dictionary mapping input names to their
            corresponding shape represented as a list of integers.
        """
        return self.input_shapes

    def get_input_batches(self) -> Dict[str, int]:
        """Get the input batches.

        Returns:
            A dictionary mapping input names to their
            corresponding batch size represented as an integer.
        """
        return self.input_batches

    def get_input_layouts(self) -> Dict[str, Dict[str, str]]:
        """Get the input layouts.

        Returns:
            A dictionary mapping input names to their corresponding
            original and expected layout.
        """
        return self.input_layouts

    def get_preprocess_info(self) -> Dict[str, Dict[str, Any]]:
        """Get the preprocess information.

        Returns:
            A dictionary mapping input names to their corresponding
            preprocess information.

            The preprocess information is represented
            as a dictionary with keys as information labels
            and values as the corresponding information values.
        """
        return self.preprocess_info_dict

    def update_input_layout_base_on_source(self, input_sources: Dict[str, str]) -> None:
        """Update the expected input layout config based on input source.

        For input whose input source set 'pyramid' or 'resizer', the expected
        input layout is determined by march, user set is forbidden.

        Args:
            input_sources: dict with keys as input name and values as input source.
        """
        self.input_sources = input_sources
        if input_sources:
            for input_name, input_source in input_sources.items():
                if input_name in self.input_layouts and input_source in (
                    "pyramid",
                    "resizer",
                ):
                    self.input_layouts[input_name]["expected_input_layout"] = "NHWC"
                    logging.info(
                        f"input {input_name} is from {input_source}. "
                        "Its layout is set to NHWC",
                    )
