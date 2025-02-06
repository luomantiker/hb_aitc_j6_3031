import logging
import os
from typing import Sequence, Set, Union


def all_existing_debug_methods() -> Set[str]:
    """Return all the supported model debug methods.

    HorizonNN support the following mode debug method configurations:
    1. "dump_all_models" is used to dump internal hybrid onnx model and
            .hbir models.
    2. "dump_calibration_data", is used to dump all calibration data which will
            be used during precision debug phase.
    3. "dump_all_layers_output", which will modify the model and register all
            CONV and MATMUL node outputs as model's outputs which always used
            to debug some consistency error of nodes.
    4. "check_model_output_consistency", is used to check the model output
            consistency between quanti.onnx and hybrid.onnx
    """
    return {
        "dump_all_models",
        "dump_calibration_data",
        "dump_all_layers_output",
        "check_model_output_consistency",
    }


class ModelDebugger:
    def __init__(self):
        self._debug_methods: Set[str] = set()

    def register_debug_methods(self, debug_methods: Union[bool, Sequence[str]]) -> None:
        if isinstance(debug_methods, bool):
            logging.warning(
                "Warning: The bool-type debug_mode will be deprecated "
                "in the future.",
            )
            # 集成老版本会显性传参debug_mode=False, 为了方便单独更新horizon_nn patch
            # 时向前兼容, 增加一个判断
            if debug_methods is False:
                self._debug_methods = set()
            else:
                self._debug_methods = {
                    "dump_all_models",
                    "dump_all_layers_output",
                    "check_model_output_consistency",
                }
        elif isinstance(debug_methods, Sequence):
            self._debug_methods = set(debug_methods)
        else:
            raise ValueError("Error: Debug method must be a list type.")
        if os.environ.get("HNN_MODEL_EXPORT"):
            self._debug_methods.add("dump_all_models")
        self.check_debug_methods_valid()

    def check_debug_methods_valid(self) -> None:
        invalid = [
            method
            for method in self._debug_methods
            if method not in all_existing_debug_methods()
        ]
        if len(invalid) > 0:
            raise ValueError(
                f"The debug method: {invalid} is not "
                f"supported by horizon_nn model debugger.",
            )
        logging.info(f"Loading horizon_nn debug methods: {self._debug_methods}.")

    def has_debug_method(self, method: str) -> bool:
        if method not in all_existing_debug_methods():
            raise ValueError(
                f"The debug method you use: {method} is not "
                f"supported by horizon_nn model debugger.",
            )
        return method in self._debug_methods

    def print_existing_debug_methods(self) -> None:
        logging.info(f"Existing debug methods: {self._debug_methods}.")
