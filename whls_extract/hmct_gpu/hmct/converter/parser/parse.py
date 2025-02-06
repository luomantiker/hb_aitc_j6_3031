from typing import TYPE_CHECKING

from .registry import PARSE_FUNC_REGISTRY

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def parse(model_type: str, *args, **kwargs) -> "OnnxModel":
    # call the corresponding parse func based on model type
    return PARSE_FUNC_REGISTRY[model_type](*args, **kwargs)
