from typing import Callable, Dict

from hmct.ir import OnnxModel

PARSE_FUNC = Callable[..., OnnxModel]

PARSE_FUNC_REGISTRY: Dict[str, PARSE_FUNC] = {}


def register_parse_func(model_type: str):
    def _register(parse_func: PARSE_FUNC) -> PARSE_FUNC:
        if model_type in PARSE_FUNC_REGISTRY:
            raise ValueError(f"Parse func for {model_type} already registered!")

        PARSE_FUNC_REGISTRY[model_type] = parse_func

        return parse_func

    return _register
