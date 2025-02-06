import json
from collections.abc import Mapping
from typing import Any

__all__ = ["is_jsonable", "strify_keys"]


def is_jsonable(x: Any):
    """To show if a variable is jsonable.

    Args:
        x : a variable.

    Returns:
        bool: True means jsonable, False the opposite.
    """
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def strify_keys(cfg: Mapping):
    """Convert keys of dict to strings if they are not for json dump.

    Args:
        cfg: dict for strify keys.

    Returns:
        dict of strified keys.
    """

    for key, value in list(cfg.items()):
        value = strify_keys(value) if isinstance(value, Mapping) else value
        if not isinstance(key, (str, int)):
            del cfg[key]
            cfg[str(key)] = value
    return cfg
