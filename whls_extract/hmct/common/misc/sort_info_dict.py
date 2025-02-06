import math
from typing import Any, Dict, Mapping, Tuple


def sort_info_dict(
    info_dict: Mapping[str, Mapping[str, Any]], key: str, reverse: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Sort multiple dictionaries based on the specified key.

    For any dictionary, if it is missing the given key or the value corresponding to the
    given key is NaN, then the dictionary will be placed at the end regardless of the
    value of reverse.

    Args:
        info_dict: A dictionary contains multiple inner dictionaries to be sorted.
        key: The given key whose value is used for sort different dictionaries.
        reverse: reverse: If True, the inner dictionaries will be sorted in descending
            order, otherwise in ascending order. Default to False.

    Returns:
        The dictionary contains multiple inner dictionaries which have been sorted based
        on the given key.
    """

    def sort_key(info: Tuple[str, Mapping[str, Any]]) -> float:
        """Obtain the value of given key for inner dictionaries sort."""
        value = float(info[1].get(key, float("nan")))
        # Ensure the dictionary of nan value for the given key always appears at the end
        if math.isnan(value):
            value = float("-inf") if reverse else float("inf")
        return value

    return dict(
        sorted(
            info_dict.items(),
            key=sort_key,
            reverse=reverse,
        )
    )
