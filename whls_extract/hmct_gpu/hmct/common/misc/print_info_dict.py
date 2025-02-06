import logging
from typing import Any, Dict, Optional


def print_info_dict(
    info_dict: Dict[str, Dict[str, Any]],
    description: str = "",
    title: str = "",
    key_type: Optional[str] = None,
) -> None:
    """Format the information of a dictionary object into a string.

    Args:
        info_dict: A dictionary object containing information.
                   The dictionary format is as follows:
                   {info_key : {info_attr: info_value, ...}, ...}
        description: Description string for info_dict.
        title: Title string for info_dict.
        key_type: The type for the key of info_dict.
    """
    max_info_name_length = 50
    key_type = "Node" if key_type is None else key_type
    key_length: Dict[str, int] = {}
    key_length[key_type] = 10
    for info_key, info_val in info_dict.items():
        key_length[key_type] = max(len(info_key), key_length[key_type])
        for key, val in info_val.items():
            key_len = max(len(key), len(str(val)))
            key_length[key] = (
                max(key_len, key_length[key]) if key in key_length else key_len
            )
    key_length[key_type] = min(max_info_name_length, key_length[key_type])

    # Calculate the header length, and obtain key_formatter from key_length.
    key_formatter: Dict[str, str] = {}
    head_length = 0
    for key in key_length:
        key_len = key_length[key] + 2
        head_length += key_len
        key_formatter[key] = "{:<" + str(key_len) + "}"

    # Generate result table.
    head = ""
    for key in key_formatter:
        head += key_formatter[key].format(key)

    if (head_length - len(title)) % 2:
        left_char_num = (head_length - len(title)) // 2
        right_char_num = (head_length - len(title)) // 2 + 1
    else:
        left_char_num = right_char_num = (head_length - len(title)) // 2
    title = "=" * left_char_num + title + "=" * right_char_num
    s = [description, title, head, "-" * head_length]

    for info_key, info_val in info_dict.items():
        if len(info_key) > max_info_name_length:
            each_log = key_formatter[key_type].format(
                "..." + info_key[len(info_key) - max_info_name_length + 3 :],
            )
        else:
            each_log = key_formatter[key_type].format(info_key)

        for key, formatter in key_formatter.items():
            if key != key_type:
                if key in info_val:
                    each_log += formatter.format(info_val[key])
                else:
                    each_log += formatter.format("")

        s.append(each_log)

    logging.info("\n".join(s))
