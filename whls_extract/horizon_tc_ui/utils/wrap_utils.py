# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
from copy import deepcopy
from typing import Any, Callable, List


def remove_keys(dictionary: dict, key_list: list) -> dict:
    def remove_single_key(d, key_string) -> None:
        keys = key_string.split('.')
        current = d
        for key in keys[:-1]:
            if key in current:
                current = current[key]
            else:
                return None

        if keys[-1] in current:
            del current[keys[-1]]

    # skip unpicklable keys, such as hbdk mlir object
    result = deepcopy(
        {k: v
         for k, v in dictionary.items() if k not in key_list})
    for key_string in key_list:
        remove_single_key(result, key_string)
    return result


def try_except_wrapper(module_info: str,
                       ignore_keys: List[str] = []) -> Callable:
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            logging.debug('Start to execute %s.', module_info)
            if hasattr(func, '__call__') and hasattr(func.__call__,
                                                     '__self__'):
                logging.debug('%s input args %s', module_info, args[1:])
            else:
                logging.debug('%s input args %s', module_info, args)
            logging.debug('%s input kwargs %s', module_info,
                          remove_keys(kwargs, ignore_keys))
            try:
                result = func(*args, **kwargs)
                logging.debug('End to execute %s.', module_info)
            except Exception as e:
                if "ERROR-OCCUR-DURING" in str(e):
                    raise ValueError(str(e)) from e
                raise ValueError(f"*** ERROR-OCCUR-DURING {module_info} ***," +
                                 f" error message: {str(e)}") from e
            return result

        return wrapper

    return decorator
