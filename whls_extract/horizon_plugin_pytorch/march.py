import inspect
from functools import wraps

import torch

__all__ = [
    "March",
    "set_march",
    "get_march",
    "with_march",
    "is_nash_series",
]


class March(object):
    """BPU platform."""

    BERNOULLI = "bernoulli"
    BERNOULLI2 = "bernoulli2"
    BAYES = "bayes"
    BAYES_E = "bayes-e"
    BAYES_A825_03 = "bayes-a825-03"
    NASH = "nash"
    NASH_E = "nash-e"
    NASH_M = "nash-m"
    NASH_P = "nash-p"
    META = "meta"


_march = None


# This function should never been modified unless authorized
# more information please refer to Makefile
def set_march(march):
    global _march
    _march = march


def get_march():
    return _march


def with_march(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        # if all positial arguments are specified, skip global march
        # Support use type:
        #
        #   @with_march
        #   @torch.jit.script / @torch.jit.script_if_tracing / @script_quantized_fn  # noqa
        #   def func
        #       ...
        #
        # or
        #
        #   @with_march
        #   def func
        #       ...
        #
        if isinstance(func, torch.jit.ScriptFunction):
            func_args_num = len(func.schema.arguments)
        else:
            arg_names, *others = inspect.getfullargspec(
                func.__original_fn if hasattr(func, "__original_fn") else func
            )
            func_args_num = len(arg_names)
        if func_args_num == len(args):
            return func(*args, **kwargs)

        # if march is specified, skip global march
        if "march" in kwargs.keys():
            return func(*args, **kwargs)

        return func(*args, **kwargs, march=get_march())

    return wrapped_func


def is_nash_series(march=None):
    if march is None:
        march = _march
    return march in (March.NASH, March.NASH_E, March.NASH_M, March.NASH_P)
