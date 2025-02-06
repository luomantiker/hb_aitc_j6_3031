import torch
from torch.utils.checkpoint import CheckpointFunction


class CheckpointState:
    _current = None

    def __init__(self) -> None:
        pass

    def __enter__(self):
        self.previous = CheckpointState._current
        CheckpointState._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CheckpointState._current = self.previous

    @classmethod
    def supress_update(cls):
        return cls._current is not None


def call_func_with_update_supression(func, *args):
    if torch.is_grad_enabled():
        # in CheckpointFunction.backward
        with CheckpointState():
            return func(*args)
    else:
        # in CheckpointFunction.forward
        return func(*args)


def checkpoint(function, *args, **kwargs):
    r"""Checkpoint a model or part of the model.

    Same as `torch.utils.checkpoint.checkpoint`.
    Except that:
    1. We supress the redundant update of FakeQuantize and BatchNorm.
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    return CheckpointFunction.apply(
        call_func_with_update_supression, preserve, *((function,) + args)
    )
