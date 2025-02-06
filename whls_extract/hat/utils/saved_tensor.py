import os
from decimal import localcontext
from distutils.version import LooseVersion

import torch

NEED_TORCH_VERSION = "1.10.2"

__all__ = [
    "support_saved_tensor",
    "clear_saved_tensors",
    "checkpoint_with_saved_tensor",
    "checkpoint_convbn_with_saved_tensor",
]


def support_saved_tensor():
    """Check whether support saved tensor."""
    match = LooseVersion(torch.__version__) >= LooseVersion(NEED_TORCH_VERSION)
    return match and bool(int(os.environ.get("HAT_USE_SAVEDTENSOR", "0")))


class SavedTensor:
    """Manage all saved tensors."""

    # all saved tensors
    tensors = {}
    # auto increment when create new object
    index = 0
    # auto increment when empty saved tensors
    version = 0

    def __init__(self, data):
        self._version = SavedTensor.version
        self.key = SavedTensor.index
        # update global
        assert self.key not in SavedTensor.tensors, "duplicated key!"
        SavedTensor.tensors[self.key] = data
        SavedTensor.index = SavedTensor.index + 1

    def __del__(self):
        if (
            self.key in SavedTensor.tensors
            and self._version == SavedTensor.version
        ):
            SavedTensor.tensors.pop(self.key)

    def data(self):
        return SavedTensor.tensors[self.key]

    @classmethod
    def empty(cls):
        cls.index = 0
        cls.tensors.clear()
        cls.version = cls.version + 1


def clear_saved_tensors():
    """Clear saved tensors."""
    SavedTensor.empty()
    # torch.cuda.empty_cache()


def checkpoint_with_saved_tensor(model):
    """Checkpoint a model with saved tensor.

    Args:
        model: The `nn.Module` instance.
    """
    for module in model.modules():
        if hasattr(module, "saved_tensor"):
            module.saved_tensor()


def _recompute(data, dtype, fn):
    if fn is None:
        return data

    if dtype != torch.float32:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
    else:
        autocast = localcontext()

    # reforward output tensor without grad
    with torch.no_grad():
        with autocast:
            res = fn(data)
    return res


def checkpoint_convbn_with_saved_tensor(conv, bn, act=None):
    """
    Return a function object that checkpoint conv+bn+act with saved tensor.

    Set conv as func1 and bn+act as func2.
    The func1 runs as usual, func2 run with context of saved tensor hooks.
    """

    def _fn1(x):
        return conv(x)

    def _fn2(x):
        x = bn(x)
        if act is not None:
            x = act(x)
        return x

    conv_grad_fn = "CudnnConvolutionBackward0"
    conv_grad_fn_2 = "ConvolutionBackward0"
    check_grad_next = False
    if conv.bias is not None:
        check_grad_next = True

    def do_conv_bn(data):
        def pack(t):
            if t.grad_fn is not None:
                if (
                    check_grad_next
                    and isinstance(t.grad_fn.next_functions[0], (list, tuple))
                    and t.grad_fn.next_functions[0][0] is not None
                    and t.grad_fn.next_functions[0][0].name() == conv_grad_fn
                ):

                    # save input, dtype and forward function
                    return SavedTensor((data, t.dtype, _fn1))
                elif (
                    t.grad_fn is not None
                    and not check_grad_next
                    and t.grad_fn.name() == conv_grad_fn
                ):
                    # save input, dtype and forward function
                    return SavedTensor((data, t.dtype, _fn1))
                elif (
                    t.grad_fn is not None
                    and t.grad_fn.name() == conv_grad_fn_2
                ):
                    # save input, dtype and forward function
                    return SavedTensor((data, t.dtype, _fn1))
                return SavedTensor((t, t.dtype, None))
            return t

        def unpack(t):
            if not isinstance(t, SavedTensor):
                return t
            # get input, dtype and forward function, then recompute output.
            return _recompute(*t.data())

        out = _fn1(data)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = _fn2(out)
        return out

    return do_conv_bn
