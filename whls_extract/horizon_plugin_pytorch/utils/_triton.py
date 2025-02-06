from functools import partial

import triton
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function
from triton import JITFunction


class ExportedJITFunction(JITFunction):
    """Generate fake kernel info to support export hbir."""

    def __getitem__(self, grid):
        callable_kernel = super().__getitem__(grid)

        def triton_export_wrapper(*args, **kwargs):
            if has_torch_function(args):
                return handle_torch_function(
                    triton_export_wrapper, args, *args, **kwargs
                )

            if "is_horizon_plugin_pytorch_export" in kwargs:
                tensor_indices = []
                for i, x in enumerate(args):
                    if isinstance(x, Tensor):
                        tensor_indices.append(i)

                hbir_kwargs = {
                    "signature": "fake",
                    "inputIndices": tensor_indices[1:],
                    "outputIndices": [tensor_indices[0]],
                }
                from hbdk4.compiler.ops import hbir

                return partial(hbir.triton_call, **hbir_kwargs)
            else:
                return callable_kernel(*args, **kwargs)

        return triton_export_wrapper


def jit(fn, *, version=None, do_not_specialize=None):
    def decorator(fn) -> ExportedJITFunction:
        assert callable(fn)
        return ExportedJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


def patch():
    triton.jit = jit
