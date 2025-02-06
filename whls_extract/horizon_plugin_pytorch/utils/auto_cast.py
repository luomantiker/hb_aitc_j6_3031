import torch

from .misc import pytree_convert


def to_float(input):
    return pytree_convert(input, torch.Tensor, lambda x: x.float())


def handle_autocast(supported_cpu_dtype=(), supported_gpu_dtype=()):
    """Auto convert the input to supported dtype when using amp.

    Used as a decorator, decorated operation must support float32 input.

    Args:
        supported_cpu_dtype: Supported input dtype on cpu besides float32.
            Defaults to ().
        supported_gpu_dtype: Supported input dtype on gpu besides float32.
            Defaults to ().
    """

    def wrapper(func):
        def wrapped_func(*args, **kwargs):
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                if dtype not in supported_gpu_dtype:
                    with torch.autocast("cuda", dtype=dtype, enabled=False):
                        return func(*to_float(args), **to_float(kwargs))
            elif torch.is_autocast_cpu_enabled():
                dtype = torch.get_autocast_cpu_dtype()
                if dtype not in supported_cpu_dtype:
                    with torch.autocast("cpu", dtype=dtype, enabled=False):
                        return func(*to_float(args), **to_float(kwargs))

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper
