import atexit
import functools
import importlib.util

from hbdk4.compiler import _mlir_libs
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.utils.cext import (
    deinit_llvm,
    register_dialects,
    register_llvm_translations,
)


def callback(d):
    if d.severity == mlir.DiagnosticSeverity.ERROR:
        print(f"{d.location}: error: {d.message}")
        print(f"  notes: {list(map(str, d.notes))}")
        raise Exception("error detected")
    else:
        print(f"{d.location}: {d.message}")
        print(f"  notes: {list(map(str, d.notes))}")
    return True


default_context = mlir.Context()
register_dialects(default_context)
register_llvm_translations(default_context)
handler = default_context.attach_diagnostic_handler(callback)
assert handler.attached


def handle_diagnostic(f):
    # J6AITC-3621
    # Redirect original function f's signature and docstring to wrapper function.
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global handler
        handler.take_error()  #  clear error state
        result = f(*args, **kwargs)
        if handler.take_error():
            raise RuntimeError("error detected")
        return result

    return wrapper


default_location = mlir.Location.unknown(default_context)

default_context.__enter__()  # push a global context
default_location.__enter__()  # push a global unknown location

# register hbdnn kernels if package exists
if importlib.util.find_spec("hbdnn") is not None:
    import hbdnn

    _mlir_libs._hbdk.Dispatcher.get().load(hbdnn.__path__[0] + "/libhbdnn.so")


def exit_default():
    default_location.__exit__(None, None, None)
    default_context.__exit__(None, None, None)
    deinit_llvm()


atexit.register(exit_default)  # pop global stuffs
