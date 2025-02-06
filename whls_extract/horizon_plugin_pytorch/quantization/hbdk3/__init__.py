import logging
import sys

logger = logging.getLogger(__name__)


def _raise_hbdk_import_error(func_name):
    def wrapper(*args, **kwargs):
        check_hbdk()
        msg = (
            "{} requires hbdk, which is uable to find and import. "
            "Please make sure hbdk is installed in the environment.".format(
                func_name
            )
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return wrapper


if sys.version_info >= (3, 10):
    import collections

    collections.Iterator = collections.abc.Iterator
    collections.Iterable = collections.abc.Iterable
    collections.Mapping = collections.abc.Mapping
    collections.MutableSet = collections.abc.MutableSet
    collections.MutableMapping = collections.abc.MutableMapping

try:
    from hbdk.torch_script.tools import (
        check_model,
        compile_model,
        export_hbir,
        perf_model,
        visualize_model,
    )

    def check_hbdk():
        pass


except ImportError as e:
    _import_exception = e

    def check_hbdk():
        raise EnvironmentError(
            "hbdk is not properly installed, origin exception is",
            "\n",
            *_import_exception.args
        )

    check_model = _raise_hbdk_import_error("check_model")
    compile_model = _raise_hbdk_import_error("compile_model")
    export_hbir = _raise_hbdk_import_error("export_hbir")
    perf_model = _raise_hbdk_import_error("perf_model")
    visualize_model = _raise_hbdk_import_error("visualize_model")

__all__ = [
    "check_model",
    "compile_model",
    "export_hbir",
    "perf_model",
    "visualize_model",
]
