import logging

from horizon_plugin_pytorch.utils import mark_interface_state

logger = logging.getLogger(__name__)


def _raise_hbdk4_import_error(func_name):
    def wrapper(*args, **kwargs):
        check_hbdk4()
        msg = (
            "{} requires hbdk4, which is unable to find and import. "
            "Please make sure hbdk is installed in the environment.".format(
                func_name
            )
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return wrapper


try:
    from .export_hbir.export_hbir import (
        check,
        export,
        get_export_hbir_plugin_version,
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
        is_exporting,
    )
    from .export_hbir.horizon_registry import *  # noqa: F401, F403
    from .export_hbir.torch_registry import *  # noqa: F401, F403

    export = mark_interface_state("beta")(export)

    def check_hbdk4():
        pass


except Exception as e:
    _import_exception = e

    def check_hbdk4():
        raise EnvironmentError(
            "hbdk4 is not properly installed, origin exception is",
            "\n",
            *_import_exception.args
        )

    def is_exporting():
        return False

    check = _raise_hbdk4_import_error("check")
    export = _raise_hbdk4_import_error("export")
    get_hbir_input_flattener = _raise_hbdk4_import_error(
        "get_hbir_input_flattener"
    )
    get_hbir_output_unflattener = _raise_hbdk4_import_error(
        "get_hbir_output_unflattener"
    )
    get_export_hbir_plugin_version = _raise_hbdk4_import_error(
        "get_export_hbir_plugin_version"
    )


__all__ = [
    "check",
    "export",
    "is_exporting",
    "get_hbir_input_flattener",
    "get_hbir_output_unflattener",
    "get_export_hbir_plugin_version",
]
