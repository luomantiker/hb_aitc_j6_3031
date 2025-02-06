import functools
import importlib
import logging
import operator
import os
import re
from functools import lru_cache
from importlib.metadata import version as find_version
from typing import Callable, Tuple, Union

from packaging.version import Version

from hat.utils.logger import MSGColor, format_msg

logger = logging.getLogger(__name__)


def parse_package_spec(package_version: str) -> Tuple[str]:
    """Parse package name, operator and version information.

    Args:
        package_version: Required package info (e.g. `torch`, `torch>=1.10`).

    Examples:
       >>> parse_package_spec("torch")
       ("torch", None, None)

       >>> parse_package_spec("torch>=1.10.2")
       ("torch", ">=", "1.10.2")

       >>> parse_package_spec("torch<2.0")
       ("torch", "<", "2.0")

    Returns:
        (package_name, operator, package_version)
    """

    regex = "(==|!=|<=|>=|<|>)"
    match = re.compile(regex).search(package_version)

    if match:
        op = match.group(0)
        package_name, require_version = tuple(package_version.split(op))
        return package_name, op, require_version
    else:
        return package_version, None, None


def compare_version(
    package_name: str,
    op: Union[str, Callable],
    version: str,
    return_version: bool = False,
    compare_full_version: bool = True,
) -> bool:
    """Compare package version with some requirements.

    Args:
        package_name: Package name.
        version: The minimum dependency version of the package.
        return_version: Whether to return the actual installed package version.
        compare_full_version: Whether to compare full version. Default True.
            If False, will only compare public version.

        Note: The package version contains two parts: public and local.
            For example, If torch version is "2.1.0+cu118", then "2.1.0" is
            the `public` version and "cu118" is the `local` version.
            According to PEP440, `2.1.0+cu118` is newer than `2.1.0`, which
            means `Version("2.1.0+cu118") > Version("2.1.0")` will be True.

            See: https://www.python.org/dev/peps/pep-0440/
    """

    _operator_funcs = {
        ">=": operator.ge,
        ">": operator.gt,
        "<=": operator.le,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if isinstance(op, str):
        assert (
            op in _operator_funcs.keys()
        ), f"`operator` should be one of {_operator_funcs.keys()}, but get `{op}`"  # noqa E501
        op = _operator_funcs[op]
    elif isinstance(op, Callable):
        pass
    else:
        raise ValueError(f"`operator={op}` is illegal.")

    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return False

    if hasattr(pkg, "__version__"):
        pkg_version = Version(pkg.__version__)
    else:
        pkg_version = Version(find_version(package_name))

    cmp_version = Version(version)

    # compare public version
    if not compare_full_version:
        pkg_version = pkg_version.public
        cmp_version = cmp_version.public

    if return_version:
        return op(pkg_version, cmp_version), pkg_version
    else:
        return op(pkg_version, cmp_version)


@lru_cache()
def check_packages_available(
    *package_versions,
    raise_exception: bool = True,
    raise_msg: str = "",
    compare_full_version: bool = True,
):
    """Check whether required packages or modules is available.

    Args:
        package_versions: Package name(e.g. `torch`, `torch>=2.0.1`), or
            modules(e.g. `torch.nn`)
        raise_exception: Whether to raise error when import failed.
        raise_msg: Suggested actions after exception.
        compare_full_version: Whether to compare full version. Default True.
            If False, will only compare public version.

    Examples:
        # Assume torch.__version__ == 2.1.0+cu118
        >>> check_packages_available("torch>=2.0.1", "torchvision>=0.15.2")
        True

        >>> check_packages_available("torch>2.1.0")
        True

        >>> check_packages_available("torch>2.1.0", compare_full_version=False)
        False

        >>> check_packages_available("torch>=100000.0.0", "torchvision")
        False



    Returns:
        bool: True if all modules import successfully else False.
    """
    _imported_caches = {}
    for module in package_versions:
        pkg, op, required_version = parse_package_spec(module)

        try:
            importlib.import_module(pkg)
            require_available = True
            msg = f"Requirement {pkg} is available."
        except ImportError as e:
            msg = f"{e.__class__.__name__}: {e}. "
            require_available = False

        # only for package with version, e.g. `torch>=2.0.1`
        if require_available and op and required_version:
            version_static, pkg_version = compare_version(
                pkg,
                op,
                required_version,
                return_version=True,
                compare_full_version=compare_full_version,
            )

            if not version_static:
                msg = f"Required {module}, but installed {pkg_version}."
                require_available = False
            else:
                msg = f"Requirement {module} is available, found `{pkg}={pkg_version}`"  # noqa E501

        _imported_caches[module] = (require_available, msg)

    all_available = all(
        list(map(lambda x: x[0], list(_imported_caches.values())))
    )

    if not all_available:
        missing_msgs = os.linesep.join(
            [x[1] for x in list(_imported_caches.values()) if not x[0]]
        )
        msg = f"Required dependencies is not available: {missing_msgs}; \n"
        if len(raise_msg) != 0:
            msg += f"{raise_msg}; \n"
        if raise_exception:
            raise ModuleNotFoundError(msg)

        logger.warning(format_msg(msg, color=MSGColor.RED))

    return all_available


def require_packages(
    *package_versions: str,
    raise_exception: bool = True,
    raise_msg: str = "",
    compare_full_version: bool = True,
) -> Callable:
    """Check whether package is installed and check its version(optional).

    Args:
        package_versions: Python package name (e.g. `torch`, `torch>=2.0`).
        raise_exception: Whether to raise exception when package not available.
        raise_msg: Suggested actions after exception.
        compare_full_version: Whether to compare full version. Default True.
            If False, will only compare public version.

    Examples:
        >>> @require_packages("torch", raise_exception=False)
        ... def my_func():
        ...     import torch
        ...     return torch.__version__

        >>> class MyCls:
        ...     @require_packages("torch>=1.0", "torchvision")
        ...     def __init__(self):
        ...         import torch
        ...         import torchvision
        ...         print(torch.__version__)
        ...         print(torchvision.__version__)
    """

    def decorator(func) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = "The exception may have been exposed "
            msg += f"from {func.__qualname__}; \n"
            msg += raise_msg
            check_packages_available(
                *package_versions,
                raise_exception=raise_exception,
                raise_msg=msg,
                compare_full_version=compare_full_version,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_error_if_import_failed(package, package_name, err_msg=None):
    if package is None:
        if err_msg is None:
            raise ModuleNotFoundError(f"Cannot import module {package_name}")
        else:
            raise ModuleNotFoundError(
                f"Cannot import module {package_name}, err_msg = {err_msg}"
            )  # noqa
