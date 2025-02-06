import importlib
import logging
import os

from .logger import DisableLogger
from .package_helper import check_packages_available, parse_package_spec

logger = logging.getLogger(__name__)


__all__ = ["optional_package_check", "static_package_check", "package_check"]


global TORCH_INDEX


torch_deps = [
    [
        "torch==1.10.2+cu102",
        "horizon_plugin_pytorch>=0.15.0",
        "torchvision==0.11.3+cu102",
        "torchaudio==0.10.2+cu102",
    ],
    [
        "torch==1.10.2+cu111",
        "horizon_plugin_pytorch>=0.15.0",
        "torchvision==0.11.3+cu111",
        "torchaudio==0.10.2+cu111",
    ],
    [
        "torch==1.10.2+cpu",
        "horizon_plugin_pytorch>=0.15.0",
        "torchvision==0.11.3+cpu",
        "torchaudio==0.10.2+cpu",
    ],
    [
        "torch==1.13.0+cu116",
        "horizon_plugin_pytorch>=1.3.0",
        "torchvision==0.14.0+cu116",
        "torchaudio==0.13.0+cu116",
    ],
    [
        "torch==1.13.0+cpu",
        "horizon_plugin_pytorch>=1.3.0",
        "torchvision==0.14.0+cpu",
        "torchaudio==0.13.0+cpu",
    ],
    [
        "torch==2.0.1+cu118",
        "horizon_plugin_pytorch>=2.0.0",
        "torchvision==0.15.2+cu118",
        "torchaudio==2.0.2+cu118",
    ],
    [
        "torch==2.0.1+cpu",
        "horizon_plugin_pytorch>=2.0.0",
        "torchvision==0.15.2+cpu",
        "torchaudio==2.0.2+cpu",
    ],
    [
        "torch==2.1.0+cu118",
        "horizon_plugin_pytorch>=2.2.0",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
    ],
    [
        "torch==2.1.0+cpu",
        "horizon_plugin_pytorch>=2.2.0",
        "torchvision==0.16.0+cpu",
        "torchaudio==2.1.0+cpu",
    ],
    [
        "torch==2.3.0+cu118",
        "horizon_plugin_pytorch>=2.4.5",
        "torchvision==0.18.0+cu118",
        "torchaudio==2.3.0+cu118",
    ],
    [
        "torch==2.3.0+cpu",
        "horizon_plugin_pytorch>=2.4.5",
        "torchvision==0.18.0+cpu",
        "torchaudio==2.3.0+cpu",
    ],
]

if int(os.environ.get("USE_DCU", 0)) == 1:
    torch_deps = [
        [
            "torch==1.10.0",
            "horizon_plugin_pytorch>=1.8.3",
            "torchvision==0.10.0",
            "torchaudio==0.10.0+a9847c3",
        ]
    ]


def static_package_check():
    # check basic torch deps
    torch_status = False
    for i, deps in enumerate(torch_deps):
        with DisableLogger(enable=True):
            result = check_packages_available(*deps[:2], raise_exception=False)
        torch_status = torch_status or result
        if torch_status:
            global TORCH_INDEX
            TORCH_INDEX = i
            break

    if not torch_status:
        logger.warning(
            f"You should check the version of torch、"
            f"horizon_pytorch_plugin. Make sure they meet any of the"
            f" conditions in {torch_deps[:2]}"
        )


optional_package = [
    "hatbc>0.9.0",
    "aidisdk>0.14.0",
]


def optional_package_check():
    # check torch related
    torch_dep = torch_deps[TORCH_INDEX]
    opt_package = torch_dep[2:] + optional_package

    for package in opt_package:
        pkg, _, _ = parse_package_spec(package)
        try:
            importlib.import_module(pkg)
        except ImportError:
            logger.warning(f"Optional requirements: {pkg} is not available!")
            continue

        # check version while import successfully
        check_packages_available(package, raise_exception=True)


def package_check():
    static_package_check()
    optional_package_check()
