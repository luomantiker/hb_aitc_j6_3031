import faulthandler
import logging
from functools import partial
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlparse  # noqa: F401

import torch
from typing_extensions import TypeAlias

try:
    from torch.serialization import MAP_LOCATION
except ImportError:
    MAP_LOCATION: TypeAlias = Optional[
        Union[
            Callable[[torch.Tensor, str], torch.Tensor],
            torch.device,
            str,
            Dict[str, str],
        ]
    ]

logger = logging.getLogger(__name__)

__all__ = [
    "download_url_to_file_with_retry",
    "load_state_dict_from_url",
]

original_download_url_to_file = torch.hub.download_url_to_file


def download_url_to_file_with_retry(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
    max_retry: int = 3,
):
    """Download object at the given URL to a local path.

    Args:
        url: URL of the object to download
        dst: Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix: If not None, the SHA256 downloaded file should start
            with ``hash_prefix``. Default: None
        progress: whether or not to display a progress bar to stderr.
            Default: True
        max_retry: Retry times if download failed.

    """

    for i in range(1, max_retry + 1):
        try:
            original_download_url_to_file(
                url=url,
                dst=dst,
                hash_prefix=hash_prefix,
                progress=progress,
            )
            break
        except Exception as e:
            if i == max_retry:
                logger.warning(
                    f"Download failed and has retried {max_retry} times:{e}"
                )
            else:
                logger.warning(
                    f"Download failed caused by: {e}. "
                    f"Remaining_retry: {max_retry - i}."
                )


def load_state_dict_from_url(
    url: str,
    map_location: MAP_LOCATION = None,
    check_hash: bool = False,
    max_retry: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Load the Torch serialized object at the given URL.

    Note: this function is same to `torch.hub.load_state_dict_from_url`,
        but with added retry download.

    Args:
        url: URL of the object to download
        map_location: a function or a dict specifying how to remap storage
            locations (see `torch.load`)
        check_hash: If True, the filename part of the URL should follow the
            naming convention ``filename-<sha256>.ext`` where ``<sha256>`` is
            the first eight or more digits of the SHA256 hash of the contents
            of the file. The hash is used to ensure unique names and to verify
            the contents of the file.
            Default: False
        max_retry: Retry times if download failed.
        kwargs: Args of `torch.hub.load_state_dict_from_url`.

    """
    try:
        # hook to replace `download_url_to_file`
        torch.hub.download_url_to_file = partial(
            download_url_to_file_with_retry, max_retry=max_retry
        )
        return torch.hub.load_state_dict_from_url(
            url=url,
            map_location=map_location,
            check_hash=check_hash,
            **kwargs,
        )
    finally:
        # rollback to avoid unknown or unexpected problem
        torch.hub.download_url_to_file = original_download_url_to_file


def hook_torch_signal_handler():
    """Replace torch dataloader signal handler.

    Note:
        This hook is designed to print stack information when the dataloader
        worker encounters a signal error.
    """
    orig_set_worker_signal_handler = (
        torch.utils.data._utils.signal_handling._set_worker_signal_handlers
    )

    def set_worker_signal_handlers(*arg: Any):
        faulthandler.disable()
        orig_set_worker_signal_handler(*arg)
        faulthandler.enable(all_threads=True)

    torch.utils.data._utils.signal_handling._set_worker_signal_handlers = (
        set_worker_signal_handlers
    )


# hook torch dataloader signal handler
hook_torch_signal_handler()
