import logging
from numbers import Integral, Real
from typing import List, Optional

import torch
from torch import Tensor
from torch._jit_internal import _overload
from torch.nn.functional import interpolate

from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.march import March, get_march

logger = logging.getLogger(__name__)

######################################################################
# torch interpolate raises Exception with cuda bfloat16 amp
# https://github.com/pytorch/pytorch/issues/86679
# TODO: remove this after interpolate support cuda bfloat16


@_overload
def autocasted_interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    pass


@fx_helper.wrap()
def autocasted_interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    if input.device.type == "cuda" and input.dtype == torch.bfloat16:
        input = input.float()
    assert not antialias, "antialias is not supported"
    return interpolate(
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
    )


@_overload
def autocasted_interpolate_outer(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate_outer(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate_outer(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    pass


@_overload
def autocasted_interpolate_outer(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    pass


# use a outer func to help fx correctly find the wrapped inner
def autocasted_interpolate_outer(  # noqa: F811
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    return autocasted_interpolate(
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias,
    )


torch.nn.functional.interpolate = autocasted_interpolate_outer
######################################################################


class Interpolate(torch.nn.Module):
    r"""Resize for float training.

    Support bilinear and nearest interpolate method and NCHW input.
    The behaviour is same as torch.nn.functional.interpolate except the default
    mode is 'bilinear'

    Parameters
    ----------
    size : int or tuple of int, optional
        the output shape of resize: if int, the output shape is (size, size)
        else the output shape is (out_height, out_width), by default None
        size and scale_factor shouldn't be set at the same time
    scale_factor : float or tuple of float, optional
        the ratio of output shape to input shape, ie. out_shape / in_shape,
        or (out_height / in_height, out_width / in_width), by default None
        size and scale_factor shouldn't be set at the same time
    mode : str, optional
        the interpolate method, by default "bilinear",
        support "bilinear" and "nearest"
    align_corners : bool, optional
    recompute_scale_factor : bool, optional
        did not support, by default None
    antialias: bool, optional
        flag to apply anti-aliasing, not supported yet
    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        super(Interpolate, self).__init__()
        assert not antialias, "antialias is not supported"
        assert isinstance(size, (Integral, type(None))) or (
            isinstance(size, (tuple, list))
            and len(size) == 2
            and isinstance(size[0], Integral)
            and isinstance(size[1], Integral)
        ), "param 'size' must be int or tuple of two int or None"
        assert isinstance(scale_factor, (Real, type(None))) or (
            isinstance(scale_factor, (tuple, list))
            and len(scale_factor) == 2
            and isinstance(scale_factor[0], Real)
            and isinstance(scale_factor[1], Real)
        ), "param 'scale_factor' must be real or tuple of two real or None"
        assert mode in (
            "bilinear",
            "nearest",
        ), "mode only support 'bilinear' and 'nearest'"
        if mode == "nearest":
            assert (
                align_corners is None
            ), "align_corners option can only be set with 'bilinear' mode"
        else:
            if align_corners is None:
                logger.warning(
                    f"default upsampling behavior when mode={mode} is changed "
                    f"to align_corners=False since torch 0.4.0. Please specify"
                    f" align_corners=True if the old behavior "
                    f"is desired. ",
                    extra={"call_times_context": ("message")},
                )
                align_corners = False
            assert isinstance(
                align_corners, bool
            ), "param 'align_corners' must be bool or None"

        if (
            get_march() in (March.BERNOULLI, March.BERNOULLI2)
            and align_corners
        ):
            raise ValueError(
                "align_corners = True is not supported "
                "on {} or {}".format(March.BERNOULLI, March.BERNOULLI2)
            )

        assert isinstance(
            recompute_scale_factor, (bool, type(None))
        ), "param 'recompute_scale_factor' must be bool or None"
        if scale_factor:
            assert (
                size is None
            ), "only one of size or scale_factor should be defined"
            assert recompute_scale_factor, (
                "only support recompute_scale_factor=True "
                + "when using scale_factor"
            )

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, data):
        return torch.nn.functional.interpolate(
            data,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
