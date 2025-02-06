# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from hat.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

__all__ = ["GradScale", "GradClip", "DelayGradScale"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GradClip(CallbackMixin):
    """Grad Clip callback.

    Args:
        type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        max_norm: Max norm of the gradients.
        norm_type: Type of the used p-norm. Default to 2.
        nan: the value to replace NaNs with. Default is zero.
        posinf: if a Number, the value to replace positive infinity values
            with. If None, positive infinity values are replaced with the
            greatest finite value representable by input's dtype. Default
            is None.
        neginf: if a Number, the value to replace negative infinity values
            with. If None, negative infinity values are replaced with the
            lowest finite value representable by input's dtype. Default
            is None.
        nonfinit_to_num: Whether to relace nan/inf/-inf value of grad
            to num.
        skip_clip: Whether to skip clip grad.
    """

    def __init__(
        self,
        max_norm: float,
        norm_type: int = 2,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        nonfinit_to_num: bool = False,
        skip_clip: bool = False,
    ) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.nonfinit_to_num = nonfinit_to_num
        self.skip_clip = skip_clip
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

        assert not (
            skip_clip and not nonfinit_to_num
        ), "should not skip both clip and nonfinit_to_num"

    def _grad_clip_(self, model) -> None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

    def _grad_nan_to_num_(self, model) -> None:
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, self.nan, self.posinf, self.neginf)

    def on_optimizer_step_begin(
        self, model, optimizer, grad_scaler, **kwargs
    ) -> None:
        grad_scaler.unscale_(optimizer)

        if self.nonfinit_to_num:
            self._grad_nan_to_num_(model)

        if not self.skip_clip:
            self._grad_clip_(model)


@OBJECT_REGISTRY.register
class GradScale(GradClip):
    """Set gradient scale for different module in each backward immediately.

    When training multitask, gradient of each task might be different.
    Comparing to changing loss weight, another more efficient method is to
    adjust gradient.

    Example:
        >>> grad_scale_callback = dict(
        ...    type="GradScale",
        ...    module_and_scale=[
        ...        ("backbone", 0.1, "real3d_fov120"),
        ...        ("bifpn", 0.1, "real3d_fov120"),
        ...        ("real3d_fov120", 1.0, "real3d_fov120"),
        ...    ],
        ...    clip_grad_norm=None,
        ...)

    Args:
        module_and_scale: module name, gradient scale and task name. Task name
            can be none if you don't need.
        clip_grad_norm: Max norm for `torch.nn.utils.clip_grad_norm_`.
        clip_norm_type: Norm type for `torch.nn.utils.clip_grad_norm_`.
        nan: the value to replace NaNs with. Default is zero.
        posinf: if a Number, the value to replace positive infinity values
            with. If None, positive infinity values are replaced with the
            greatest finite value representable by input's dtype. Default
            is None.
        neginf: if a Number, the value to replace negative infinity values
            with. If None, negative infinity values are replaced with the
            lowest finite value representable by input's dtype. Default
            is None.
        nonfinit_to_num: Whether to relace nan/inf/-inf value of grad
            to num.
    """

    def __init__(
        self,
        module_and_scale: List,
        clip_grad_norm: Optional[float] = None,
        clip_norm_type: Optional[int] = 2,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        nonfinit_to_num: Optional[bool] = False,
    ):
        super().__init__(
            clip_grad_norm,
            clip_norm_type,
            nan,
            posinf,
            neginf,
            nonfinit_to_num,
        )
        self.module_and_scale = module_and_scale
        self._grad_cache = defaultdict(float)
        self.skip_scale = False

        if len(module_and_scale) <= 0:
            logger.warning(
                "[GradScale] No module_and_scale info, please use GradClip."
            )
            self.skip_scale = True

    def on_loop_begin(self, **kwargs):
        logger.info(f"[GradScale] {self.module_and_scale}")

    def on_backward_end(self, model, batch, optimizer, **kwargs):
        """Task-wise backward_end."""
        if self.skip_scale:
            return
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if self.nonfinit_to_num:
                torch.nan_to_num_(
                    param.grad, self.nan, self.posinf, self.neginf
                )
            for item in self.module_and_scale:
                # get task from item
                if len(item) == 3:
                    module, scale, task = item
                elif len(item) == 2:
                    module, scale = item
                    task = None
                else:
                    raise ValueError(f"Unvalid args: {item}")
                # do scale
                if (module in name) and (not task or task in batch):
                    param.grad *= scale
            # cache grad
            self._grad_cache[name] += param.grad.detach()
        optimizer.zero_grad(set_to_none=False)

    def on_optimizer_step_begin(self, model, optimizer, grad_scaler, **kwargs):
        grad_scaler.unscale_(optimizer)

        if self.skip_scale and self.nonfinit_to_num:
            # do nan/inf/-inf to num
            self._grad_nan_to_num_(model)
        elif not self.skip_scale:
            # move grad from cache to param.grad
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad = self._grad_cache[name]

        # do clip grad norm
        if self.max_norm:
            self._grad_clip_(model)

        # empty grad cache
        self._grad_cache.clear()


@OBJECT_REGISTRY.register
class DelayGradScale(GradClip):
    """Delay setting gradient scale for different module.

    This callback enable delay scale to speed up grad scale, if there
    is no need to scale some param gradient in the grad accumulate progress.
    If you want to make sure the grad accumulate progress is correct, please
    use `GradScale`. But we should traverse all params then scale the grad,
    which may be time-consuming.

    .. Note::
        If you want to use this callback, You need to determine whether there
        are cases during gradient accumulation where some gradients do not need
        to be scaled. If there are, even though this might make `GradScale`
        itself a bottleneck in training speed, you should use `GradScale` to
        ensure the correctness of the gradient scaling logic.

    Example:
        >>> delay_grad_scale_callback = dict(
        ...    type="DelayGradScale",
        ...    module_and_scale=[
        ...        ("backbone", 0.1, "real3d_fov120"),
        ...        ("bifpn", 0.1, "real3d_fov120"),
        ...        ("real3d_fov120", 1.0, "real3d_fov120"),
        ...    ],
        ...    clip_grad_norm=None,
        ...)

    Args:
        module_and_scale: module name, gradient scale and task name. Task name
            can be none if you don't need.
        clip_grad_norm: Max norm for `torch.nn.utils.clip_grad_norm_`.
        clip_norm_type: Norm type for `torch.nn.utils.clip_grad_norm_`.
        nan: the value to replace NaNs with. Default is zero.
        posinf: if a Number, the value to replace positive infinity values
            with. If None, positive infinity values are replaced with the
            greatest finite value representable by input's dtype. Default
            is None.
        neginf: if a Number, the value to replace negative infinity values
            with. If None, negative infinity values are replaced with the
            lowest finite value representable by input's dtype. Default
            is None.
        nonfinit_to_num: Whether to relace nan/inf/-inf value of grad
            to num.
    """

    def __init__(
        self,
        module_and_scale: List,
        clip_grad_norm: Optional[float] = None,
        clip_norm_type: Optional[int] = 2,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        nonfinit_to_num: Optional[bool] = False,
    ):
        assert len(module_and_scale) > 0, "module_and_scale should be set!"
        super().__init__(
            clip_grad_norm,
            clip_norm_type,
            nan,
            posinf,
            neginf,
            nonfinit_to_num,
        )
        self.raw_module_and_scale_info = module_and_scale
        self._group_param_cache: Dict[str, Dict[str, Any]] = None
        self.map_dict = self._format_module_and_scale(module_and_scale)
        self._need_scale_task_set = set()

    @staticmethod
    def _group_tensors_by_module_name_and_task_name(
        model: torch.nn.Module, map_dict: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        group_tensors_dict = {}
        for task, module_and_scale in map_dict.items():
            if task not in group_tensors_dict:
                group_tensors_dict[task] = {}

            for module_name, scale in module_and_scale.items():
                for name, param in model.named_parameters():
                    if module_name in name:
                        if module_name not in group_tensors_dict[task]:
                            group_tensors_dict[task][module_name] = {
                                "scale": scale,
                                "tensors": [],
                            }
                        group_tensors_dict[task][module_name][
                            "tensors"
                        ].append(param)
        return group_tensors_dict

    def _format_module_and_scale(
        self, module_and_scale: List[Tuple[str, float, Optional[str]]]
    ) -> Dict[str, Dict[str, float]]:
        map_dict = {}
        module_name_set = set()
        for item in module_and_scale:
            # get task from item
            if len(item) == 3:
                module, scale, task = item
                if not task:
                    task = "all"
            elif len(item) == 2:
                module, scale = item
                task = "all"
            else:
                raise ValueError(f"Unvalid args: {item}")
            if task not in map_dict:
                map_dict[task] = {}
            map_dict[task][module] = scale
            if module in module_name_set:
                logger.warning(
                    f"module name {module} is duplicate, "
                    "please make sure scale value in all the same "
                    "module name is the same."
                )
            module_name_set.add(module)
        return map_dict

    def on_loop_begin(self, **kwargs) -> None:
        logger.info(f"[DelayGradScale] {self.raw_module_and_scale_info}")

    def on_backward_end(self, model, batch, optimizer, **kwargs) -> None:
        # group tensors by module name and task name
        # and cache it in first step
        # avoid to group tensors every step
        if self._group_param_cache is None:
            self._group_param_cache = (
                DelayGradScale._group_tensors_by_module_name_and_task_name(
                    model=model, map_dict=self.map_dict
                )
            )

        for task, _ in self.map_dict.items():
            if task == "all":
                self._need_scale_task_set.add(task)
            elif isinstance(batch, dict) and task in batch:
                self._need_scale_task_set.add(task)
            elif isinstance(batch, tuple) and task == batch[1]:
                self._need_scale_task_set.add(task)

    @torch.no_grad()
    def _do_grad_scaled(self) -> None:
        for task in self._need_scale_task_set:
            for _, scale_params in self._group_param_cache[task].items():
                grads = [
                    p.grad
                    for p in scale_params["tensors"]
                    if p.grad is not None
                ]
                torch._foreach_mul_(grads, scale_params["scale"])

    def on_optimizer_step_begin(
        self, model, optimizer, grad_scaler, **kwargs
    ) -> None:
        grad_scaler.unscale_(optimizer)

        # do nan/inf/-inf to num
        if self.nonfinit_to_num:
            self._grad_nan_to_num_(model)

        # delay scale gradient
        self._scaled_tensors_group = self._do_grad_scaled()

        # do clip grad norm
        if self.max_norm:
            self._grad_clip_(model)

        # clear scale task set
        self._need_scale_task_set.clear()
