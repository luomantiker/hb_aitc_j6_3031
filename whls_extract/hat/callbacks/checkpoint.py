# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.distributed import (
    all_gather_object,
    get_device_count,
    get_dist_info,
    rank_zero_only,
)
from hat.utils.elastic import ElasticState, use_elastic
from hat.utils.hash import generate_sha256_file
from .callbacks import CallbackMixin

try:
    import apex
except ImportError:
    apex = None

try:
    import deepspeed
except ImportError:
    deepspeed = None

__all__ = ["Checkpoint"]

logger = logging.getLogger(__name__)

TRAIN_CHECKPOINT_FORMAT = "%scheckpoint-%s.pth.tar"


def get_valid_state_dict(model: nn.Module, only_save_ddp=False) -> dict:
    """Remove 'module' prefix in param name if model is ddp or dp model.

    Args:
        model: Module that need to get state dict.
        only_save_ddp: Only save ddp module.

    Returns:
        Dict of param names and values
    """
    if isinstance(
        model, (nn.parallel.DistributedDataParallel, nn.DataParallel)
    ):
        return model.module.state_dict()
    elif deepspeed is not None and isinstance(
        model, deepspeed.DeepSpeedEngine
    ):
        return model.module.state_dict()
    elif apex is not None and isinstance(
        model, apex.parallel.distributed.DistributedDataParallel
    ):
        return model.module.state_dict()

    elif isinstance(model, nn.Module):
        ddp_module_dict = {}
        for name, module in model.named_children():
            if isinstance(module, nn.parallel.DistributedDataParallel):
                ddp_module_dict[name] = module

        if ddp_module_dict:
            # only save ddp module
            if only_save_ddp:
                model_bak = {}
                for name, mod in ddp_module_dict.items():
                    for key, values in mod.state_dict().items():
                        model_bak[key.replace("module", name)] = values
                return model_bak

            # support submodule is ddp.
            model_bak = copy.deepcopy(model)
            for name, module in ddp_module_dict.items():
                setattr(model_bak, name, module.module)
            return model_bak.state_dict()
        else:
            return model.state_dict()
    else:
        raise NotImplementedError("unknown model type: %s" % type(model))


def _get_average_tensor(tensors: List[Tensor]):
    """Get average tensor of a list of tensors.

    When compute average for low precision tensor, using high precision dtype.
    """
    with torch.no_grad():
        origin_type = tensors[0].dtype
        for i, t in enumerate(tensors):
            assert (
                origin_type == t.dtype
            ), f"{i}th tensor type is {t.dtype} not {origin_type}"

        if origin_type == torch.float16:
            tmp = tensors[0].to(torch.float32)
        elif origin_type == torch.uint8:
            tmp = tensors[0].to(torch.int32)
        elif origin_type == torch.int8:
            tmp = tensors[0].to(torch.int32)
        else:
            tmp = tensors[0]

        for t in tensors[1:]:
            tmp = tmp.add(t.to(tmp.device))
        tmp = tmp / len(tensors)
        return tmp.to(origin_type)


@OBJECT_REGISTRY.register
class Checkpoint(CallbackMixin):  # noqa: D205,D400
    """
    Checkpoint Callback is used for saving model after training
    and resume model before training as the same times.

    Args:
        save_dir: Directory to save checkpoints.
        name_prefix: Checkpoint name prefix.
        save_interval: Save checkpoint every `save_interval` epoch or step.
        interval_by: Set `save_interval` unit to step or epoch.
            Default is epoch.
        save_on_train_end: Whether save checkpoint when `on_loop_end` is
            triggered.
        strict_match: Whether to strictly enforce that the keys in
            `model.state_dict()` (train model) match the keys in
            `test_model.state_dict()`. Default: ``False``
        mode: State of monitor for saving model.
        monitor_metric_key: Monitor metric for saving best checkpoint.
        best_refer_metric: Metric that evaluate which epoch is the best.
        save_hash: Whether to save the hash value to the name of the
             Checkpoint file. Default is True.
        only_save_ddp: Only save ddp module. Default is False.
        keep_n: Keep the last n checkpoints. Default is None.
    """

    SUPPORTED_MODES = ["min", "max"]

    def __init__(
        self,
        save_dir: str,
        name_prefix: Optional[str] = "",
        save_interval: Optional[int] = 1,
        interval_by: Optional[str] = "epoch",
        save_on_train_end: Optional[bool] = True,
        strict_match: Optional[bool] = False,
        mode: Optional[str] = None,
        monitor_metric_key: Optional[str] = None,
        best_refer_metric: Optional[Union[dict, EvalMetric]] = None,
        task_sampler=None,
        save_hash: bool = True,
        only_save_ddp: bool = False,
        keep_n: Optional[int] = None,
    ):
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.save_interval = save_interval
        self.interval_by = interval_by
        self.save_on_train_end = save_on_train_end
        self.strict_match = strict_match
        self.task_sampler = task_sampler
        self.save_hash = save_hash
        self.only_save_ddp = only_save_ddp

        assert self.interval_by in ("epoch", "step")

        self.mode = mode
        self.monitor_metric_key = monitor_metric_key
        if best_refer_metric is not None:
            self.best_refer_metric = _as_list(best_refer_metric)
        else:
            self.best_refer_metric = None
        if self.mode is not None:
            if self.mode == "min":
                self.monitor_op = np.less
                self.best = np.Inf
            elif self.mode == "max":
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                raise ValueError(
                    "Supported modes: %s, but get %s"
                    % (self.SUPPORTED_MODES, self.mode)
                )

        self.keep_n = keep_n
        self.epoch_ckpt_files = []

    @rank_zero_only
    def save_to_file(
        self,
        state,
        epoch_id,
        save_best,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        step_id=None,
    ):

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

        assert save_type in [
            "epoch",
            "step",
        ], f"save_type should be one of ['epoch', 'step'], but get {save_type}"
        if save_epoch_or_step:
            ckpt_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (
                    self.name_prefix,
                    "step-%d" % step_id
                    if save_type == "step"
                    else "epoch-%04d" % epoch_id,
                ),
            )
            torch.save(state, ckpt_file)
            if use_elastic():
                ElasticState.save_checkpoint(ckpt_file)
            if self.save_hash:
                ckpt_file = generate_sha256_file(ckpt_file)
            logger.info("Save model checkpoint: %s" % ckpt_file)

            self.epoch_ckpt_files.append(ckpt_file)
            # remove old ckpt files to save the disk space
            if (
                self.keep_n is not None
                and len(self.epoch_ckpt_files) > self.keep_n
            ):
                for f in self.epoch_ckpt_files[: -self.keep_n]:
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old checkpoint file: {f},",
                            "please check the reason.",
                        )
                        logger.warning(
                            "1. Make sure there are not two or more jobs, "
                            "which have the same path for saving models; "
                            "2. Make sure you have the permission to "
                            "manipulate the folder where the models are saved."
                        )
                        logger.warning(str(e))
                self.epoch_ckpt_files = self.epoch_ckpt_files[-self.keep_n :]

        if save_last:
            last_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "last"),
            )
            torch.save(state, last_file)

            if use_elastic():
                ElasticState.save_checkpoint(last_file)

            if self.save_hash:
                last_file = generate_sha256_file(last_file, remove_old=True)
            logger.info("Save last model checkpoint: %s" % last_file)

        if save_best:
            best_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "best"),
            )
            best_state = {}
            for k in state:
                if k == "ema_model":
                    continue
                elif k == "state_dict" and "ema_model" in state.keys():
                    best_state[k] = state["ema_model"]
                else:
                    best_state[k] = state[k]

            torch.save(best_state, best_file)

            if self.save_hash:
                best_file = generate_sha256_file(best_file, remove_old=True)
            logger.info("Save best model checkpoint: %s" % best_file)

    def _sync_model(self, model):
        """
        Sync model's buffers that distributed on diff host.

        Gather all buffers from diff host, then merge them together.
        When the buffer of diff host has diff values, compute average of them.
        """
        if self.task_sampler is None or not self.task_sampler.is_parallel():
            return

        if not isinstance(model, nn.parallel.DistributedDataParallel):
            return
        else:
            model = model.module
            if not hasattr(model, "named_parameters_by_outname"):
                return

        with torch.no_grad():
            tasks = self.task_sampler.tasks
            # get current task params and buffers
            states = {}
            for n, b in model.named_buffers_by_outname(tasks):
                states[n] = b

            # gather all state from every ranks
            rank, world_size = get_dist_info()
            global_states = [None for _ in range(world_size)]
            all_gather_object(global_states, states)

            merged_states = {}  # {p_name => [buffer, cnt]}
            for s in global_states:
                for k, v in s.items():
                    if k not in merged_states:
                        merged_states[k] = [v]
                    else:
                        merged_states[k].append(v)

            # check and average same tensor on different ranks
            def _merge_tensor(key, tensor):
                # check all buffer in model state_dict
                if k not in merged_states:
                    logger.warning(f"tensor: {k} not updated on training.")
                    return

                v = merged_states[k]
                if len(v) == 1:
                    # set into model state dict directly
                    v = v[0]
                else:
                    v = _get_average_tensor(v)
                if tensor.shape != v.shape:
                    tensor.resize_(v.shape)
                tensor.copy_(v)

            # check and average same buffer on different ranks
            for k, buff in model.named_buffers():
                _merge_tensor(k, buff)

    def save_checkpoint(
        self,
        model,
        optimizer,
        save_best,
        storage,
        ema_model=None,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        epoch_id=None,
        step_id=None,
    ):
        self._sync_model(model)
        try:
            grad_scaler_state = storage.get("grad_scaler")[0]
        except Exception:
            grad_scaler_state = {}

        from horizon_plugin_pytorch import __version__

        contiguous_model = model.to(memory_format=torch.contiguous_format)

        state = {
            "epoch": epoch_id,
            "step": step_id,
            "devices": get_device_count(),
            "grad_scaler": grad_scaler_state,
            "state_dict": get_valid_state_dict(
                contiguous_model, self.only_save_ddp
            ),
            "horizon-plugin-version": __version__,
        }
        if ema_model is not None:
            ema_c_model = ema_model.to(memory_format=torch.contiguous_format)
            state["ema_model"] = get_valid_state_dict(
                ema_c_model, self.only_save_ddp
            )

        if deepspeed is not None and isinstance(
            contiguous_model, deepspeed.DeepSpeedEngine
        ):
            tag = "%s-ds-ckpt-%s" % (
                self.name_prefix,
                "step-%d" % step_id
                if save_type == "step"
                else "epoch-%04d" % epoch_id,
            )
            contiguous_model.save_checkpoint(
                save_dir=self.save_dir,
                tag=tag,
                client_state=state,
            )
            if save_last:
                tag = "ds-last-ckpt"
                contiguous_model.save_checkpoint(
                    save_dir=self.save_dir,
                    tag=tag,
                    client_state=state,
                )
            if save_best:
                tag = "ds-best-ckpt"
                contiguous_model.save_checkpoint(
                    save_dir=self.save_dir,
                    tag=tag,
                    client_state=state,
                )
        else:
            if optimizer is not None:
                state["optimizer"] = optimizer.state_dict()
            else:
                state["optimizer"] = None

            self.save_to_file(
                state,
                epoch_id,
                save_best,
                save_epoch_or_step,
                save_type,
                save_last,
                step_id,
            )

    def _get_ckp_model(self, model, ema_model):
        ckp_model = model
        if ema_model is not None:
            ckp_model = ema_model
        return ckp_model

    def on_loop_begin(self, loop, **kwargs):
        if self.best_refer_metric is not None:
            for m in self.best_refer_metric:
                m.to(loop.device)

    def on_step_end(
        self,
        epoch_id,
        global_step_id,
        val_metrics,
        storage=None,
        model=None,
        ema_model=None,
        optimizer=None,
        **kwargs,
    ):
        # ckp_model = self._get_ckp_model(model, ema_model)

        if self.interval_by == "step" and (
            (global_step_id + 1) % self.save_interval == 0
        ):
            self.do_checkpoint(
                model,
                optimizer,
                epoch_id,
                ema_model=ema_model,
                save_type="step",
                step_id=global_step_id,
                val_metrics=val_metrics,
                storage=storage,
            )

    def on_epoch_end(
        self,
        epoch_id,
        global_step_id,
        model,
        optimizer,
        val_metrics,
        storage=None,
        ema_model=None,
        **kwargs,
    ):
        # ckp_model = self._get_ckp_model(model, ema_model)

        if self.interval_by == "epoch" and (
            (epoch_id + 1) % self.save_interval == 0
        ):
            self.do_checkpoint(
                model,
                optimizer,
                epoch_id,
                ema_model=ema_model,
                save_type="epoch",
                step_id=global_step_id,
                val_metrics=val_metrics,
                storage=storage,
            )

    def do_checkpoint(
        self,
        model,
        optimizer,
        epoch_id,
        storage,
        ema_model=None,
        save_type="epoch",
        step_id=None,
        val_metrics=None,
    ):
        if self.mode is not None:
            if self.best_refer_metric is not None:
                metrics = self.best_refer_metric
            else:
                assert val_metrics is not None
                metrics = val_metrics
            value = None
            for val_metric in metrics:
                if hasattr(val_metric, "fast_get"):
                    names, values = val_metric.fast_get()
                else:
                    names, values = val_metric.get()
                if self.monitor_metric_key is None:
                    values = _as_list(values)
                    if len(values) != 1:
                        raise KeyError(
                            "Cannot resolve more than one metric values"
                            + "while monitor_metric_key is None."
                        )
                    value = values[0]
                    break
                else:
                    if self.monitor_metric_key in names:
                        index = _as_list(names).index(self.monitor_metric_key)
                        value = _as_list(values)[index]
                        break
            assert value is not None

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            is_best = self.monitor_op(value, self.best)
            if is_best:
                logger.info("former best:%f, current:%f" % (self.best, value))
                self.best = value
        else:
            is_best = False

        # save train model
        self.save_checkpoint(
            model,
            ema_model=ema_model,
            epoch_id=epoch_id,
            optimizer=optimizer,
            save_best=is_best,
            save_type=save_type,
            step_id=step_id,
            storage=storage,
        )

    def on_loop_end(
        self,
        model,
        optimizer,
        epoch_id,
        global_step_id,
        storage=None,
        ema_model=None,
        **kwargs,
    ):
        if not self.save_on_train_end:
            return

        ckp_model = self._get_ckp_model(model, ema_model)

        self.save_checkpoint(
            ckp_model,
            epoch_id=epoch_id,
            step_id=global_step_id,
            optimizer=optimizer,
            save_best=False,
            save_epoch_or_step=False,
            save_last=True,
            storage=storage,
        )
