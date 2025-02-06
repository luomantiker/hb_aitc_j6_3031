import logging
import os
import pickle
from typing import Callable, Optional

import torch

from hat.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from hat.utils.distributed import all_gather_object, get_dist_info
from hat.utils.filesystem import check_file_exist_with_warning

logger = logging.getLogger(__name__)

__all__ = ["DumpData"]


@OBJECT_REGISTRY.register
class DumpData(CallbackMixin):  # noqa: D205,D400
    """DumpData Callback is used for saving data, such as batch_data,
        model_outs, model_grad, model_state during training or evaluation loop.

    Args:
        output_dir: Output dir of file to save.
        name_prefix: Name prefix of saved file.
        save_interval: Save dump files every `save_interval` epoch or step.
        interval_by: Set `save_interval` unit to step or epoch.
            Default is step.
        dump_batch_data: Whether to dump batch data.
        dump_model_outs: Whether to dump outputs of model.
        dump_model_grad: Whether to dump gradient of model.
        dump_model_param: Whether to dump param(weight/bias) of model.
        reformat_model_outs_fn: Callable function to reformat model outputs.
        overwrite: Whether to overwrite existing old files.
    """

    def __init__(
        self,
        output_dir: str,
        name_prefix: str = "",
        save_interval: int = 1,
        interval_by: str = "step",
        dump_batch_data: bool = True,
        dump_model_outs: bool = True,
        dump_model_grad: bool = True,
        dump_model_param: bool = True,
        reformat_model_outs_fn: Optional[Callable] = None,
        overwrite: bool = True,
    ):
        assert interval_by in [
            "step",
            "epoch",
        ], f"`interval_by` should be step or epoch, but get {interval_by}"

        self.output_dir = output_dir
        self.name_prefix = name_prefix
        self.save_interval = save_interval
        self.interval_by = interval_by
        self.dump_mode_outs = dump_model_outs
        self.dump_batch_data = dump_batch_data
        self.dump_model_grad = dump_model_grad
        self.dump_model_param = dump_model_param
        self.reformat_model_outs_fn = reformat_model_outs_fn
        self.overwrite = overwrite

        self.rank, self.global_world_size = get_dist_info()

        if self.rank == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
        self.dump_data = {}

    def on_forward_begin(self, batch, epoch_id, global_step_id, **kwargs):
        # dump batch data
        idx = kwargs.get("batch_idx", None)
        if self.dump_batch_data and self._do_save(epoch_id, global_step_id):
            np_batch = convert_numpy(batch)
            self._update_dump_data(
                key=f"epoch_{epoch_id}_step_{global_step_id}",
                sub_key=f"batch_data_{idx}"
                if idx is not None
                else "batch_data",
                data=np_batch,
            )

    def on_optimizer_step_begin(
        self, model, epoch_id, global_step_id, **kwargs
    ):
        if self.dump_model_param and self._do_save(epoch_id, global_step_id):
            self._dump_model_param(
                model,
                name="param_before_optimizer",
                epoch_id=epoch_id,
                global_step_id=global_step_id,
            )

    def on_forward_end(self, model_outs, epoch_id, global_step_id, **kwargs):
        # dump model_outs
        idx = kwargs.get("batch_idx", None)
        if self.dump_mode_outs and self._do_save(epoch_id, global_step_id):
            data_outs = model_outs
            if self.reformat_model_outs_fn is not None:
                data_outs = self.reformat_model_outs_fn(data_outs)

            if isinstance(data_outs, tuple) and (
                isinstance(data_outs[0], tuple)
                and hasattr(data_outs[0], "_fields")
            ):
                # for Tuple[namedtuple]
                dump_data = {}
                for data in model_outs:
                    for k, v in data._asdict().items():
                        dump_data[k] = convert_numpy(v)
            else:
                # normal tuple or dict
                dump_data = convert_numpy(data_outs)

            self._update_dump_data(
                key=f"epoch_{epoch_id}_step_{global_step_id}",
                sub_key=f"model_outs_{idx}"
                if idx is not None
                else "model_outs",
                data=dump_data,
            )

    def on_backward_end(self, model, epoch_id, global_step_id, **kwargs):
        # dump grad before optimizer
        idx = kwargs.get("batch_idx", None)
        if self.dump_model_grad and self._do_save(epoch_id, global_step_id):
            self._dump_model_grad(
                model,
                name=f"grad_before_optimizer_{idx}"
                if idx is not None
                else "grad_before_optimizer",
                epoch_id=epoch_id,
                global_step_id=global_step_id,
            )

    def on_step_end(self, **kwargs):
        # dump step data
        self._dump_data_to_file(
            data=self.dump_data, name_prefix=self.name_prefix
        )

    def _dump_model_grad(self, model, name, epoch_id, global_step_id):
        grads = {}
        for _name, _param in model.named_parameters():
            if _param.requires_grad and _param.grad is not None:
                grads[_name] = convert_numpy(_param.grad)

        self._update_dump_data(
            key=f"epoch_{epoch_id}_step_{global_step_id}",
            sub_key=name,
            data=grads,
        )

    def _dump_model_param(self, model, name, epoch_id, global_step_id):
        params = {}
        for _name, _param in model.named_parameters():
            params[_name] = convert_numpy(_param)
        self._update_dump_data(
            key=f"epoch_{epoch_id}_step_{global_step_id}",
            sub_key=name,
            data=params,
        )

    def _dump_data_to_file(self, data, name_prefix):

        global_out_data = [None for _ in range(self.global_world_size)]
        all_gather_object(global_out_data, data)

        if self.rank == 0:
            dump_file = os.path.join(
                self.output_dir,
                name_prefix + ".pkl",
            )

            # check file
            check_file_exist_with_warning(dump_file, self.overwrite)

            # dump to file
            with open(dump_file, "wb") as f:
                pickle.dump(global_out_data, f)
        if self.global_world_size > 1:
            torch.distributed.barrier()

    def _do_save(self, epoch_id, global_step_id):
        if self.interval_by == "epoch":
            return (epoch_id + 1) % self.save_interval == 0
        else:
            return (global_step_id + 1) % self.save_interval == 0

    def _update_dump_data(self, key, sub_key, data):
        """Update dump data.

        dump_data format:
            {
                epoch_0_step_0: {
                    batch_data_0: data,
                    batch_data_1: data,
                    model_outs_0: data,
                    model_outs_1: data,
                },
                epoch_0_step_0: {...},
                ...
            },

        """
        if key in self.dump_data:
            self.dump_data[key][sub_key] = data
        else:
            self.dump_data[key] = {sub_key: data}
