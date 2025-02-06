import logging
from copy import deepcopy
from typing import Callable, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.nn.qat import ConvBN2d
from horizon_plugin_pytorch.qat_mode import QATMode, get_qat_mode
from horizon_plugin_pytorch.utils.misc import to_device
from .adaround_fake_quantize import AdaRoundFakeQuantize, FakeQuantize

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)
logger = logging.getLogger(__name__)


class StopForwardError(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


def replace_adaround_fake_quant(module: torch.nn.Module) -> FakeQuantize:
    adaround_fake_quant = AdaRoundFakeQuantize(
        module.weight_fake_quant, module.weight
    )
    module.orig_weight_fake_quant = module.weight_fake_quant
    module.weight_fake_quant = adaround_fake_quant

    if module.activation_post_process is not None:
        # Make FakeQuantize in training mode to support autograd
        module.activation_post_process.train()
        # Disable observer to avoid scale update
        module.orig_observer_state = getattr(
            module.activation_post_process, "_observer_enabled", False
        )
        module.activation_post_process.disable_observer()


def restore_original_fake_quant(module: torch.nn.Module) -> FakeQuantize:
    module.weight_fake_quant = module.orig_weight_fake_quant
    del module.orig_weight_fake_quant
    del module.weight_fake_quant.alpha
    # Restore FakeQuantize state
    if module.activation_post_process is not None:
        module.activation_post_process.eval()
        if module.orig_observer_state:
            module.activation_post_process.enable_observer()
        del module.orig_observer_state


class DataSaverHook:
    """Forward hook that stores the input and output of a submodule/block."""

    def __init__(
        self,
        store_input=False,
        store_output=False,
        stop_forward=False,
    ):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_stored = None
        self.output_stored = None

    def __call__(self, module, input, output):
        if self.store_input:
            self.input_stored = input
        if self.store_output:
            self.output_stored = output
        if self.stop_forward:
            raise StopForwardError


def l2_loss(pred, tgt):
    """Loss function measured in L_p Norm."""
    return (pred - tgt).pow(2).sum(1).mean()


class LinearTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(
                0.0, (1 - rel_t)
            )


class LossFunction:
    """Loss function used in weight reconstrucion.

    Calculate mse reconstruction loss and relaxation loss use some tempdecay
    to balance the two losses.
    """

    def __init__(
        self,
        submodule: nn.Module,
        weight: float = 1.0,
        num_steps: int = 10000,
        b_range: tuple = (20, 2),
        warm_up: float = 0.0,
        p: float = 2.0,
    ):
        self.submodule = submodule
        self.weight = weight
        self.loss_start = num_steps * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(
            num_steps, warm_up=warm_up, start_b=b_range[0], end_b=b_range[1]
        )
        self.step = 0
        self.log_interval = max(num_steps // 10, 1)

    def __call__(self, pred, tgt):
        """Compute the total loss for adaptive rounding.

        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy.
        """
        self.step += 1
        rec_loss = l2_loss(pred, tgt)
        b = self.temp_decay(self.step)
        if self.step < self.loss_start:
            round_loss = 0
        else:
            round_vals = self.submodule.weight_fake_quant.rectified_sigmoid()
            round_loss = (
                self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
            )

        total_loss = rec_loss + round_loss
        if self.step % self.log_interval == 0:
            logger.info(
                "Total loss:\t{:.2f} (rec:{:.4f}, "
                "round:{:.2f})\tb={:.2f}\tstep={}".format(
                    float(total_loss),
                    float(rec_loss),
                    float(round_loss),
                    b,
                    self.step,
                )
            )
        return total_loss


def submodule_reconstruction(submodule, quant_inputs, float_outputs, config):
    w_opt, w_scheduler = None, None
    replace_adaround_fake_quant(submodule)
    adaround_fake_quant = submodule.weight_fake_quant
    w_para = [adaround_fake_quant.alpha]
    loss_func = LossFunction(
        submodule=submodule,
        weight=config.weight,
        num_steps=config.num_steps,
        b_range=config.b_range,
        warm_up=config.warm_up,
    )

    # quant inps: batch x args x data
    num_batches = len(quant_inputs)

    w_opt = torch.optim.Adam(w_para)

    """start training"""

    if dist.is_initialized():
        world_size = dist.get_world_size()
        logger.info("The world size is {}.".format(world_size))
    else:
        world_size = 1

    for _ in range(config.num_steps):
        idx = np.random.randint(0, num_batches)
        sampled_quant_inputs = quant_inputs[idx]
        sampled_float_output = float_outputs[idx]
        w_opt.zero_grad()
        quant_output = submodule(*sampled_quant_inputs)
        err = loss_func(
            quant_output.as_subclass(torch.Tensor),
            sampled_float_output.as_subclass(torch.Tensor),
        )
        err /= world_size
        err.backward()
        if world_size > 1:
            for param in w_para:
                dist.all_reduce(param.grad)
        w_opt.step()
        if w_scheduler:
            w_scheduler.step()
    torch.cuda.empty_cache()
    # We need to do bn fold simulation for WithBNReverseFold.
    if (
        get_qat_mode() == QATMode.WithBNReverseFold
        and isinstance(submodule, ConvBN2d)
        and hasattr(submodule, "bn")
        and isinstance(submodule.bn, torch.nn.modules.batchnorm._BatchNorm)
    ):
        fused_weight, scale_factor = submodule._fuse_bn_weight()
        merged_rounded_weight = adaround_fake_quant.get_hard_value(
            fused_weight
        )

        submodule.weight.data = merged_rounded_weight / scale_factor.reshape(
            [-1] + [1] * (len(merged_rounded_weight.shape) - 1)
        )
    else:
        submodule.weight.data = adaround_fake_quant.get_hard_value(
            submodule.weight
        )
    restore_original_fake_quant(submodule)


class Config:
    def __init__(
        self,
        num_batches=10,
        num_steps=100,
        exclude_prefix=(),
        warm_up=0.2,
        weight=0.01,
        b_range=(20, 2),
        **kwargs,
    ) -> None:
        self.num_batches = num_batches
        self.num_steps = num_steps
        self.exclude_prefix = exclude_prefix
        self.warm_up = warm_up
        self.weight = weight
        self.b_range = b_range
        if kwargs:
            raise KeyError(
                f"Unrecognized config keys: {kwargs}! "
                "Please check your custom_config_dict."
            )


def weight_reconstruction(
    calib_model: torch.nn.Module,
    batches: Union[list, tuple, DataLoader],
    batch_process_func: Callable = None,
    custom_config_dict: dict = None,
):
    r"""Reconstruct model weights using adaround.

    Args:
        `calib_model`:
            A prepared and calibrated model.
        `batches`:
            Calibration data. Can be list, tuple, or
            torch.utils.data.DataLoader. Using list or tuple is more effective
            in performance because all batches will be preloaded on gpu, which
            also causes more gpu memory occupation. If you have limited gpu
            memory resource, you could use DataLoader instead. It loads the
            batch only when it is needed so it's more gpu memory friendly.
        `batch_process_func`:
            Custom batch process function. Will be applied to the batch before
            it is fed to the model. Usefull when you use torch DataLoader to
            feed data and it has complex return values, such as
            tuple(image, label) or dict("img":image, "label":label).
        `custom_config_dict`:
            Dictionary for custom configurations. If not set, use the default
            value:

            .. code-block:: python

                custom_config_dict = {
                    # number of batches to do adaround
                    # greater value costs greater gpu memory occupation
                    # only valid when batches is Dataloader
                    "num_batches": 10,

                    # optimization iteration
                    "num_steps": 100,

                    # module with these prefix will not perform adaround
                    "exclude_prefix": [],

                    # 0.2 * num_steps iters without regularization to
                    # floor or ceil
                    "warm_up": 0.2,

                    # loss weight for regularization item
                    "weight": 0.01,

                    # beta decaying range
                    "b_range": [20, 2],

                }

    """

    config = Config(**custom_config_dict) if custom_config_dict else Config()
    logger.info(f"weight reconstruction config: {config.__dict__}")
    device = next(calib_model.parameters()).device
    if isinstance(batches, (list, tuple)):
        batches = to_device(batches, device)

    def direct_pass(x):
        return x

    if batch_process_func is None:
        batch_process_func = direct_pass

    float_model = deepcopy(calib_model)
    float_model.eval()
    horizon.quantization.set_fake_quantize(
        float_model, horizon.quantization.FakeQuantState._FLOAT
    )
    calib_model.eval()
    horizon.quantization.set_fake_quantize(
        calib_model, horizon.quantization.FakeQuantState.VALIDATION
    )
    ordered_modules = []
    module_handles = []

    def module_order_hook(module, input, output):
        ordered_modules.append(module)

    for name, module in calib_model.named_modules():
        if (
            isinstance(module, _ADAROUND_SUPPORT_TYPE)
            and hasattr(module, "weight_fake_quant")
            and isinstance(module.weight_fake_quant, FakeQuantize)
        ):
            skip = False
            for prefix in config.exclude_prefix:
                if name.startswith(prefix):
                    logger.info(
                        f"Skip reconstruction of {name} "
                        "as set by exclude_prefix."
                    )
                    skip = True
                    break
            if not skip:
                module_handles.append(
                    module.register_forward_hook(module_order_hook)
                )
    name2float_module = dict(float_model.named_modules())

    with torch.no_grad():
        calib_model(batch_process_func(to_device(next(iter(batches)), device)))

    for module_handle in module_handles:
        module_handle.remove()

    logger.info(
        f"Total numbers of modules to reconstruct: {len(ordered_modules)}."
    )

    quant_module2name = {
        module: name for name, module in calib_model.named_modules()
    }
    for module in ordered_modules:
        name = quant_module2name[module]
        logger.info(f"Reconstruting {name} of type {type(module)}")
        if ordered_modules.count(module) > 1:
            logger.info(
                f"Detected shared op {name}. " "it will not be optimized."
            )
            continue

        input_saver = DataSaverHook(
            store_input=True,
            store_output=False,
            stop_forward=True,
        )
        output_saver = DataSaverHook(
            store_input=False,
            store_output=True,
            stop_forward=True,
        )
        input_saver_handle = module.register_forward_hook(input_saver)
        output_saver_handle = name2float_module[name].register_forward_hook(
            output_saver
        )
        quant_inputs, float_outputs = [], []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                if isinstance(batches, DataLoader) and i >= config.num_batches:
                    break
                batch = batch_process_func(to_device(batch, device))

                try:
                    _ = calib_model(batch)
                except StopForwardError:
                    pass
                quant_inputs.append(input_saver.input_stored)

                try:
                    _ = float_model(batch)
                except StopForwardError:
                    pass
                float_outputs.append(output_saver.output_stored)

        input_saver_handle.remove()
        output_saver_handle.remove()
        torch.cuda.empty_cache()
        submodule_reconstruction(module, quant_inputs, float_outputs, config)
