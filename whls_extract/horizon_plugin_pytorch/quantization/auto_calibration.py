import logging
import random
from functools import partial
from typing import Callable, Union

import torch
from torch.quantization import FakeQuantizeBase
from torch.utils.data import DataLoader

import horizon_plugin_pytorch as horizon
from horizon_plugin_pytorch.quantization.observer_v2 import (
    ClipObserver,
    FixedScaleObserver,
    KLObserver,
    MinMaxObserver,
    MSEObserver,
    PercentileObserver,
    l2_loss,
)
from horizon_plugin_pytorch.utils.misc import to_device

logger = logging.getLogger(__name__)


class RandomSeed(object):
    def __init__(self, seed=None):
        self._seed = seed

    def __enter__(self):
        if self._seed is not None:
            random.seed(self._seed)
            torch.manual_seed(self._seed)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._seed is not None:
            random.seed(torch.Generator().seed())
            torch.manual_seed(torch.Generator().seed())


class StopForwardError(Exception):
    pass


class StopForwardHook:
    def __init__(
        self,
        expected_called_times=1,
    ):
        self.expected_called_times = expected_called_times
        self.called_times = 0

    def __call__(self, module, input, output):
        self.called_times += 1
        if self.called_times == self.expected_called_times:
            self.called_times = 0
            raise StopForwardError


def consistent_forward(
    calib_model, batches, num_batches, batch_process_func, device
):
    """Forward with consistent batches and early stop."""

    with RandomSeed(0):
        for i, batch in enumerate(batches):
            if isinstance(batches, DataLoader) and i >= num_batches:
                break
            with torch.no_grad():
                try:
                    calib_model(batch_process_func(to_device(batch, device)))
                except StopForwardError:
                    pass


def auto_calibrate(
    calib_model: torch.nn.Module,
    batches: Union[list, tuple, DataLoader],
    num_batches: int = 10,
    batch_process_func: Callable = None,
    observer_list: list = ("percentile", "mse", "kl", "min_max"),
    percentile_list: list = None,
):
    """Auto search the best observer and parameter for the given model.

    Args:
        `calib_model`:
            Prepared calibration model.
        `batches`:
            Calibration data. Can be list, tuple, or
            torch.utils.data.DataLoader. Using list or tuple is more effective
            in performance because all batches will be preloaded on gpu, which
            also causes more gpu memory occupation. If you have limited gpu
            memory resource, you could use DataLoader instead. It loads the
            batch only when it is needed so it's more gpu memory friendly.
        `num_batches`:
            Number of batches to do auto calibration. Greater value costs
            greater computation and gpu memory occupation.
            Only valid when `batches` is DataLoader.
        `batch_process_func`:
            Custom batch process function. Will be applied to the batch before
            it is fed to the model. Usefull when you use torch DataLoader to
            feed data and it has complex return values, such as
            tuple(image, label) or dict("img":image, "label":label).
        `observer_list`:
            A list of activation observer candidates. Can only be
            percentile, mse, kl, min_max.
        `percentile_list`:
            A list of percentile candidates, each ranging
            (0, 100]. Only valid when percentile is in observer_list.
            If the list is empty, will search default percentile 99.995.
    """
    assert observer_list, "Please set at least one observer!"
    observer_dict = {
        "percentile": PercentileObserver,
        "mse": MSEObserver,
        "kl": KLObserver,
        "min_max": MinMaxObserver,
    }
    for observer_name in observer_list:
        assert (
            observer_name in observer_dict
        ), f"unsupport observer :{observer_name}!"
    if "percentile" in observer_list and not percentile_list:
        percentile_list = [99.995]

    assert batches, "batches is empty!"
    device = next(calib_model.parameters()).device
    if isinstance(batches, (list, tuple)):
        batches = to_device(batches, device)

    def direct_pass(x):
        return x

    if batch_process_func is None:
        batch_process_func = direct_pass

    calib_model.eval()
    horizon.quantization.set_fake_quantize(
        calib_model, horizon.quantization.FakeQuantState.CALIBRATION
    )

    ordered_modules, float_outputs, quant_outputs = [], [], []
    module_handles, float_handles = [], []

    def module_order_hook(module, input, output):
        ordered_modules.append(module)

    def float_saver_hook(module, input, output):
        float_outputs.append(output)

    def quant_saver_hook(module, input, output):
        quant_outputs.append(output)

    for module in calib_model.modules():
        module_handles.append(module.register_forward_hook(module_order_hook))

    # 记录 module 顺序，并且初始化 calib model 中的 scale
    with torch.no_grad():
        calib_model(batch_process_func(to_device(next(iter(batches)), device)))

    logger.debug(f"ordered modules length: {len(ordered_modules)}.")

    for module_handle in module_handles:
        module_handle.remove()

    calib_model.eval()
    horizon.quantization.set_fake_quantize(
        calib_model, horizon.quantization.FakeQuantState._FLOAT
    )

    for module in calib_model.modules():
        if isinstance(module, torch.quantization.DeQuantStub):
            float_handles.append(
                module.register_forward_hook(float_saver_hook)
            )

    # 采样浮点模型的输出
    consistent_forward(
        calib_model, batches, num_batches, batch_process_func, device
    )

    for handle in float_handles:
        handle.remove()

    module2name = {
        module: name for name, module in calib_model.named_modules()
    }
    searched_module = set()
    for module in ordered_modules:
        name = module2name[module]
        if module in searched_module:
            logger.info(
                f"detected shared op {name}. "
                "it will only be searched for the first time call."
            )
            continue

        if hasattr(module, "weight_fake_quant") and isinstance(
            module.weight_fake_quant, FakeQuantizeBase
        ):
            module.weight_fake_quant.enable_fake_quant()
            logger.debug(f"enable weight fake quant of {name}.")

        if (
            hasattr(module, "activation_post_process")
            and isinstance(module.activation_post_process, FakeQuantizeBase)
            and not isinstance(
                module.activation_post_process.activation_post_process,
                (FixedScaleObserver, ClipObserver),
            )
            and module.activation_post_process._observer_enabled
        ):
            logger.debug(
                f"enable activation fake quant of {name} and search observers."
            )
            min_loss = float("inf")
            optimal_percentile = None
            optimal_observer = None
            optimal_fake_quantizer = None

            for observer_name in observer_list:
                observer = observer_dict.get(observer_name)
                activation = partial(
                    module.qconfig.activation,
                    observer=observer,
                    averaging_constant=1 / len(batches),
                )
                if observer == KLObserver:
                    activation = partial(
                        activation, update_interval=len(batches)
                    )
                num_iters = (
                    len(percentile_list)
                    if observer == PercentileObserver
                    else 1
                )
                for p in range(num_iters):
                    if observer == PercentileObserver:
                        activation = partial(
                            activation, percentile=percentile_list[p]
                        )

                    fake_quantizer = (
                        module.activation_post_process
                    ) = activation().to(device)

                    fake_quantizer.enable_observer()
                    fake_quantizer.enable_fake_quant()
                    fake_quantizer.train()

                    # 当目标 module 完成 forward 后，提前终止模型 forward，
                    # 避免无效计算，提高性能
                    stop_forward_hook = StopForwardHook(
                        expected_called_times=ordered_modules.count(module)
                    )

                    stop_handle = fake_quantizer.register_forward_hook(
                        stop_forward_hook
                    )

                    # 计算 scale
                    consistent_forward(
                        calib_model,
                        batches,
                        num_batches,
                        batch_process_func,
                        device,
                    )

                    stop_handle.remove()
                    fake_quantizer.eval()

                    # 采样当前 scale 下量化模型的输出
                    quant_handles = []
                    for m in calib_model.modules():
                        if isinstance(m, torch.quantization.DeQuantStub):
                            quant_handles.append(
                                m.register_forward_hook(quant_saver_hook)
                            )

                    quant_outputs = []
                    consistent_forward(
                        calib_model,
                        batches,
                        num_batches,
                        batch_process_func,
                        device,
                    )

                    for handle in quant_handles:
                        handle.remove()

                    # 计算浮点输出和量化输出的 L2 距离
                    # 对不同 batch、不同输出的距离做求和处理
                    assert len(quant_outputs) == len(float_outputs)
                    cur_loss = sum(
                        [
                            l2_loss(
                                quant_output.flatten(), float_output.flatten()
                            ).item()
                            for (quant_output, float_output) in zip(
                                quant_outputs,
                                float_outputs,
                            )
                        ]
                    )
                    logger.debug(
                        f"{observer_name} scale: "
                        f"{fake_quantizer.scale.item()}, loss: {cur_loss}"
                    )
                    if cur_loss < min_loss:
                        min_loss = cur_loss

                        optimal_observer = observer_name
                        optimal_fake_quantizer = fake_quantizer
                        if observer == PercentileObserver:
                            optimal_percentile = percentile_list[p]

            fake_quantizer = (
                module.activation_post_process
            ) = optimal_fake_quantizer
            if optimal_observer == "percentile":
                logger.info(
                    f"layer {name}, "
                    f"optimal observer is percentile {optimal_percentile}, "
                    f"scale is {fake_quantizer.scale.item():.4g}"
                )
            else:
                logger.info(
                    f"layer {name}, "
                    f"optimal observer is {optimal_observer}, "
                    f"scale is {fake_quantizer.scale.item():.4g}"
                )
            searched_module.add(module)
            # 结束量化该层后，需要手动设为 eval，固定 scale
            fake_quantizer.eval()
