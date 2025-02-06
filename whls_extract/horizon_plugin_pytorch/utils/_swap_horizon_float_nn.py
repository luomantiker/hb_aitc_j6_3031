import torch

from .model_helper import ModelState

_TORCH_NN_TO_HORIZON_NN_MAPPING = {}


def replace_torch_nn_module(torch_nn_class):
    def wrapper(horizon_nn_class):
        assert hasattr(horizon_nn_class, "from_torch")
        _TORCH_NN_TO_HORIZON_NN_MAPPING[torch_nn_class] = horizon_nn_class
        return horizon_nn_class

    return wrapper


def swap_nn_with_horizonnn(model: torch.nn.Module) -> None:
    model_state = ModelState.record(model)

    modules_to_swap = []
    for name, module in model.named_children():
        if type(module) in _TORCH_NN_TO_HORIZON_NN_MAPPING:
            horizon_nn_mod = _TORCH_NN_TO_HORIZON_NN_MAPPING[
                type(module)
            ].from_torch(module)
            horizon_nn_mod = model_state.apply(horizon_nn_mod)
            modules_to_swap.append(
                (
                    name,
                    horizon_nn_mod,
                )
            )
        else:
            swap_nn_with_horizonnn(module)

    for name, horizon_nn_mod in modules_to_swap:
        setattr(model, name, horizon_nn_mod)
