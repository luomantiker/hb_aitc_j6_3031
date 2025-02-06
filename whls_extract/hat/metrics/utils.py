from typing import List, Union

import torch

__all__ = ["cat_tensor_to_numpy"]


def cat_tensor_to_numpy(value: Union[torch.Tensor, List[torch.Tensor]]):
    """Cat a list of Tensor to one, and convert it to numpy."""
    if isinstance(value, list):
        value = [y.unsqueeze(0) if y.ndim == 0 else y for y in value]
        if len(value) == 0:
            value = torch.tensor([])
        else:
            value = torch.cat(value)
    return value.cpu().numpy()
