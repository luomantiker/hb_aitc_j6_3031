import torch

from horizon_plugin_pytorch.qtensor import QTensor, copy_from
from ..setitem import SetItem as FloatSetItem
from .qat_meta import QATModuleMeta


class SetItem(FloatSetItem, metaclass=QATModuleMeta, input_num=3):
    def forward(self, tensor: torch.Tensor, indices, val):
        if isinstance(val, torch.Tensor):
            val = val.as_subclass(torch.Tensor)

        torch.Tensor.__setitem__(
            tensor.as_subclass(torch.Tensor),
            indices,
            val,
        )
        if self.activation_post_process is not None:
            new_tensor = self.activation_post_process(
                tensor.as_subclass(torch.Tensor)
            )
            assert tensor.__class__ == new_tensor.__class__, (
                "The output type of setitem must match its input type "
                "(both float or both quantized)"
            )
            if isinstance(tensor, QTensor):
                # make inplace fake quant, same as Tensor.__setitem__
                copy_from(tensor, new_tensor)
            else:
                tensor.copy_(new_tensor)

        return tensor
