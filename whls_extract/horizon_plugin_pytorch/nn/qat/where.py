import torch
from torch import Tensor

from ..where import Where as FloatWhere
from .qat_meta import QATModuleMeta


class Where(FloatWhere, metaclass=QATModuleMeta, input_num=3):
    def forward(self, condition, input, other):
        if self.activation_pre_process is not None:
            if isinstance(input, Tensor):
                input = self.activation_pre_process[1](input)
            if isinstance(other, Tensor):
                other = self.activation_pre_process[2](other)

        output = torch.where(
            condition, input.as_subclass(Tensor), other.as_subclass(Tensor)
        )

        if self.activation_post_process is None:
            return output
        else:
            return self.activation_post_process(output)
