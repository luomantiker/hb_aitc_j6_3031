import math
from itertools import chain
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)


@replace_torch_nn_module(nn.GRUCell)
class GRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(
            input_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype
        )
        self.h2h = nn.Linear(
            hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype
        )

        self.add_r = FloatFunctional()
        self.sigmoid_r = nn.Sigmoid()
        self.add_i = FloatFunctional()
        self.sigmoid_i = nn.Sigmoid()
        self.reset_mul = FloatFunctional()
        self.reset_add = FloatFunctional()
        self.tanh = nn.Tanh()
        self.out_sub = FloatFunctional()
        self.out_mul = FloatFunctional()
        self.out_add = FloatFunctional()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ):
        if hx is None:
            hx = torch.zeros(
                *input.shape[:-1],
                self.hidden_size,
                dtype=input.as_subclass(torch.Tensor).dtype,
                device=input.device
            )
            if hasattr(self.out_add, "activation_post_process"):
                is_enabled = (
                    self.out_add.activation_post_process._observer_enabled
                )
                self.out_add.activation_post_process._observer_enabled = False
                hx = self.out_add.activation_post_process(hx)
                self.out_add.activation_post_process._observer_enabled = (
                    is_enabled
                )

        gate_x = self.x2h(input)
        gate_h = self.h2h(hx)

        i_r, i_i, i_n = gate_x.chunk(3, -1)
        h_r, h_i, h_n = gate_h.chunk(3, -1)

        resetgate = self.sigmoid_r(self.add_r.add(i_r, h_r))
        inputgate = self.sigmoid_i(self.add_i.add(i_i, h_i))
        newgate = self.tanh(
            self.reset_add.add(i_n, self.reset_mul.mul(resetgate, h_n))
        )

        hy = self.out_add.add(
            newgate, self.out_mul.mul(inputgate, self.out_sub.sub(hx, newgate))
        )

        return hy

    @classmethod
    def from_torch(cls, mod: nn.GRUCell):
        new_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
        )

        with torch.no_grad():
            new_mod.x2h.weight.copy_(mod.get_parameter("weight_ih"))
            new_mod.h2h.weight.copy_(mod.get_parameter("weight_hh"))
            if mod.bias:
                new_mod.x2h.bias.copy_(mod.get_parameter("bias_ih"))
                new_mod.h2h.bias.copy_(mod.get_parameter("bias_hh"))

        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod


@replace_torch_nn_module(nn.GRU)
class GRU(nn.Module):
    _FLOAT_MODULE = nn.GRU

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if dropout > 0:
            raise ValueError("dropout is unsupported")
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GRUCell(
                    (
                        input_size
                        if i == 0
                        else hidden_size * self.num_directions
                    ),
                    hidden_size,
                    bias,
                    device,
                    dtype,
                )
            )

        self.output_stack = FloatFunctional()
        self.output_hs_stack = FloatFunctional()

        if bidirectional:
            self.reverse_layers = nn.ModuleList()
            self.bidirectional_cats = nn.ModuleList()
            for i in range(num_layers):
                self.reverse_layers.append(
                    GRUCell(
                        (
                            input_size
                            if i == 0
                            else hidden_size * self.num_directions
                        ),
                        hidden_size,
                        bias,
                        device,
                        dtype,
                    )
                )
                self.bidirectional_cats.append(FloatFunctional())

    def forward(
        self,
        input: Union[torch.Tensor, PackedSequence],
        hx: Optional[torch.Tensor] = None,
    ):
        if isinstance(input, PackedSequence):
            raise ValueError("PackedSequence is unsupported")

        batched_input = input.ndim > 2

        hx_shape = [self.num_layers, self.num_directions, self.hidden_size]
        if batched_input:
            batch_size = input.size(0) if self.batch_first else input.size(1)
            hx_shape.insert(2, batch_size)

        if hx is None:
            # Generate hidden state in GRUCell, because
            # their scale is different.
            hs_list = [None] * self.num_layers
            reverse_hs_list = [None] * self.num_layers
        else:
            hx = hx.reshape(hx_shape)
            hs_list = list(hx[:, 0, ...].unbind(0))
            if self.bidirectional:
                reverse_hs_list = list(hx[:, 1, ...].unbind(0))

        if batched_input:
            input_seq = list(input.unbind(1 if self.batch_first else 0))
        else:
            input_seq = list(input.unbind(0))

        if self.bidirectional:
            for i, (layer, reverse_layer) in enumerate(
                zip(self.layers, self.reverse_layers)
            ):
                output_list = []
                reverse_output_list = []
                for x in input_seq:
                    x = layer(x, hs_list[i])
                    hs_list[i] = x
                    output_list.append(x)
                for x in reversed(input_seq):
                    x = reverse_layer(x, reverse_hs_list[i])
                    reverse_hs_list[i] = x
                    reverse_output_list.append(x)
                for j, (ret, reverse_ret) in enumerate(
                    zip(output_list, reversed(reverse_output_list))
                ):
                    input_seq[j] = self.bidirectional_cats[i].cat(
                        (ret, reverse_ret), dim=-1
                    )

            hs_list = tuple(chain.from_iterable(zip(hs_list, reverse_hs_list)))
        else:
            for i, layer in enumerate(self.layers):
                for j, x in enumerate(input_seq):
                    x = layer(x, hs_list[i])
                    hs_list[i] = x
                    input_seq[j] = x
            reverse_hs_list = []

        output = self.output_stack.stack(
            input_seq, dim=1 if batched_input and self.batch_first else 0
        )
        output_hs = self.output_hs_stack.stack(hs_list, dim=0)

        return output, output_hs

    @classmethod
    def from_torch(cls, mod: nn.GRU):
        new_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
        )

        with torch.no_grad():
            for i, layer in enumerate(new_mod.layers):
                layer: GRUCell
                layer.x2h.weight.copy_(
                    mod.get_parameter("weight_ih_l{}".format(i))
                )
                layer.h2h.weight.copy_(
                    mod.get_parameter("weight_hh_l{}".format(i))
                )
                if mod.bias:
                    layer.x2h.bias.copy_(
                        mod.get_parameter("bias_ih_l{}".format(i))
                    )
                    layer.h2h.bias.copy_(
                        mod.get_parameter("bias_hh_l{}".format(i))
                    )
            if mod.bidirectional:
                for i, layer in enumerate(new_mod.reverse_layers):
                    layer: GRUCell
                    layer.x2h.weight.copy_(
                        mod.get_parameter("weight_ih_l{}_reverse".format(i))
                    )
                    layer.h2h.weight.copy_(
                        mod.get_parameter("weight_hh_l{}_reverse".format(i))
                    )
                    if mod.bias:
                        layer.x2h.bias.copy_(
                            mod.get_parameter("bias_ih_l{}_reverse".format(i))
                        )
                        layer.h2h.bias.copy_(
                            mod.get_parameter("bias_hh_l{}_reverse".format(i))
                        )

        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod
