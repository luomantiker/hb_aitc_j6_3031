import copy
from typing import Callable, Optional, Union

from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)


@replace_torch_nn_module(nn.TransformerDecoderLayer)
class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        from horizon_plugin_pytorch.nn.quantized import FloatFunctional

        self.add_0 = FloatFunctional()
        self.add_1 = FloatFunctional()
        self.add_2 = FloatFunctional()

        # replace self.activation with module
        if not isinstance(self.activation, nn.Module):
            act_name = getattr(self.activation, "__name__", None)
            if act_name is not None:
                if act_name == "relu":
                    self.activation = nn.ReLU()
                else:
                    from horizon_plugin_pytorch.fx.fx_helper import (
                        _torch_horizon_nn_op_mapping,
                        _torch_horizon_op_mapping,
                    )

                    if act_name in _torch_horizon_nn_op_mapping:
                        self.activation = _torch_horizon_nn_op_mapping[
                            act_name
                        ]()
                    elif act_name in _torch_horizon_op_mapping:
                        self.activation = _torch_horizon_op_mapping[act_name]()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        assert (
            not tgt_is_causal
        ), "tgt_is_causal is not supported by {}".format(
            self.__class__.__name__
        )
        assert (
            not memory_is_causal
        ), "memory_is_causal is not supported by {}".format(
            self.__class__.__name__
        )

        x = tgt
        if self.norm_first:
            x = self.add_0.add(
                x,
                self._sa_block(
                    self.norm1(x),
                    tgt_mask,
                    tgt_key_padding_mask,
                ),
            )
            x = self.add_1.add(
                x,
                self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                ),
            )
            x = self.add_2.add(x, self._ff_block(self.norm3(x)))
        else:
            x = self.norm1(
                self.add_0.add(
                    x,
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                )
            )
            x = self.norm2(
                self.add_1.add(
                    x,
                    self._mha_block(
                        x,
                        memory,
                        memory_mask,
                        memory_key_padding_mask,
                    ),
                )
            )
            x = self.norm3(self.add_2.add(x, self._ff_block(x)))

        return x

    @classmethod
    def from_torch(cls, mod: nn.TransformerDecoderLayer):
        new_mod = cls(
            d_model=mod.self_attn.embed_dim,
            nhead=mod.self_attn.num_heads,
            dim_feedforward=mod.linear1.out_features,
            dropout=mod.dropout.p,
            activation=mod.activation,
            layer_norm_eps=mod.norm1.eps,
            batch_first=mod.self_attn.batch_first,
            norm_first=mod.norm_first,
        )
        for name, _ in new_mod.named_children():
            if name != "activation" and hasattr(mod, name):
                setattr(new_mod, name, mod.get_submodule(name))
        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod

    # Attr 'activation' becomes one normal attr and another named_modules item
    # unexpectly. Add custom deepcopy to avoid this.
    def __deepcopy__(self, memo):
        cls = self.__class__
        new_mod = cls.__new__(cls)
        memo[id(self)] = new_mod
        for k, v in self.__dict__.items():
            setattr(new_mod, k, copy.deepcopy(v, memo))
        return new_mod
