# modules used when debug bc
import logging
from collections import OrderedDict

import torch
from hbdk4.compiler import convert
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.nn.qat import FloatFunctional, QuantStub, SetItem
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import get_qconfig
from horizon_plugin_pytorch.quantization.hbdk4 import export


def get_single_bc_error(mod, args, kwargs, output):
    same_in_output = None
    is_nn_mod = isinstance(mod, torch.nn.Module)

    from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
        DispatchedTensorWrapper,
    )

    args = DispatchedTensorWrapper.unwrap(args)
    kwargs = DispatchedTensorWrapper.unwrap(kwargs)

    if is_nn_mod:
        # model with hooks cannot be deepcopy...
        origin_forward_hooks = mod._forward_hooks
        origin_forward_pre_hooks = mod._forward_pre_hooks
        mod._forward_hooks = OrderedDict()
        mod._forward_pre_hooks = OrderedDict()
        single_op_net = SingleModOpNet(mod, args, kwargs)
        export_args = args
        if isinstance(mod, SetItem):
            export_args = (args[0], args[-1])
    elif any([isinstance(x, QTensor) for x in tree_flatten(output)[0]]):
        single_op_net = SingleFuncOpNet(mod, args, kwargs)
        export_args = args
    else:
        return None

    # must be tensor here to export fake quant
    flatten_export_args, spec = tree_flatten(export_args)
    flatten_export_args = [
        x.as_subclass(torch.Tensor) if isinstance(x, Tensor) else x
        for x in flatten_export_args
    ]
    export_args = tree_unflatten(flatten_export_args, spec)

    # disable log when export
    plugin_logger = logging.getLogger("horizon_plugin_pytorch")
    plugin_logger.setLevel(logging.ERROR)

    single_op_hbir = export(single_op_net, export_args, native_pytree=False)
    int_hbir = convert(single_op_hbir, get_march())

    plugin_logger.setLevel(logging.INFO)

    flatten_export_args = [
        x.cpu() for x in flatten_export_args if isinstance(x, Tensor)
    ]
    same_in_output = {
        "qatbc": single_op_hbir[0](*flatten_export_args),
        "intbc": int_hbir[0](*flatten_export_args),
    }
    if is_nn_mod:
        mod._forward_hooks = origin_forward_hooks
        mod._forward_pre_hooks = origin_forward_pre_hooks
    return same_in_output


class MultiQuantStubs(torch.nn.Module):
    def __init__(self, args):
        super(MultiQuantStubs, self).__init__()
        args = tree_flatten(args)[0]
        quant_list = []
        self.qtensor_index = []
        for i, x in enumerate(args):
            if isinstance(x, QTensor):
                quant_list.append(
                    QuantStub(
                        scale=x.q_scale(),
                        zero_point=x.q_zero_point(),
                        qconfig=get_qconfig(out_dtype=x.dtype),
                    )
                    .eval()
                    .to(x.device)
                )
                self.qtensor_index.append(i)
        if len(quant_list) == 0:
            self.quants = None
        else:
            self.quants = torch.nn.ModuleList(quant_list)

    def forward(self, args):
        if self.quants is None:
            return args
        args, spec = tree_flatten(args)
        rets = []
        i = 0
        for x in args:
            if isinstance(x, Tensor) and x.is_floating_point():
                rets.append(self.quants[i](x))
                i += 1
            else:
                rets.append(x)
        return tree_unflatten(rets, spec)


class SingleFuncOpNet(torch.nn.Module):
    def __init__(self, func, args, kwargs):
        super(SingleFuncOpNet, self).__init__()
        self.quants = MultiQuantStubs(args)
        self.func = func
        self.kwargs = kwargs

    def forward(self, *args):
        args = self.quants(args)
        return self.func(*args, **self.kwargs)


class SingleModOpNet(torch.nn.Module):
    def __init__(self, mod, args, kwargs):
        super(SingleModOpNet, self).__init__()
        self.mod = mod
        self.quants = MultiQuantStubs(args)
        if isinstance(mod, SetItem):
            self.indices = args[1]
        self.kwargs = kwargs

    def forward(self, *args):
        if not isinstance(self.mod, QuantStub):
            args = self.quants(args)
        if hasattr(self.mod, "_swap_inputs") and self.mod._swap_inputs:
            args = (args[1], args[0])
        if isinstance(self.mod, FloatFunctional):
            return getattr(self.mod, self.mod._last_called_method_name)(
                *args, **self.kwargs
            )
        elif isinstance(self.mod, SetItem):
            return self.mod(args[0], self.indices, args[1])
        return self.mod(*args, **self.kwargs)
