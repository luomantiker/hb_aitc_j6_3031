import torch
from packaging import version


def node_get(node: torch._C.Node, key: str):
    def torch20_impl(node: torch._C.Node, key: str):
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def torch13_impl(node: torch._C.Node, key: str):
        return node[key]

    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        return torch20_impl(node, key)
    else:
        return torch13_impl(node, key)


def get_unique_devices(module):
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        from torch.quantization.quantize import _get_unique_devices_

        return _get_unique_devices_(module)
    else:
        from torch.quantization.quantize import get_unique_devices_

        return get_unique_devices_(module)
