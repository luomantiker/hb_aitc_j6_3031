import logging
import tempfile

import torch
from torch import Graph, Node, ScriptModule

from horizon_plugin_pytorch.utils.serialization import load, save  # noqa F401

__all__ = ["save", "load", "to_device"]


_ignored_to_device_files = {
    "/nn/quantized/table_generator.py",
    "/nn/quantized/segment_lut.py",
}

logger = logging.getLogger(__name__)


def _should_ignore_to_device(node: Node, ignored_to_device_files=None):
    if ignored_to_device_files is None:
        ignored_to_device_files = _ignored_to_device_files

    parts = str(node).split(__name__.split(".")[0])[-1].split(":")

    # Check if node's callstack is valid
    # <file>:<line_number>:<position>
    if len(parts) != 3:
        return False

    file_path, _, _ = parts
    return file_path in _ignored_to_device_files


def _move_graph_device(graph: Graph, device: torch.device):
    for n in graph.findAllNodes("prim::Constant"):
        repr = str(n)
        if 'value="cpu' in repr or 'value="cuda' in repr:
            logger.debug("Deal with Node: {}".format(n))
            if not _should_ignore_to_device(n):
                new_n = graph.insertConstant(device).node()
                new_n.moveBefore(n)
                n.replaceAllUsesWith(new_n)
                n.destroy()
                logger.debug("Replaced with {}".format(new_n))
            else:
                logger.debug("Skipped")


def to_device(model: ScriptModule, device: torch.device) -> ScriptModule:
    """Move a ScriptModule to the target device, include Tensor generations.

    Args:
        model: Input ScriptModule.
        device: Target device.

    Returns:
        A new ScriptModule which carrys out Tensor generation
        on the target device.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # move the buffer
    model.to(device)

    # modify the graph of each module
    for name, mod in model.named_modules():
        # Unused mod do not have graph
        try:
            mod.graph
        except RuntimeError:
            continue
        logger.debug(
            "=" * 70
            + "\nModifying mod {}\nGraph before:\n{}".format(name, mod.graph)
        )
        _move_graph_device(mod.graph, device)
        logger.debug("-" * 70 + "\nGraph after:\n{}".format(mod.graph))

    # save and load to make the changes to graph take effect
    with tempfile.TemporaryFile() as f:
        torch.jit.save(model, f)
        f.seek(0)
        model = torch.jit.load(f)

    return model
