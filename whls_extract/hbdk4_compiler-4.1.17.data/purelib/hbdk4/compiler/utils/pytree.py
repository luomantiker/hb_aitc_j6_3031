from typing import Optional, Any
from pickle import dumps, loads
from base64 import b32encode, b32decode


def pickle_object(obj) -> Optional[str]:
    if obj is not None:
        return str(b32encode(dumps(obj)), encoding="utf-8")
    else:
        return None


def unpickle_object(d_str: str):
    if d_str is not None and len(d_str) > 0:
        return loads(b32decode(bytes(d_str, encoding="utf-8")))
    else:
        return None


class TreeLikeFuncBase:
    @property
    def _in_tree_spec(self):
        raise ValueError("this method should be override")

    @property
    def _out_tree_spec(self):
        raise ValueError("this method should be override")

    @property
    def support_pytree(self) -> bool:
        return self._in_tree_spec is not None and self._out_tree_spec is not None

    @property
    def flatten_inputs(self):
        raise ValueError("this method should be override")

    @property
    def flatten_outputs(self):
        raise ValueError("this method should be override")

    @property
    def inputs(self):
        if self.support_pytree:
            from torch.utils._pytree import tree_unflatten

            inputs_tree = tree_unflatten(self.flatten_inputs, self._in_tree_spec)
            return inputs_tree
        else:
            return self.flatten_inputs

    @property
    def outputs(self):
        if self.support_pytree:
            from torch.utils._pytree import tree_unflatten

            outputs_tree = tree_unflatten(self.flatten_outputs, self._out_tree_spec)
            return outputs_tree
        else:
            return self.flatten_outputs
