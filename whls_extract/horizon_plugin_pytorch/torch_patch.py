from torch.utils import _pytree
from torch.utils._pytree import _register_pytree_node


def _slice_flatten(d):
    return [d.start, d.stop, d.step], None


def _slice_unflatten(values, context):
    return slice(values[0], values[1], values[2])


# support slice in tree_flatten
_register_pytree_node(slice, _slice_flatten, _slice_unflatten)


def _is_namedtuple_instance(pytree):
    return (
        isinstance(pytree, tuple)
        and hasattr(pytree, "_asdict")
        and hasattr(pytree, "_fields")
    )


_pytree._is_namedtuple_instance = _is_namedtuple_instance
