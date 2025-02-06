import copy
from inspect import ismethod
from typing import Any, Dict, List, Set, Union

import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph


def _copy_attr(src, dist, attr_names):
    for attr in attr_names:
        if hasattr(src, attr):
            if ismethod(getattr(src, attr)):
                break  # do not copy method because we use inherit
            else:
                setattr(dist, attr, getattr(src, attr))


class GraphModuleWithAttr(GraphModule):
    def __new__(
        cls,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: List[str] = None,
    ):
        if isinstance(root, GraphModuleWithAttr):
            # use origin root to prevent nested inherit
            root = root.root

        for t in cls.__mro__:
            c = t.__qualname__.split(".")[-1]
            if c != "GraphModuleWithAttrImpl":
                cls = t
                break

        class GraphModuleWithAttrImpl(cls, root.__class__):
            pass

        return object.__new__(GraphModuleWithAttrImpl)

    def __init__(
        self,
        root: torch.nn.Module,
        graph: Graph,
        preserved_attr_names: List[str] = None,
    ):
        if preserved_attr_names is None:
            preserved_attr_names = set()
        else:
            preserved_attr_names = set(preserved_attr_names)

        if isinstance(root, GraphModuleWithAttr):
            # use origin root to prevent nested inherit
            # copy root attr to root.root to preserve attrs
            # added after getting GraphModuleWithAttr.
            _copy_attr(
                root,
                root.root,
                preserved_attr_names,
            )
            root = root.root
        assert isinstance(root, torch.nn.Module)

        torch.nn.Module.__init__(self)

        self.__class__.__name__ = root.__class__.__name__
        # use object.__setattr__ because root should not be a submodule
        object.__setattr__(self, "root", root)
        # used to support self.method
        object.__setattr__(self, "_self", self)

        # Store the Tracer class responsible for creating a Graph separately
        # as part of the GraphModule state, except when the Tracer is defined
        # in a local namespace. Locally defined Tracers are not pickleable.
        # This is needed because torch.package will serialize a GraphModule
        # without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls = None
        if (
            hasattr(graph, "_tracer_cls")
            and graph._tracer_cls
            and "<locals>" not in graph._tracer_cls.__qualname__
        ):
            self._tracer_cls = graph._tracer_cls

        _copy_attr(
            root,
            self,
            preserved_attr_names.union(set(root.__dict__.keys())),
        )

        # must set self.graph after all submodule registered, or bool(self)
        # will return False when self is a Sequential and cause unexpeced error
        self.graph = graph
        self.preserved_attr_names = preserved_attr_names

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module. So we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = copy.deepcopy(self.root)

        new_obj = self.__class__.__mro__[1](
            fake_mod,
            copy.deepcopy(self.graph),
            copy.deepcopy(self.preserved_attr_names),
        )

        # We must manually set the training state
        # because new_obj.training is the same with root, not self!
        new_obj.train(self.training)

        return new_obj


class FusedGraphModule(GraphModuleWithAttr):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str] = None,
    ):
        super(FusedGraphModule, self).__init__(
            root, graph, preserved_attr_names
        )


class ObservedGraphModule(GraphModuleWithAttr):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str] = None,
    ):
        super(ObservedGraphModule, self).__init__(
            root,
            graph,
            preserved_attr_names,
        )


def is_observed_module(module: Any) -> bool:
    return isinstance(module, ObservedGraphModule)


class QuantizedGraphModule(GraphModuleWithAttr):
    """Refine this docstring in the future.

    This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """

    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str] = None,
    ):
        super(QuantizedGraphModule, self).__init__(
            root, graph, preserved_attr_names
        )
