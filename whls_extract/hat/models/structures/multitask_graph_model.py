# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import threading
from collections import ChainMap, OrderedDict, namedtuple
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

try:
    from hatbc.workflow.engine import SymbolExecutor
    from hatbc.workflow.passes import get_split_node_v2, split_by_node_name
    from hatbc.workflow.proxy import (
        OptionalVariable,
        Variable,
        WorkflowVariable,
        get_traced_graph,
    )
    from hatbc.workflow.symbol import Symbol, group
    from hatbc.workflow.trace import GraphTracer
except ImportError:
    SymbolExecutor = None
    get_split_node_v2 = None
    split_by_node_name = None
    OptionalVariable = None
    Variable = None
    WorkflowVariable = None
    get_traced_graph = None
    Symbol = None
    group = None
    GraphTracer = None


from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import (
    _as_list,
    apply_to_collection,
    flatten,
    is_list_of_type,
    regroup,
    to_flat_ordered_dict,
)
from hat.utils.module_patch import TorchModulePatch, merge_symbol_nodes

__all__ = ["MultitaskGraphModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MultitaskGraphModel(nn.Module):
    r"""Graph model used to construct multitask model structure.

    Structures of each task can be declared independently (while some
    modules are actually shared among multiple tasks), each corresponds
    to a separately built computational graph.

    Then, some other modules that take outputs of multiple tasks as
    inputs, named as 'funnel modules', are called to generate final outputs.

    By defining that nodes with the same inputs and shared operator (module)
    are identical, we can conduct a node merge in the multitask graph in a
    layer-by-layer manner (implemented as BFS).

    This class differs from GraphModel primarily in the graph initialization
    stage.

    Args:
        inputs: key-value pairs used to describe task-agnostic inputs. During
            initialization, they are used in tracing, to build the topology
            of the whole computational graph. Generally, keys are strings,
            while values can be tensor or None (for symbolic mode only).
        task_inputs: key-value pairs used to describe task-specific inputs,
            which functions similar as inputs. The difference is, each task
            has its own namespace, so its can be better represented as
            {task_name1: task_inputs1, task_name2: task_inputs2, ...}.
        task_modules: key-value pairs used to describe the model structure
            of each task.
        opt_inputs: key-value pairs used to describe task-agnostic inputs
            that are optional to the whole graph.
        funnel_modules: key-value pairs used to describe "funnel" modules
            that collect outputs from multiple tasks and generate final
            results. Each funnel module corresponds to a key structured as
            (input_names, out_name), which means it "absorbs" (dict pop)
            outputs keyed by input_names and then pushes back its output
            keyed by out_name to the output dict.
        flatten_outputs: whether to flatten final outputs to NamedTuple,
            in order to support tracing.
        lazy_forward: whether to conduct symbolic tracing or not. If contents
            of any outputs of a graph node need expanding (for example,
            query value of a dict with a key), lazy_forward is not available.
        force_cpu_init: force to init model on cpu, mainly to avoid
        Gpu oom when tasks increases.
    """

    # GraphTracer is not threading safe #
    TRACER_LOCK = threading.Lock()

    @classmethod
    def from_existed_graph_and_modules(
        cls,
        graph: Symbol,
        node2name: Dict[nn.Module, str],
        flatten_outputs: bool = True,
        output_names: Optional[List[str]] = None,
        name2inds_fmts: Optional[Dict[str, Tuple[List[int], str]]] = None,
    ):
        empty_mgm = cls({}, {}, {})

        empty_mgm._graph = graph
        empty_mgm._register_nodes(node2name)
        empty_mgm.node2name = node2name
        # no subgraph info
        empty_mgm._output_names = output_names
        tag = (
            ">".join(output_names)
            if isinstance(output_names, Sequence)
            else output_names
        )
        empty_mgm._cached_graphs[tag] = graph
        empty_mgm.flatten_outputs = flatten_outputs
        empty_mgm._name2inds_fmts = name2inds_fmts
        empty_mgm._set_flat_conditions()
        return empty_mgm

    def __init__(
        self,
        inputs: Dict[str, Any],
        task_inputs: Dict[str, Dict[str, Any]],
        task_modules: Dict[str, nn.Module],
        opt_inputs: Optional[Dict[str, Any]] = None,
        funnel_modules: Optional[
            Dict[Tuple[Tuple[str], str], nn.Module]
        ] = None,
        flatten_outputs: bool = True,
        lazy_forward: Optional[bool] = True,
        force_cpu_init: Optional[bool] = False,
        force_eval_init: Optional[bool] = False,
    ):
        super().__init__()
        self._cached_execs = {}

        self.flatten_outputs = flatten_outputs

        task_names = list(task_inputs.keys())

        # use gpu for initialization if available
        if torch.cuda.is_available() and not force_cpu_init:
            rank = torch.cuda.current_device()
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        if funnel_modules is None:
            funnel_modules = {}

        task_modules = {k: v.to(device) for k, v in task_modules.items()}
        funnel_modules = {k: v.to(device) for k, v in funnel_modules.items()}

        opt_inputs = {} if opt_inputs is None else opt_inputs
        inputs_var, task_inputs_var = self._build_variables(
            inputs, opt_inputs, task_inputs, lazy_forward, device=device
        )

        # 1. build: synchronously execute GraphTracer, since it is not
        # threading safe.
        with self.TRACER_LOCK, GraphTracer(imperative=not lazy_forward):
            tmp_outputs = OrderedDict()

            # trace task modules one by one
            for task in task_names:
                _task_module = task_modules[task]
                t_inputs = task_inputs_var[task]
                input_dict = {k: t_inputs[k] for k in t_inputs.keys()}
                input_dict.update(inputs_var)
                if force_eval_init:
                    _task_module.eval()
                tmp_outputs[task] = _task_module(input_dict)

            # filter task outputs with funnel modules one by one
            for (input_names, out_name), module in funnel_modules.items():
                funnel_inputs = [tmp_outputs.pop(name) for name in input_names]
                tmp_outputs[out_name] = module(*funnel_inputs)

            name2fmts, name2nodes = {}, {}
            self._cached_graphs = OrderedDict()

            # get traced graph from each output variable, as well as
            # recording the internal data structure
            for out_name, _outputs in tmp_outputs.items():
                _outputs, _formats = flatten(_outputs)
                out_graph = get_traced_graph(_outputs)
                self._cached_graphs[out_name] = out_graph

                name2fmts[out_name] = _formats
                name2nodes[out_name] = list(out_graph._outputs)

                for obj in _outputs:
                    assert isinstance(obj, WorkflowVariable), (
                        "graph outputs should be one or a sequence of "
                        "`WorkflowVariable` objects, but get %s" % type(obj)
                    )

            self.name2nodes = name2nodes
            self.name2fmts = name2fmts
            self._output_names = list(tmp_outputs.keys())

        del tmp_outputs
        torch.cuda.empty_cache()

        symbols = list(self._cached_graphs.values())

        self._graph = None
        self._name2inds_fmts = OrderedDict()
        if symbols:
            self._graph = group(*symbols)
            merge_map = merge_symbol_nodes(self._graph)

            nodes2indices = {
                node: i for i, node in enumerate(self._graph._outputs)
            }
            for out_name in self._cached_graphs:
                out_nodes = [
                    merge_map.get(node, node) for node in name2nodes[out_name]
                ]
                self._name2inds_fmts[out_name] = (
                    [nodes2indices[node] for node in out_nodes],
                    name2fmts[out_name],
                )

        if TorchModulePatch.current() is None:
            nodes = {}
        else:
            nodes = {
                k: v.obj
                for k, v in TorchModulePatch.current().named_op_instance.items()  # noqa
            }
        self.node2name = {v: k for k, v in nodes.items()}

        self._node_name_2_mod_name = {}

        # 2. register modules by traversing the traced graph
        self._register_nodes(self.node2name)

        # 3. check if all trainable parameters and buffers are covered
        self._check_params_and_buffers(ChainMap(task_modules, funnel_modules))
        self.cpu()

        self._set_flat_conditions()

    # set default flat_conditions
    def _set_flat_conditions(self):
        self._output_names2flat_conditions = {}
        for name in self._output_names:
            self._output_names2flat_conditions[
                name
            ] = lambda _, v: not is_list_of_type(v, torch.Tensor)

    def _register_nodes(self, node2name):
        def _register(node):
            if isinstance(node.op, nn.Module) and node.op in node2name:
                mod_name = node2name[node.op]
                self.add_module(mod_name, node.op)
                self._node_name_2_mod_name[node.name] = mod_name

            if callable(node.op) and not isinstance(node.op, nn.Module):
                # pass nn.Module to a traceable function
                for value in chain(node.args, node.kwargs.values()):
                    if isinstance(value, nn.Module) and value in node2name:
                        mod_name = node2name[value]
                        self.add_module(mod_name, value)
                        self._node_name_2_mod_name[node.name] = mod_name

        if self._graph is not None:
            # register modules by traversing the traced graph
            self._graph.post_order_dfs_visit(fvisit=_register)

    def _check_params_and_buffers(self, module_dict):
        """Check parameters and buffers.

        Make sure all trainable variables in each task module
        are covered in graph model scope
        """

        # 1. collect all parameters and buffers under management
        _params_n_buffers = set()
        for p in self.parameters():
            _params_n_buffers.add(p)
        for b in self.buffers():
            _params_n_buffers.add(b)

        # 2. check
        for t, m in module_dict.items():
            for name, p in m.named_parameters():
                assert (
                    p in _params_n_buffers
                ), f"Parameter {name} of task {t} not covered"
            for name, b in m.named_buffers():
                assert (
                    b in _params_n_buffers
                ), f"Buffer {name} of task {t} not covered"
        return

    @staticmethod
    def _build_variables(
        inputs, opt_inputs, task_inputs, lazy_forward, device=None
    ):
        def _to_device(value, device):
            return apply_to_collection(
                value,
                torch.Tensor,
                lambda x: x.to(device),
            )

        assert not set(inputs.keys()).intersection(
            set(opt_inputs.keys())
        ), "No repeated keys in inputs and opt_inputs allowed"

        inputs_var = {
            k: Variable(k, _to_device(v, device)) for k, v in inputs.items()
        }
        inputs_var.update(
            {
                k: OptionalVariable(k, _to_device(v, device))
                for k, v in opt_inputs.items()
            }
        )

        task_inputs_var = {
            k: Variable(k, _to_device(v, device))
            for k, v in task_inputs.items()
        }

        return inputs_var, task_inputs_var

    def get_sub_graph(self, out_names: Union[str, Sequence[str]]) -> Symbol:
        """Select part of the graph outputs by `out_names` to get sub graph.

        Args:
            out_names: Names of graph outputs, should be a subset of
            `self._output_names` .

        Returns:
            :class:`hatbc.workflow.symbol.Symbol`:
                A sub graph of `self._graph` .
        """
        assert self._cached_graphs, "build graph topology first"
        return group(
            *[self._cached_graphs[name] for name in _as_list(out_names)]
        )

    def forward(
        self,
        inputs: Dict[str, Any],
        out_names: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[NamedTuple, Dict]:
        r"""Forward full or subgraph given output names and input data.

        Args:
            out_names: Graph output names, should be a subset of
                `self._output_names` , i.e. should keep accordance with
                the keys of `name2out` which is returned from
                `self.topology_builder` .

                If None, means to forward the whole graph.

                If not None, we will use it to get a sub graph then forward.

            inputs: A dict of (input name, data), should be a subset of
                `self.inputs` , providing necessary input data to forward the
                full or sub graph.

                .. note::

                    Only provide reliable inputs used in graph forward,
                    extra inputs will cause error.

        """
        # 0. check
        assert self._graph is not None, "init graph first"
        assert isinstance(
            inputs, dict
        ), "MultitaskGraphModel inputs should be a dict but get %s" % type(
            inputs
        )

        if out_names is None:
            out_names = self._output_names
        else:
            out_names = tuple(_as_list(out_names))

            # make sure names don't repeat
            assert len(out_names) and len(set(out_names)) == len(out_names)
            result_idxs = []
            for i, n in enumerate(out_names):
                assert (
                    n in self._output_names
                ), "%s not in output names: %s" % (
                    n,
                    self._output_names,
                )
                result_idxs.append((i, self._output_names.index(n)))

            result_idxs = [
                v[0] for v in sorted(result_idxs, key=lambda x: x[1])
            ]

            # sort names to get cached graph, executor, so that
            # ['name1', 'name2'], ['name2', 'name1'] share the same executor.
            out_names = [out_names[i] for i in result_idxs]

        # 1. make sure it's not in DataParallel mode
        assert not getattr(
            self, "_is_replica", False
        ), "Don't use DataParallel"

        # 2. get executor
        sort_names = out_names
        tag = (
            ">".join(sort_names)
            if isinstance(sort_names, Sequence)
            else sort_names
        )
        if tag not in self._cached_execs:
            if tag not in self._cached_graphs:
                self._cached_graphs[tag] = self.get_sub_graph(sort_names)

            sub_graph = self._cached_graphs[tag]
            self._cached_execs[tag] = SymbolExecutor(sub_graph)

        # TODO: zihan.qiu 2022-11-16: as list turns a -> [a]
        # 3. forward
        results = _as_list(self._cached_execs[tag](inputs))
        # restore the original data structure
        new_results = OrderedDict()
        if len(sort_names) == 1:
            name = sort_names[0]
            _, fmts = self._name2inds_fmts[name]
            grouped, end_idx = regroup(results, fmts)
            if len(results) != end_idx or not isinstance(fmts, Iterable):
                new_results[name] = results
            else:
                new_results[name] = grouped
        else:
            indices_set = set()
            for name in sort_names:
                indices = self._name2inds_fmts[name][0]
                indices_set |= set(indices)

            assert len(indices_set) == len(results)
            all_indices = sorted(indices_set)
            inv_inds = {v: i for i, v in enumerate(all_indices)}

            for name in sort_names:
                indices, fmts = self._name2inds_fmts[name]
                task_results = [results[inv_inds[i]] for i in indices]
                new_results[name] = regroup(task_results, fmts)[0]

        results = new_results

        assert len(results) == len(sort_names), "%d vs. %d" % (
            len(results),
            len(sort_names),
        )
        if self.flatten_outputs:
            # 3. reorder results in order of out_names
            outs = []
            for name, res in results.items():

                name2out = to_flat_ordered_dict(
                    res,
                    key_prefix=name,
                    flat_condition=self._output_names2flat_conditions[name],
                )

                for k in name2out:
                    if isinstance(name2out[k], list):
                        name2out[k] = tuple(name2out[k])

                # torch.jit.trace() recommend us to convert dict to namedtuple
                try:
                    OrderedOutput = namedtuple(
                        "OrderedOutput", name2out.keys()
                    )
                except SyntaxError as e:
                    logger.error("name2out keys: {name2out.keys()}")
                    raise e
                outs.append(OrderedOutput(**name2out))

            return tuple(outs)
        else:
            return results

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def set_calibration_qconfig(self):
        import horizon_plugin_pytorch as horizon

        self.qconfig = horizon.quantization.get_default_calib_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_calibration_qconfig"):
                    module.set_calibration_qconfig()

    @property
    def output_names(self):
        """Names of graph output variables."""
        return self._output_names

    @property
    def graph(self):
        """Full graph which represents GraphModel's computational topology."""  # noqa: E501
        return self._graph

    def _named_modules_by_outname(
        self, out_names: Tuple[str], iter_fn: Callable, prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get named modules contained by the sub graph of output names."""
        sub_graph = self.get_sub_graph(out_names)
        _modules = {}

        def _get_modules(node):
            if isinstance(node.op, nn.Module):
                mod_name = self._node_name_2_mod_name[node.name]
                _modules[mod_name] = self._modules[mod_name]

        sub_graph.post_order_dfs_visit(_get_modules)

        memo = set()
        for n, m in _modules.items():
            iters = iter_fn(m)
            for k, v in iters:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = prefix + n + ("." if n else "") + k
                yield name, v

    def named_parameters_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named parameters that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_parameters(), prefix
        )
        return gen

    def named_buffers_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named buffers that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_buffers(), prefix
        )
        return gen

    def named_modules_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named modules that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_modules(), prefix
        )
        return gen

    def split_module(
        self,
        out_names,
        split_node_name=None,
        start_node_name="img",
        common_module_flatten=False,
    ):
        """
        Split the model into two parts, the first part is common part, the second part is split part.

        Args:
            out_names (list): output names of the model.
            split_node_name (str): the name of the node which is used to split the model,
                                   if None, will auto search the graph starting by start_node.
            start_node_name (str): the name of the node to start searching the computation graph.

        Note:
            Due to the limitation of the current implementation,
            the split node encountered first will be used.
            Visit function 'get_split_node_v2' for more details.

        """  # noqa
        if out_names is None:
            out_names = self._output_names
        else:
            if len(_as_list(out_names)) <= 1:
                return None, None, None, None
        graph = self.get_sub_graph(out_names)
        if not split_node_name:
            split_node = get_split_node_v2(
                graph, start_node_name=start_node_name
            )
            split_node_name = split_node.name

        common_graph, split_graph = split_by_node_name(graph, split_node_name)
        common_name2inds_fmts = OrderedDict()
        common_name2inds_fmts[common_graph.name] = tuple([[0], object])  # noqa

        common_module = self.from_existed_graph_and_modules(
            common_graph,
            node2name=self.node2name,
            output_names=[output.name for output in common_graph._outputs],
            name2inds_fmts=common_name2inds_fmts,
            flatten_outputs=common_module_flatten,
        )

        split_module = self.from_existed_graph_and_modules(
            split_graph,
            node2name=self.node2name,
            flatten_outputs=self.flatten_outputs,
            output_names=list(out_names),
            name2inds_fmts=self._name2inds_fmts,
        )

        logger.info(
            f"Split graph by node name: {split_node_name};"
            + "Common nodes are {common_graph.get_children_name(True)}"
        )
        split_module_input_names = split_graph.input_names
        split_module_opt_input_names = split_graph.optional_input_names
        for name in self.name2nodes.keys():
            output_nodes = []
            for node in self.name2nodes[name]:
                for split_node in split_graph._outputs:
                    if node.name == split_node.name:
                        output_nodes.append(split_node)
            if len(output_nodes) > 0:
                split_module._cached_graphs[name] = Symbol(output_nodes)
                split_module.name2nodes[name] = output_nodes
        return (
            common_module,
            split_module,
            split_module_input_names,
            split_module_opt_input_names,
        )
