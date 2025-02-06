import collections
import dataclasses
import os
import traceback
from typing import Any, Callable, List, Optional

import torch
from tabulate import tabulate

try:
    from torch import _dynamo as torch_dynamo
    from torch._dynamo.output_graph import GraphCompileReason
    from torch._guards import Guard
    from torch.fx import GraphModule, Node
except ImportError:
    pass

from hat.utils.global_var import set_value
from hat.utils.package_helper import check_packages_available


def disable_compile(
    fn: [Callable] = None,
    recursive: bool = True,
):  # noqa D401
    """Wrapper of `torch.compiler.disable`.

    This function provides both a decorator and a context manager to disable
    compilation on a function. And It also provides the option of recursively
    disabling called functions.

    Args:
        fn: The function to disable.
        recursive: Whether the disabling should be recursive.
    """
    if torch_dynamo and check_packages_available(
        "torch>=2.0", raise_exception=False
    ):
        if recursive:
            return torch_dynamo.disable(fn=fn)
        else:
            return torch_dynamo.skip(fn=fn)
    else:
        pass


@dataclasses.dataclass
class ExplainOutput:
    """Custom output of `torch._dynamo.explain()`."""

    graphs: List["GraphModule"]
    graph_count: int
    graph_break_count: int
    break_reasons: List[Any]
    op_count: int
    ops_per_graph: Optional[List["Node"]] = None
    out_guards: Optional[List["Guard"]] = None
    compile_times: Optional[str] = None

    def __str__(self):
        output = f"Graph Count: {self.graph_count}\n"
        output += f"Graph Break Count: {self.graph_break_count}\n"
        output += f"Op Count: {self.op_count}\n"

        output += "Break Reasons:\n"

        deduped_reasons = {}
        for reason in self.break_reasons:
            innermost_frame = reason.user_stack[-1]
            # __repr__ uniquely identifies a FrameSummary
            # so we can use it for deduping
            deduped_reasons[repr(innermost_frame)] = reason

        for idx, break_reason in enumerate(deduped_reasons.values()):
            output += f"  Break Reason {idx+1}:\n"
            output += f"    Reason: {break_reason.reason}\n"
            formatted_stack = "".join(
                traceback.format_list(break_reason.user_stack)
            )
            output += f"    User Stack:\n{formatted_stack}\n"

        if self.ops_per_graph is not None:
            output += "Ops per Graph:\n"
            for idx, ops in enumerate(self.ops_per_graph):
                output += f"  Ops {idx+1}:\n"
                for op in ops:
                    output += f"    {op}\n"

        if self.out_guards is not None:
            output += "Out Guards:\n"
            for i, guard in enumerate(self.out_guards):
                output += f"  Guard {i+1}:\n"
                output += f"    {str(guard)}"

        if self.compile_times is not None:
            output += f"Compile Times: {self.compile_times}\n"
        return output


@dataclasses.dataclass
class GuardFailureOutput:
    """Custom output of `CompileProfiler`."""

    guard_failures: collections.defaultdict(list)

    def __str__(self) -> str:
        def num_recompiles(code):
            return len(self.guard_failures[code])

        def recompile_reasons(code):
            return "\n".join([str(x) for x in self.guard_failures[code]])

        def format_func_info(code):
            short_filename = code.co_filename.split("/")[-1]
            return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"

        summarized_gf = [
            [
                format_func_info(code),
                num_recompiles(code),
                recompile_reasons(code),
            ]
            for code in self.guard_failures
        ]

        output = "Torchdynamo Profiler Report:\n"
        if len(self.guard_failures) > 0:
            max_recompiles = max(
                [num_recompiles(code) for code in self.guard_failures]
            )
            output += "\n"
            output += (
                "These subgraphs were recompiled more than once due "
                "to guard failures."
            )
            output += (
                "Guard failures indicate some condition assumed to be static "
                "by the tracer changed, making it unsafe to reuse the compiled"
                " program."
            )
            output += tabulate(
                summarized_gf,
                headers=["Function", "Num Recompiles", "Recompile Reasons"],
                tablefmt="grid",
            )
            output += "\n"
            output += (
                f"Set torch._dynamo.config.cache_size_limit to "
                f"{max_recompiles} to avoid being cache limited.\n"
            )

        else:
            output += "No cache-limited recompilations detected.\n"

        return output


def _explain_graph_detail(
    gm: "GraphModule",
    graphs: List["GraphModule"],
    op_count: int,
    ops_per_graph: List[Callable],
    break_reasons: List["GraphCompileReason"],
):
    """Get details from Dynamo's graph capture.

    Note:
    This function is a utility which processes a torch.fx.GraphModule and
    accumulates information about its ops, graph breaks, and other details.

    Args:
        gm: The GraphModule to be processed.
        graphs: A list that accumulates all the GraphModules processed.
        op_count: Number of operations in all GraphModules processed so far.
        ops_per_graph: List of the operations of each GraphModule.
        break_reasons: List of the reasons for breaks in each GraphModule.
    """

    graphs.append(gm)
    ops = [
        node.target for node in gm.graph.nodes if node.op == "call_function"
    ]
    op_count += len(ops)
    ops_per_graph.append(ops)
    if gm.compile_subgraph_reason is not None:
        break_reasons.append(gm.compile_subgraph_reason)

    return gm, graphs, op_count, ops_per_graph, break_reasons


class CompileBackendWrapper:
    """Wrapper for torch compile backend, which support debug compile.

    Note:
    This class is intended to be used as a backend for `torch.compile`. It
    accumulates information about graph breaks, ops, guard_failures and other
    info and provides a string representation summarizing this information.

    Args:
        backend: Compile backend.
        kwargs: Args of backend.

    Example:
    >>> def fn(x):
    ...    x = torch.sigmoid(x)
    ...    return x
    >>> torch._dynamo.reset()
    >>> eb = CompileBackendWrapper(backend="inductor")
    >>> optimized_fn = torch.compile(fn, backend=eb)
    >>> result = optimized_fn(torch.randn(5))
    >>> print(eb.get_explain_output(), eb.get_guard_failures_output())
    """

    compile_wrapper_name = "explain_with_backend"

    def __init__(
        self,
        backend="inductor",
        **kwargs,
    ) -> None:
        self._with_profiler = None
        self.backend = self._get_compile_backend(backend=backend, **kwargs)

        if self.with_explain:
            self.reset_status()
            self.backend_ctx_ctor = (
                lambda: torch_dynamo.utils.disable_cache_limit()
            )
            set_value(self.compile_wrapper_name, self)

    def __call__(self, gm: "GraphModule", example_inputs, **kwargs) -> Any:
        if self.with_explain:
            (
                gm,
                self.graphs,
                self.op_count,
                _,
                self.break_reasons,
            ) = _explain_graph_detail(
                gm, self.graphs, self.op_count, [], self.break_reasons
            )
        return self.backend(gm, example_inputs)

    def get_explain_output(self):
        graph_count = len(self.graphs)
        compile_times = torch_dynamo.utils.compile_times()
        output = ExplainOutput(
            self.graphs,
            graph_count,
            graph_count - 1,
            self.break_reasons,
            self.op_count,
            compile_times=compile_times,
        )

        return output

    def get_guard_failures_output(self):
        guard_failures = torch_dynamo.utils.guard_failures
        output = GuardFailureOutput(guard_failures)

        return output

    def _get_compile_backend(self, backend, **kwargs):
        from torch._dynamo.backends.registry import lookup_backend

        if backend == "inductor":
            backend = torch._TorchCompileInductorWrapper(
                mode=kwargs.get("mode", None),
                options=kwargs.get("options", None),
                dynamic=kwargs.get("dynamic", False),
            )
        else:
            # TODO(mengyang.duan): fix other backend
            backend = lookup_backend(backend)

        name_prefix = "wrapper_" if self.with_explain else ""
        if hasattr(backend, "compiler_name"):
            self.compiler_name = name_prefix + backend.compiler_name
        elif isinstance(backend, str):
            self.compiler_name = name_prefix + backend
        else:
            self.compiler_name = None

        return backend

    @property
    def with_explain(self):
        if self._with_profiler is None:
            self._with_profiler = bool(
                os.getenv("HAT_WITH_DYNAMO_PROFILER", "0") == "1"
            )
        return self._with_profiler

    def reset(self):
        if hasattr(self.backend, "reset"):
            self.backend.reset()

    def reset_status(self):
        self.graphs = []
        self.op_count = 0
        self.break_reasons = []
        self.frame_count = 0
        torch_dynamo.reset()
