import csv
import os

import torch
from tabulate import tabulate

from .location_info import LocationManager
from .misc import tensor_struct_repr
from .model_helper import HookAndTorchFunctionHelper


def _write_tabulate_to_csv(output_path, contents, headers):
    if not output_path.endswith(".csv"):
        output_path += ".csv"
    dir_name = os.path.dirname(output_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(contents)


class OpsRecorder(HookAndTorchFunctionHelper):
    _current = None

    class TracerTensor(HookAndTorchFunctionHelper.TracerTensor):
        """Patch Tensor for tracing."""

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__."""
            assert OpsRecorder._current is not None
            tracer = OpsRecorder._current
            loc = LocationManager.get(func, update_user_stack=tracer.verbose)
            tracer.add_item(
                loc.op_name,
                loc.mod_name,
                tensor_struct_repr(cls.unwrap(args)),
                tensor_struct_repr(cls.unwrap(kwargs)),
                tensor_struct_repr(func_ret),
                "",
                loc.user_stack,
            )
            return super()._torch_function_postprocess(
                func, types, args, kwargs, func_ret
            )

    def __init__(self, verbose) -> None:
        super().__init__()
        self._headers = ["OP Name", "Scope", "Args", "Kwargs", "Rets"]
        self.verbose = verbose
        if verbose:
            self._headers += ["Mod info", "Call Stack"]
        self._records = []

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook."""
        loc = LocationManager.get(mod)
        self.add_item(
            loc.op_name,
            loc.mod_name,
            tensor_struct_repr(args),
            tensor_struct_repr(kwargs),
            tensor_struct_repr(output),
            str(mod) if self.verbose else None,
            loc.user_stack,
        )
        return super()._forward_hook(mod, args, kwargs, output)

    def add_item(
        self,
        op_name: str,
        scope: str,
        args: str,
        kwargs: str,
        rets: str,
        mod_info: str,
        call_stack: str,
    ):
        if self.verbose:
            self._records.append(
                (op_name, scope, args, kwargs, rets, mod_info, call_stack)
            )
        else:
            self._records.append((op_name, scope, args, kwargs, rets))

    def trace(self, *args, **kwargs):
        OpsRecorder._current = self
        ret = super()._register_hook_and_forward(*args, **kwargs)
        OpsRecorder._current = None
        return ret

    def summary(self):
        return tabulate(self._records, self._headers)

    def save_to(self, output_path: str):
        _write_tabulate_to_csv(
            output_path,
            self._records,
            self._headers,
        )


def record_ops(
    model: torch.nn.Module,
    example_inputs,
    example_kw_inputs: dict = None,
    output_path: str = None,
    print_to_screen: bool = False,
    verbose: bool = False,
):
    """Record all operations and their args, inputs, outputs.

    Args:
        model (torch.nn.Module): Input model.
        example_inputs : Model inputs.
        example_kw_inputs (dict, optional): Model key work inputs.
            Defaults to None.
        output_path (str, optional): Save report as csv file. Defaults to None.
        print_to_screen (bool, optional): Whether print output to screen.
            Defaults to None.
        verbose (bool, optional): Whether include the code line of each op.
            Defaults to False.
    """
    if output_path is None:
        print_to_screen = True
    tracer = OpsRecorder(verbose)
    tracer.trace(model, example_inputs, example_kw_inputs)
    if output_path is not None:
        tracer.save_to(output_path)
    if print_to_screen:
        print(tracer.summary())


def record_hbir_ops(
    model: torch.nn.Module,
    example_inputs,
    output_path: str = None,
    print_to_screen: bool = False,
    verbose: bool = False,
):
    from horizon_plugin_pytorch.quantization.hbdk4.export_hbir.export_hbir import (  # noqa: E501
        Exporter,
        JitTensor,
    )

    JitTensor._allow_fall_through = True
    raw_hbir_graph = Exporter.export(
        model, example_inputs, False, _get_raw_graph=True
    )
    JitTensor._allow_fall_through = False

    op_records = {}

    for op in raw_hbir_graph.operation.opview.body.operations[0].body.blocks[
        0
    ]:
        op_name = op.name

        if op_name in ("func.return", "hbir.constant"):
            continue

        unsupported_reason = ""
        op_sig = str(op)

        if op_name == "hbtl.call":
            if hasattr(op, "parameters"):
                unsupported_reason = str(op.parameters[0])
            op_name = op.signature.value.split("(")[0].replace("::", ".")

        if op_name in op_records:
            op_records[op_name].append((op_sig, unsupported_reason))
        else:
            op_records[op_name] = [(op_sig, unsupported_reason)]

    if verbose:
        headers = ("HBIR OP", "Signature", "Unsupported Reason")

        def expand_op_records(op_records):
            for k, v in op_records.items():
                for op_sig, unsupported_reason in v:
                    yield (k, op_sig, unsupported_reason)

        contents = expand_op_records(op_records)
    else:
        headers = ("HBIR OP",)
        contents = ((k,) for k in op_records.keys())

    if output_path is None:
        print_to_screen = True
    else:
        _write_tabulate_to_csv(output_path, contents, headers)

    if print_to_screen:
        print(tabulate(contents, headers))
