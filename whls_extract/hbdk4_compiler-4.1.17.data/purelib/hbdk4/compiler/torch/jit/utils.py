from typing import Union
import re
import torch


def dfs(x, type, func):
    if isinstance(x, (list, tuple)):
        return [dfs(i, type, func) for i in x]
    if isinstance(x, dict):
        return {k: dfs(x[k], type, func) for k in sorted(x.keys())}
    if isinstance(x, type):
        return func(x)
    return x


def get_graph(jit: Union[torch.jit.TracedModule, torch.jit.ScriptModule, torch.Graph]):
    if isinstance(jit, torch.jit.ScriptFunction):
        jit = jit.graph
    if isinstance(jit, torch.jit.ScriptModule):
        jit = jit.forward.graph
    if isinstance(jit, torch.jit.TracedModule):
        jit = jit.forward.graph
    if not isinstance(jit, torch.Graph):
        raise ValueError("unknown jit object")
    return jit


def split_type_value(arg: str):
    # match " "
    for i in range(1, len(arg)):
        if arg[-i - 1 : -i] == " ":
            return arg[: -i - 1], arg[-i:]
    return arg, ""


class Argument:
    def __init__(self, arg: str, is_kwd: bool = False):
        self.is_kwd = is_kwd
        self.type, self.name = split_type_value(arg)

        if "=" in self.name:  # with default value
            self.name, self.value = self.name.split("=")

    def is_optional(self) -> bool:
        return hasattr(self, "value")

    def is_return(self) -> bool:
        return hasattr(self, "name")


def split_return(schema: str):
    # match " -> "
    for i in range(1, len(schema)):
        if schema[-i - 4 : -i] == " -> ":
            return schema[: -i - 4], schema[-i:]


class Schema:
    def __init__(self, schema: str):
        self.schema = schema

        sig_arg_part, return_part = split_return(schema)

        # Extracting function returns
        self.results = [Argument(res) for res in return_part.split(", ")]

        # Extracting namespace, function, and function variant
        sig_match = re.search(r"^([^()]+)", sig_arg_part)
        assert sig_match.span()[0] == 0  # must in the beginning

        namespace = sig_match.group(1).split("::")[0]
        signature = sig_match.group(1).split("::")[1].split(".")

        self.namespace = namespace.strip()
        self.function = signature[0]
        self.variant = None if len(signature) == 1 else signature[1]

        args_match = sig_arg_part[sig_match.span()[1] :]
        assert args_match[0] == "(" and args_match[-1] == ")"
        args_match = args_match[1:-1]

        self.pargs = []
        self.kwargs = []

        is_kwd = False
        for arg in args_match.split(", "):
            if arg.strip() == "*":
                is_kwd = True
                continue

            if is_kwd:
                self.kwargs.append(Argument(arg, True))
            else:
                self.pargs.append(Argument(arg))

    def format(self, *args):
        assert len(args) == len(self.pargs) + len(self.kwargs)
        pargs = args[: len(self.pargs)]

        kwargs = {}
        for a, v in zip(self.kwargs, args[len(self.pargs) :]):
            kwargs[a.name] = v
        return pargs, kwargs
