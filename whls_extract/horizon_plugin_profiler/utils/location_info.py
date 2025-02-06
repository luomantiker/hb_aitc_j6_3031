import os
import traceback
from inspect import isfunction
from typing import Any, Dict, List, Tuple

import torch

from horizon_plugin_pytorch import nn

_custom_skipped_path = []


def add_custom_ignored_stack_filepath(path: str):
    """Add custom ignored file when searching stack for user code."""
    _custom_skipped_path.append(path)


def get_user_stack_info() -> str:
    """Seek current stack and find the most deep frame of user code.

    Returns:
        str: Trace back info ready for printing.
    """
    import horizon_plugin_pytorch

    torch_path = os.path.abspath(os.path.dirname(torch.__file__))
    plugin_path = os.path.abspath(
        os.path.dirname(horizon_plugin_pytorch.__file__)
    )

    skipped_path = [torch_path, plugin_path] + _custom_skipped_path

    try:
        import horizon_plugin_profiler

        skipped_path.append(os.path.dirname(horizon_plugin_profiler.__file__))
    except ImportError:
        pass

    stacks = traceback.extract_stack()
    for stack in reversed(stacks):
        source_path = os.path.abspath(stack.filename)
        if not source_path.startswith(tuple(skipped_path)):
            break
        elif "torchvision" in source_path:
            break

    return traceback.format_list([stack])[0]


class PicklableStrDict(dict):
    """A dict that can be serilized to a string.

    Keys and values must be string.
    """

    _split = chr(30)
    _escapes = chr(31)
    _replace_map = {}

    def pickle(self):
        ret = ""
        for k, v in self.items():
            ret += "[{}]: {}{}".format(k, v, self._split)
        ret = ret[:-1]

        for k, v in self._replace_map.items():
            ret = ret.replace(k, v)

        return ret

    @classmethod
    def unpickle(cls, string: str):
        if "\\" in string:
            string = string.replace("\\1E", cls._split)
            string = string.replace("\\1F", cls._escapes)

        obj = cls()

        for k, v in cls._replace_map.items():
            string = string.replace(v, k)

        items = string.split(cls._split)
        for item in items:
            right_bracket_idx = item.find("]")
            k = item[1:right_bracket_idx]
            v = item[right_bracket_idx + 3 :]

            obj[k] = v

        return obj

    @classmethod
    def init_class(cls):
        if len(cls._replace_map) > 0:
            raise RuntimeError("PicklableStrDict is already inited")

        for special_char, replace in zip(
            (chr(i) for i in range(32, 48)), (chr(i) for i in range(65, 81))
        ):
            cls._replace_map[special_char] = cls._escapes + replace
        for special_char, replace in zip(
            (chr(i) for i in range(58, 65)), (chr(i) for i in range(81, 88))
        ):
            cls._replace_map[special_char] = cls._escapes + replace
        for special_char, replace in zip(
            (chr(i) for i in range(91, 97)), (chr(i) for i in range(97, 103))
        ):
            cls._replace_map[special_char] = cls._escapes + replace
        for special_char, replace in zip(
            (chr(i) for i in range(123, 127)),
            (chr(i) for i in range(103, 107)),
        ):
            cls._replace_map[special_char] = cls._escapes + replace
        cls._replace_map["\n"] = cls._escapes + chr(107)


PicklableStrDict.init_class()


class TorchLocationInfo:
    """Information to identify a torch operation calling."""

    def __init__(
        self, op, mod_name: str, idx: int, user_stack: str = None, op_type=None
    ) -> None:
        self.op_name = self.format_op_name(op)
        self.mod_name = mod_name
        self.idx = idx
        if op_type is None:
            self.op_type = (
                "call_module"
                if isinstance(op, torch.nn.Module)
                else "call_function"
            )
        else:
            self.op_type = op_type
        self.user_stack = (
            get_user_stack_info() if user_stack is None else user_stack
        )

    @staticmethod
    def _format_func_name(op):
        if getattr(torch.Tensor, op.__name__, None) is op:
            return "torch.Tensor.{}".format(op.__name__)
        elif getattr(torch, op.__name__, None) is op:
            return "torch.{}".format(op.__name__)
        elif isfunction(op):
            return "{}.{}".format(op.__module__, op.__name__)
        else:
            return str(op)

    @staticmethod
    def _format_module_name(op: torch.nn.Module):
        if isinstance(
            op,
            (
                nn.quantized.FloatFunctional,
                nn.quantized.QFunctional,
                nn.qat.FloatFunctional,
            ),
        ):
            name = f"{op.__module__}.{op.__class__.__name__}"
            if hasattr(op, "_last_called_method_name"):
                name += f".{op._last_called_method_name}"
            return name
        elif isinstance(op, torch.nn.Module):
            return "{}.{}".format(op.__module__, op.__class__.__name__)
        else:
            raise TypeError(
                "op must be a nn.Module, but receive{}".format(type(op))
            )

    @staticmethod
    def format_op_name(op):
        if isinstance(op, str):
            return op
        elif isinstance(op, torch.nn.Module):
            return TorchLocationInfo._format_module_name(op)
        else:
            return TorchLocationInfo._format_func_name(op)

    @staticmethod
    def format_op_type(op):
        if type(op) == type and issubclass(op, torch.nn.Module):
            return "{}.{}".format(op.__module__, op.__name__)
        else:
            return TorchLocationInfo._format_func_name(op)

    def type(self):
        return self.op_type

    def to_dict(self):
        return PicklableStrDict(
            {
                "OP": self.op_name,
                "Mod": self.mod_name,
                "Id": str(self.idx),
                "Type": self.op_type,
                "Code": self.user_stack,
            }
        )

    def pickle(self):
        return self.to_dict().pickle()

    @classmethod
    def from_dict(cls, str_dict):
        try:
            return cls(
                str_dict["OP"],
                str_dict["Mod"],
                int(str_dict["Id"]),
                str_dict["Code"],
                str_dict["Type"],
            )
        except KeyError:
            return cls("unknown", "unknown", -1, "unknown", "unknown")

    @classmethod
    def unpickle(cls, string: str):
        str_dict = PicklableStrDict.unpickle(string)
        return cls.from_dict(str_dict)

    def get_contents(self):
        return (
            self.op_name,
            self.mod_name,
            str(self.idx),
            self.user_stack,
        )

    @classmethod
    def get_headers(cls):
        return ("OP", "Mod", "Id", "Code")

    def __hash__(self) -> int:
        return str.__hash__(self.pickle())


class LocationManager:
    """Manage the TorchLocationInfo in model forward.

    Used as a context manager:

    with LocationManager(model):
        model(inputs)

    And code in model forward can get current location info
    by `LocationManager.get()`.

    Args:
        model (torch.nn.Module): The model to run.
        with_stack_info: whether record code stack info. Default: True.
    """

    _current = None

    def __init__(
        self, model: torch.nn.Module, with_stack_info: bool = True
    ) -> None:
        self.model = model
        self.with_stack_info = with_stack_info
        self._reset()

    def _reset(self):
        self.handles = []
        self.loc_info_stack: List[Tuple[str, str]] = []
        self.mod_called_times: Dict[str, int] = {}
        self.op_called_times_in_scope: Dict[str, Dict[Any, int]] = {}

    @classmethod
    def _get_instance(cls):
        if cls._current is None:
            raise RuntimeError(
                "No {} instance is activated now".format(cls.__name__)
            )
        return cls._current

    @classmethod
    def push(cls, mod_name: str, with_stack_info: bool = True):
        self = cls._get_instance()
        stack_info = get_user_stack_info() if with_stack_info else None
        self.loc_info_stack.append((mod_name, stack_info))

    @classmethod
    def pop(cls):
        self = cls._get_instance()
        self.loc_info_stack.pop()

    @classmethod
    def get(cls, op, update_user_stack=False):
        self = cls._get_instance()
        mod_name, user_stack = self.loc_info_stack[-1]
        if update_user_stack:
            user_stack = get_user_stack_info()

        if isinstance(op, torch.nn.Module):
            idx = self.mod_called_times.get(mod_name, 0)
            self.mod_called_times[mod_name] = idx + 1
        else:
            op_called_times = self.op_called_times_in_scope.get(mod_name, {})
            idx = op_called_times.get(op, 0)
            op_called_times[op] = idx + 1
            self.op_called_times_in_scope[mod_name] = op_called_times

        return TorchLocationInfo(op, mod_name, idx, user_stack)

    @classmethod
    def gen_hooks(cls, qualified_name: str, with_stack_info: bool = True):
        def _pre_forward_hook(mod, *args, **kwargs):
            cls.push(qualified_name, with_stack_info)

        def _forward_hook(mod, *args, **kwargs):
            cls.pop()

        return _pre_forward_hook, _forward_hook

    def __enter__(self, *args, **kwargs):
        self.old_instance = self.__class__._current
        self.__class__._current = self

        self._reset()

        for n, m in self.model.named_modules():
            # quantized models may have scriptmodule
            if isinstance(m, torch.jit.ScriptModule):
                continue

            location_push, location_pop = self.gen_hooks(
                n, self.with_stack_info
            )
            self.handles.append(
                (
                    m.register_forward_pre_hook(location_push),
                    m.register_forward_hook(location_pop),
                )
            )

    def __exit__(self, *args, **kwargs):
        self.__class__._current = self.old_instance

        for pre_h, h in self.handles:
            pre_h.remove()
            h.remove()
