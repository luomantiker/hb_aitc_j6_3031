from typing import Any, Dict, Optional, Sequence


class MemoryOptSwitch:
    def __init__(
        self,
        name: str,
        default_value,
        valid_values: Sequence,
        levels: Sequence[int] = None,
    ) -> None:
        self.name = name
        self.valid_values = valid_values

        if levels is None:
            levels = range(len(valid_values))

        self.levels = levels
        self.level_to_value = {}

        for v, l in zip(valid_values, levels):
            if l is not None:
                self.level_to_value[l] = v

        self.value = default_value
        self.default_value = default_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        assert (
            v in self.valid_values
        ), "Value {} is invalid, only can be chosen in {}".format(
            v, self.valid_values
        )
        self._value = v

    def set_by_level(self, level):
        if level not in self.level_to_value:
            level = max(self.level_to_value.keys())

        self.value = self.level_to_value[level]

    def reset(self):
        self.value = self.default_value

    def __repr__(self) -> str:
        ret = "MemoryOptSwitch({}): ".format(self.name)
        ret += "Default {}, ".format(self.default_value)
        ret += "Current {}, ".format(self.value)
        ret += "Options ["
        for v, l in zip(self.valid_values, self.levels):
            if l is None:
                ret += "{}, ".format(v)
            else:
                ret += "{}(O{}), ".format(v, l)

        ret = ret[:-2] + "]"

        return ret


class MemoryOptManager:
    """A context manager to control plugin internal memory optimizations.

    Args:
        level: Control the level of policy to be applied.
            0 or None -- Do not apply any policies.
            1 -- Apply the most efficient policies with high memory saving and
                low computation overhead.
            2 -- Save most memory and do not care the computation overhead.
            Defaults to None.
        others: A mapping from policy name to the value to be setted.
            Get all availiable policies using `self.list_switch`.
            Defaults to None.
    """

    _SWITCHS: Dict[str, MemoryOptSwitch] = {}

    def __init__(
        self,
        level: Optional[int] = None,
        others: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.level = level
        self.others = others
        self.old_state = {}

    def __enter__(self):
        for name, s in self._SWITCHS.items():
            self.old_state[name] = s.value

        self.set(self.level, self.others)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, s in self._SWITCHS.items():
            s.value = self.old_state[name]

    @classmethod
    def register_switch(cls, switch: MemoryOptSwitch):
        cls._SWITCHS[switch.name] = switch

    @classmethod
    def list_switch(cls):
        return cls._SWITCHS.values()

    @classmethod
    def set(
        cls,
        level: Optional[int] = None,
        others: Optional[Dict[str, Any]] = None,
    ):
        if level is not None:
            for s in cls._SWITCHS.values():
                s.set_by_level(level)
        if others is not None:
            for name, value in others.items():
                assert name in cls._SWITCHS
                cls._SWITCHS[name].value = value

    @classmethod
    def reset(cls):
        for s in cls._SWITCHS.values():
            s.reset()

    def __repr__(self) -> str:
        ret = "MemoryOptManager(O{}) with following switches:\n".format(
            self.level
        )
        for s in self._SWITCHS.values():
            ret += str(s) + "\n"

        return ret
