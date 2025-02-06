# Globally disable the fake quantize in Module and function
from contextlib import contextmanager


class GlobalFakeQuantSwitch:
    _value = True

    @classmethod
    def state(cls):
        return cls._value

    @classmethod
    def enable(cls):
        cls._value = True

    @classmethod
    def disable(cls):
        cls._value = False

    @classmethod
    @contextmanager
    def fake_quant_enabled(cls, v: bool):
        old_value = cls._value
        cls._value = v
        yield
        cls._value = old_value
