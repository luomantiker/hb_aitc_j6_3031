import enum
from typing import Optional


@enum.unique
class MarchSeries(enum.Enum):
    # This class is used for external
    # NOTE-BPU-MARCH
    bayes = enum.auto()
    nash = enum.auto()


@enum.unique
class MarchBase(enum.Enum):
    def __repr__(self):
        return '"%s"' % self.name  # use double quote, different from str


@enum.unique
class March(MarchBase):
    # This class is used for external
    # NOTE-BPU-MARCH
    unnamed = 0
    bayes = enum.auto()
    nash_e = enum.auto()
    nash_m = enum.auto()
    nash_p = enum.auto()
    nash_b = enum.auto()

    def __init__(self, _, data=None):
        self.__data = data

    @property
    def series(self):
        if self == March.bayes:
            return MarchSeries.bayes
        elif self in (
            March.nash_e,
            March.nash_m,
            March.nash_p,
            March.nash_b,
        ):
            return MarchSeries.nash

        assert False, "invalid march %s" % self.name

    @property
    def data(self):
        return self.__data

    @staticmethod
    def get(march: str, return_invalid_str: bool = False) -> Optional["March"]:
        march_name = march

        if not isinstance(march_name, str):
            march_name = str(march_name)
        march_name_lower = march_name.lower()
        if march_name_lower in "unnamed":
            new_march = March(March.unnamed)
            new_march.__data = march
            return new_march
        elif march_name_lower in "bayes":
            return March.bayes
        elif march_name_lower in "nash-b":
            return March.nash_b
        elif march_name_lower in "nash-e":
            return March.nash_e
        elif march_name_lower in "nash-m":
            return March.nash_m
        elif march_name_lower in "nash-p":
            return March.nash_p

        return march if return_invalid_str else None

    @staticmethod
    def test():
        assert March.bayes in March
        assert list(March) == [
            March.unnamed,
            March.bayes,
            March.nash_b,
            March.nash_e,
            March.nash_m,
            March.nash_p,
        ]
        assert len(March) == 6


if __name__ == "__main__":
    March.test()
