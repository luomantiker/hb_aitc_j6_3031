class QuantRoundMode:
    BPU_ROUND = "bpu_round"
    HALF_TO_EVEN = "half_to_even"
    _round_mode = HALF_TO_EVEN

    @classmethod
    def set(cls, v):
        assert v in (cls.BPU_ROUND, cls.HALF_TO_EVEN)
        cls._round_mode = v

    @classmethod
    def get(cls):
        return cls._round_mode
