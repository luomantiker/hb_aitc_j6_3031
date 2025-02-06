class QuantizationType:
    def __init__(self):
        # max, kl, percentile, minmax
        self.calibration_method = ""

        self.asymmetric = False

        self.perchannel = False

        self.perblock = False

        self.int16 = False

        self.bias_correction = False

        self.default = False

        self.weight = False

    def type_str(self):
        cal_type = f"{self.calibration_method}"

        if self.asymmetric:
            cal_type = "{}_{}".format(cal_type, "asy")

        if self.perchannel:
            cal_type = "{}_{}".format(cal_type, "perchannel")

        if self.perblock:
            cal_type = "{}_{}".format(cal_type, "perblock")

        if self.int16:
            cal_type = "{}_{}".format(cal_type, "int16")

        if self.bias_correction:
            cal_type = "{}_{}".format(cal_type, "bias")

        if self.weight:
            cal_type = "{}_{}".format(cal_type, "weight")

        return cal_type

    def set_method(self, calibration_method):
        if "percentile" in calibration_method:
            self.calibration_method = "percentile"
        elif "kl" in calibration_method:
            self.calibration_method = "kl"
        elif "min-max" in calibration_method:
            self.calibration_method = "minmax"
        elif "max" in calibration_method:
            self.calibration_method = "max"
        else:
            self.calibration_method = calibration_method
        if "per_channel" in calibration_method:
            self.perchannel = True
        else:
            self.perchannel = False
        if "block_size" in calibration_method:
            self.perblock = True
        else:
            self.perblock = False
        if "asymmetric" in calibration_method:
            self.asymmetric = True
        else:
            self.asymmetric = False

    def update(self, other: "QuantizationType") -> None:
        if other.calibration_method:
            self.calibration_method = other.calibration_method
        if other.asymmetric:
            self.asymmetric = other.asymmetric
        if other.perchannel:
            self.perchannel = other.perchannel
        if other.perblock:
            self.perblock = other.perblock
        if other.int16:
            self.int16 = other.int16
        if other.bias_correction:
            self.bias_correction = other.bias_correction
        if other.weight:
            self.weight = other.weight
        if other.default:
            self.default = other.default

    @property
    def method(self):
        if self.default:
            return "default"
        return self.calibration_method
