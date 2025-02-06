# Temporarily put registered module class mapping here to avoid circular import
# TODO: Merge with horizon_plugin_pytorch/quantization/quantization_mappings.py

_QAT_MODULE_MAPPINGS = {}
_QUANT_MODULE_MAPPINGS = {}


def register_float_to_qat_mapping(float_mod_class, qat_mod_class):
    if (
        float_mod_class in _QAT_MODULE_MAPPINGS
        and _QAT_MODULE_MAPPINGS[float_mod_class] is not qat_mod_class
    ):
        raise ValueError(
            "Trying to register multi qat mod class for {}".format(
                float_mod_class
            )
        )
    _QAT_MODULE_MAPPINGS[float_mod_class] = qat_mod_class


def get_float_to_qat_mapping():
    return _QAT_MODULE_MAPPINGS


def register_qat_to_quantized_mapping(qat_mod_class, quantized_mod_class):
    if (
        qat_mod_class in _QUANT_MODULE_MAPPINGS
        and _QUANT_MODULE_MAPPINGS[qat_mod_class] is not quantized_mod_class
    ):
        raise ValueError(
            "Trying to register multi qat mod class for {}".format(
                qat_mod_class
            )
        )
    _QUANT_MODULE_MAPPINGS[qat_mod_class] = quantized_mod_class


def get_qat_to_quantized_mapping():
    return _QUANT_MODULE_MAPPINGS
