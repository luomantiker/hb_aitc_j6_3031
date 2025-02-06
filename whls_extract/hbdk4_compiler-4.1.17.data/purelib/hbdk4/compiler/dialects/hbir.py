from ._hbir_ops_gen import *
from hbdk4.compiler._mlir_libs._hbdk import (
    HbirBool8Type,
    create_bool8_attr,
    HbdkTrackAttr,
)
from hbdk4.compiler.ir import Context
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
import codecs


class Bool8Type(HbirBool8Type):
    @staticmethod
    def get(context: Context = None) -> HbirBool8Type:
        if context is None:
            return HbirBool8Type.get(get_default_loc_context())
        else:
            return HbirBool8Type.get(context)


class TrackAttr(HbdkTrackAttr):
    @staticmethod
    def get(debug_info, context: Context = None) -> HbdkTrackAttr:
        def convert_str(s):
            mapping = {
                '"': "'",
                "\n": "\\n",
                "\t": "\\t",
                "\r": "",
                "\b": "",
            }
            for k, v in mapping.items():
                s = s.replace(k, v)
            return s

        if not isinstance(debug_info, dict):
            raise TypeError("debug info should be dict type")
        keys = list([convert_str(str(k)) for k in debug_info.keys()])
        values = list([convert_str(str(v)) for v in debug_info.values()])
        if context is None:
            return HbdkTrackAttr.get(get_default_loc_context(), keys, values)
        else:
            return HbdkTrackAttr.get(context, keys, values)

    @property
    def debug_info(self):
        return {k: v for k, v in zip(self.debug_key, self.debug_value)}
