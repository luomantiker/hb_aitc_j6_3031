from hmct.ir import OnnxModel

from .caffe_parser_impl.caffe_to_onnx import convert_caffe_to_onnx
from ..registry import register_parse_func


@register_parse_func(model_type="caffe")
def parse_caffe(prototxt_file: str, caffemodel_file: str) -> OnnxModel:
    model_proto = convert_caffe_to_onnx(
        prototxt_file=prototxt_file,
        caffemodel_file=caffemodel_file,
    )
    return OnnxModel(model_proto)
