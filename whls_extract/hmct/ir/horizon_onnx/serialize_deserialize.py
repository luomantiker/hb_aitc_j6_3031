import os
import time
from typing import Union

from onnx import ModelProto, load, load_from_string, save

from hmct.utility import TEMP_DIR

LARGEST_NO_EXTERNAL_DATA_MODEL = 2**31 - 1


def serialize_proto(onnx_proto: ModelProto) -> Union[str, bytes]:
    """序列化ModelProto对象.

    小于2GB的模型序列化为字节流, 大于2GB的模型序列化为存储路径
    """
    if onnx_proto.ByteSize() < LARGEST_NO_EXTERNAL_DATA_MODEL:
        model_str = onnx_proto.SerializeToString()
    else:
        model_str = TEMP_DIR.relpath(f"temp.{int(time.time())}.onnx")
        save(
            onnx_proto,
            model_str,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )
    return model_str


def deserialize_proto(model_str: Union[str, bytes]) -> ModelProto:
    """从模型的字节流或者存储路径中反序列化得到ModelProto对象."""
    if isinstance(model_str, bytes):
        model_proto = load_from_string(model_str)
    else:
        if os.path.isfile(model_str):
            model_proto = load(model_str)
        else:
            raise TypeError(f"{model_str} is not a valid file path, please check it.")
    return model_proto
