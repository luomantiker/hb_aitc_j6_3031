import os

import onnxruntime as ort
from onnxruntime_extensions import get_library_path

from hmct.ir import serialize_model

from .ort import ORTExecutorBase


def get_operators_path():
    library_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    library_path += "/horizon_operators.so"
    return library_path


class ORTExecutor2(ORTExecutorBase):
    def __init__(self, model):
        super().__init__(model)
        self.to(ORTExecutor2.get_support_devices())

    @staticmethod
    def _ort():
        return ort

    def _cuda_provider(self):
        return ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})

    def create_session(self):
        """Create onnxruntime inference session."""
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = (
            10  # 设置执行单个操作的并行线程数, 以避免 pthread_setaffinity_np 报错
        )
        so.use_deterministic_compute = True
        so.register_custom_ops_library(get_operators_path())
        so.register_custom_ops_library(get_library_path())
        self._sess = ort.InferenceSession(
            serialize_model(self._model),
            sess_options=so,
            providers=self._providers,
        )
        # obtain model inputs && outputs information
        self._load_inputs_outputs()
        return self
