from hmct.ir import serialize_model

from . import session as ort
from .ort import ORTExecutorBase


class ORTExecutor1(ORTExecutorBase):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def _ort():
        return ort

    def _cuda_provider(self):
        return "CUDAExecutionProvider"

    def create_session(self):
        """Create onnxruntime inference session."""
        self._sess = ort.InferenceSession(
            serialize_model(self._model),
            providers=self._providers,
        )
        # obtain model inputs && outputs information
        self._load_inputs_outputs()
        return self
