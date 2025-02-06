import logging
from typing import Dict, Sequence, Union

import multiprocess
import numpy as np
from tqdm import tqdm

from hmct.ir import DataType, load_model, numpy_dtype_to_onnx_dtype


def ort_dtype_to_onnx_dtype(dtype: str) -> int:
    """Convert ORT dtype to ONNX dtype.

    Args:
        dtype: ORT dtype

    Returns:
        Corresponding ONNX data type.
    """
    _dtypes: Dict[str, int] = {
        "tensor(float)": DataType.FLOAT32.value,
        "tensor(uint8)": DataType.UINT8.value,
        "tensor(int8)": DataType.INT8.value,
        "tensor(uint16)": DataType.UINT16.value,
        "tensor(int16)": DataType.INT16.value,
        "tensor(int32)": DataType.INT32.value,
        "tensor(int64)": DataType.INT64.value,
        "tensor(string)": DataType.STRING.value,
        "tensor(bool)": DataType.BOOL.value,
        "tensor(float16)": DataType.FLOAT16.value,
        "tensor(double)": DataType.FLOAT64.value,
        "tensor(uint32)": DataType.UINT32.value,
        "tensor(uint64)": DataType.UINT64.value,
        "tensor(complex64)": DataType.COMPLEX64.value,
        "tensor(complex128)": DataType.COMPLEX128.value,
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype={dtype} is not supported") from exc


class NodeArg:
    def __init__(self, name: str, type: int, shape: Sequence[Union[int, str]]):  # noqa: A002
        self.name = name
        self.type = type
        self.shape = shape

    def __str__(self):
        return (
            f"NodeArg(name='{self.name}', "
            f"type='{DataType(self.type).name}', shape={self.shape})"
        )


class ORTExecutorBase:
    def __init__(self, model):
        self._model = None
        self._sess = None
        self._input_names = None
        self._output_names = None
        self._input_nodes = None
        self._output_nodes = None
        self._providers = []
        self._model = load_model(model)

    def to(self, device: Union[str, Sequence[str]]):
        device = [device] if isinstance(device, str) else device
        assert isinstance(device, Sequence), (
            f"type(device) should be either str or Sequence[str], "
            f"but got {type(device)}."
        )

        providers = []
        for d in device:
            if d in ["cuda", "CUDAExecutionProvider"]:
                providers.append(self._cuda_provider())
            elif d in ["cpu", "CPUExecutionProvider"]:
                providers.append("CPUExecutionProvider")
            else:
                raise ValueError(f"Only support cpu or cuda devices, but got {d}.")
        self._providers = providers

        if self._sess is not None:
            self._sess.set_providers(self._providers)

        return self

    @classmethod
    def get_support_devices(cls):
        devices = []
        for p in cls._ort().get_available_providers():
            if p == "CPUExecutionProvider":
                devices.append("cpu")
            if p == "CUDAExecutionProvider":
                devices.append("cuda")
        return devices

    def _load_inputs_outputs(self):
        self._input_nodes = [
            NodeArg(i.name, ort_dtype_to_onnx_dtype(i.type), i.shape)
            for i in self._sess.get_inputs()
        ]
        self._output_nodes = [
            NodeArg(i.name, ort_dtype_to_onnx_dtype(i.type), i.shape)
            for i in self._sess.get_outputs()
        ]
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [i.name for i in self._sess.get_outputs()]

    def get_inputs(self):
        return self._input_nodes

    def get_outputs(self):
        return self._output_nodes

    def _prepare_input(self, inputs, check_type=False):
        inputs_dict = {}
        if isinstance(inputs, np.ndarray) and len(self._input_names) == 1:
            inputs_dict = {self._input_names[0]: inputs}
        if isinstance(inputs, dict):
            inputs_dict = inputs
        if isinstance(inputs, list):
            assert len(self._input_names) == len(inputs)
            for name_, input_ in zip(self._input_names, inputs):
                inputs_dict[name_] = input_

        if check_type:
            for idx_, name_ in enumerate(self._input_names):
                expected_type = DataType(self._input_nodes[idx_].type)
                actual_type = numpy_dtype_to_onnx_dtype(inputs_dict[name_].dtype)
                if expected_type != actual_type:
                    raise ValueError(
                        f"Unexpected data type for input: {name_} , "
                        f"Expected: {expected_type.name}, "
                        f"Actual: {actual_type.name} .",
                    )

        return inputs_dict

    def forward(self, inputs, output_names=None):
        """Feed the inputs into the model to calculate the outputs all at once.

        Returns:
            A dictionary, the key is the output name,
            and the value is the calculation result.
        """
        if output_names is None:
            output_names = self._output_names
        outputs = self._sess.run(
            output_names,
            self._prepare_input(inputs, check_type=True),
        )

        return dict(zip(output_names, outputs))

    def forward_with_batch(
        self,
        input_data,
        batch_size=1,
        output_names=None,
        progressbar=None,
    ):
        """Split the input into multiple batches and then calculate the outputs.

        Returns:
            A dictionary, the key is the output name, and
            the value is a list of calculation result.

        Note:
            If the model does not support inference with the specified
            batch_size, it will fall back to batch_size=1 for inference.
        """
        try:
            return self._forward_with_batch(
                input_data,
                batch_size=batch_size,
                output_names=output_names,
                progressbar=progressbar,
            )
        except Exception as exc:
            if batch_size == 1:
                logging.error(
                    "Model inference failed, check if the model "
                    "matches the given input!",
                )
                raise ValueError(
                    "onnx model inference failed even if batch_size == 1"
                ) from exc
            logging.info("Reset batch_size=1 and execute forward again...")
        # recreate session to free gpu memory allocated to old session
        # this statement cannot take effect in try-except
        self.create_session()
        return self._forward_with_batch(
            input_data,
            batch_size=1,
            output_names=output_names,
            progressbar=progressbar,
        )

    def _forward_with_batch(
        self,
        input_data,
        batch_size=1,
        output_names=None,
        progressbar=None,
    ):
        if output_names is None:
            output_names = self._output_names

        start = end = 0
        output_dict = {}
        for name in output_names:
            output_dict[name] = []

        def tqdm_(range_, progressbar=None):
            if progressbar is not None:
                return tqdm(range_, desc=progressbar)
            return range_

        for start in tqdm_(
            range(0, input_data.number_of_samples, batch_size),
            progressbar=progressbar,
        ):
            end += batch_size
            inputs = input_data.get_data(start=start, end=end)
            outputs = self._sess.run(
                output_names,
                self._prepare_input(inputs, check_type=(start == 0)),
            )

            for name, output in zip(output_names, outputs):
                output_dict[name].append(output)
            start = end

        return output_dict

    def inference(self, inputs):
        """Create session and then calculate the model outputs."""
        if self._sess is None:
            self.create_session()
        return self.forward(inputs)

    def inference_in_subprocess(self, inputs):
        """Inference the model in a subprocess."""

        def f(i, o):
            o["output"] = self.to("cpu").inference(i)

        manager = multiprocess.Manager()
        return_dict = manager.dict()
        p = multiprocess.Process(target=f, args=(inputs, return_dict))
        p.start()
        p.join()
        return return_dict["output"]
