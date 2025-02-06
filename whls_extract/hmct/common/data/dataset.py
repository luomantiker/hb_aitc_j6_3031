import hashlib
import logging
import os
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)

import numpy as np

from hmct.ir import DataType, onnx_dtype_to_numpy_dtype
from hmct.utility import random_data


class Dataset:
    """Dataset is used to parse and store data.

    The stored data will be represented as a dict, the key is input name,
    and the value is a list of data, e.g.
        {"input_name1": [data1_0, data1_1, data1_2...],
         "input_name2": [data2_0, data2_1, data2_2...]}
    The data in the list should be numpy.ndarray. The length of the list
    should be the same, and the data in the same index of the list should
    be corresponding to each other.

    The stored data can be get via `get_data` function.

    The stored data can be saved to the disk via `save` function.

    The stored data can be copied to a new Dataset with specified
    number of samples via `copy` function.

    Attributes:
        number_of_samples: The number of samples in the dataset.
    """

    @overload
    def __init__(
        self,
        input_data: Mapping[str, Union[Iterable[np.ndarray], str]],
    ): ...

    @overload
    def __init__(
        self,
        *,
        input_shapes: Mapping[str, Sequence[int]],
        input_dtypes: Mapping[str, DataType],
    ): ...

    def __init__(
        self,
        input_data: Optional[Mapping[str, Union[Iterable[np.ndarray], str]]] = None,
        input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        input_dtypes: Optional[Mapping[str, DataType]] = None,
    ):
        """Initialize the Dataset.

        Either input_data or input_shapes/input_dtypes should be given.
        If input_data is None, random data with given input_shapes/input_dtypes
        will be generated and used instead.

        Args:
            input_data: Dict of input data, the key is input_name, the value
                should be iterable object of ndarray or directory containing ndarray.
            input_shapes: Dict of input shapes, the key is input_name, the value
                should be the shape of the input.
            input_dtypes: Dict of input dtypes. The key is input_name, the value
                should be the dtype of the input.
        """
        self._input_data: Dict[str, List[np.ndarray]] = {}
        self.number_of_samples: int = 0

        if input_data is None:
            assert (
                input_shapes is not None and input_dtypes is not None
            ), "input_shapes/input_dtypes should be given when input_data is None."
            input_data = self._generate_random_data(
                input_shapes,
                input_dtypes,
            )

        self._load_data(input_data)

    def _generate_random_data(
        self,
        input_shapes: Mapping[str, Sequence[int]],
        input_dtypes: Mapping[str, DataType],
    ) -> Dict[str, List[np.ndarray]]:
        """Generate random data with specified shape and dtype."""
        assert len(input_shapes) == len(
            input_dtypes
        ), "input_shapes and input_dtypes should have the same length."

        input_data: Dict[str, List[np.ndarray]] = {}
        for input_name, input_shape in input_shapes.items():
            input_data[input_name] = [
                random_data(
                    input_shape, onnx_dtype_to_numpy_dtype(input_dtypes[input_name])
                )
            ]
        return input_data

    def _load_data(
        self, input_data: Mapping[str, Union[Iterable[np.ndarray], str]]
    ) -> None:
        """Load data from input_data."""
        for name, data in input_data.items():
            if isinstance(data, str):
                self._load_data_from_path(name, data)
            elif isinstance(data, Iterable):
                self._load_data_from_iterable(name, data)
            else:
                raise TypeError(
                    f"Type of data should be str or iterable object,"
                    f"but got {type(data)}."
                )

        # Calculate and check number of samples from the input_data received.
        for name, data in self._input_data.items():
            number_of_samples = len(data)
            logging.debug(f"input name: {name}, number_of_samples: {number_of_samples}")
            if not self.number_of_samples:
                self.number_of_samples = number_of_samples
            else:
                if self.number_of_samples != number_of_samples:
                    raise ValueError(
                        f"Input {name} received wrong number of samples. "
                        f"Previous number of samples: {self.number_of_samples}, "
                        f"Input {name}'s number of samples: {number_of_samples}"
                    )
        logging.info(f"There are {self.number_of_samples} samples in the dataset.")

    def _load_data_from_iterable(self, name: str, data: Iterable[np.ndarray]) -> None:
        """Load data from iterable object.

        Args:
            name: The input name
            data: The iterable object to load data
        """
        data = list(data)
        for data_item in data:
            if not isinstance(data_item, np.ndarray):
                raise TypeError(
                    f"Type of data item received in the iterable object "
                    f"should be np.ndarray, but got {type(data_item)}."
                )
        self._input_data[name] = data

    def _load_data_from_path(self, name: str, path: str) -> None:
        """Load data from directory.

        Args:
            name: The input name
            path: The directory containing ndarray
        """
        if os.path.isdir(path):
            self._input_data[name] = [
                np.load(os.path.join(path, f)) for f in sorted(os.listdir(path))
            ]
        else:
            raise ValueError(f"{path} is not a directory.")

    @property
    def md5(self) -> str:
        """Return the md5 of the dataset."""
        md5 = hashlib.md5()
        for data in self._input_data.values():
            for data_item in data:
                md5.update(data_item.tobytes())
        return md5.hexdigest()

    def get_data(
        self,
        input_names: Optional[Sequence[str]] = None,
        start: int = 0,
        end: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Get data from the dataset.

        Args:
            input_names: The input names to get data.
                if None, all inputs will be returned.
            start: The start index of the data to get.
            end: The end index of the data to get.

        Returns:
            Dict of sliced data, the key is input name, the value is
            the concatenated data.
        """
        end = min(end, self.number_of_samples)
        start = min(start, end)

        sliced_data: Dict[str, np.ndarray] = {}
        if input_names is None:
            for name in self._input_data:
                sliced_data[name] = np.concatenate(
                    self._input_data[name][start:end], axis=0
                )
        else:
            for name in input_names:
                if name in self._input_data:
                    sliced_data[name] = np.concatenate(
                        self._input_data[name][start:end], axis=0
                    )
                else:
                    raise ValueError(f"There exists no input {name} in the dataset.")
        return sliced_data

    def save(self, path: str) -> None:
        """Save the dataset to the given path."""
        for name, data in self._input_data.items():
            saved_path = os.path.join(path, name)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            for data_idx, data_item in enumerate(data):
                np.save(f"{saved_path}/{data_idx}.npy", data_item)

    def copy(self, number_of_samples: int) -> "Dataset":
        """Return a subset of the dataset with specified number_of_samples."""
        number_of_samples = min(self.number_of_samples, number_of_samples)
        return Dataset(
            input_data={
                name: data[-number_of_samples:]
                for name, data in self._input_data.items()
            }
        )
