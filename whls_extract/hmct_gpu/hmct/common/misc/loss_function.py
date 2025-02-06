from abc import ABC, abstractmethod
from typing import Callable, Iterable, Mapping, Sequence, Union

import numpy as np


class Loss(ABC):
    """The abstract base class for loss functions.

    The loss function is used to calculate the difference between the actual
    and desired value.

    The loss function can be used for different data types, including ndarray,
    iterable of ndarray, and mapping of ndarray or iterable of ndarray.
    """

    def run(
        self,
        actual: Union[
            np.ndarray,
            Iterable[np.ndarray],
            Mapping[str, Union[np.ndarray, Iterable[np.ndarray]]],
        ],
        desired: Union[
            np.ndarray,
            Iterable[np.ndarray],
            Mapping[str, Union[np.ndarray, Iterable[np.ndarray]]],
        ],
    ) -> float:
        """Run the loss calculation.

        Args:
            actual: The actual value.
            desired: The desired value.

        Returns:
            The loss value.
        """
        if isinstance(actual, Mapping) and isinstance(desired, Mapping):
            return self._run_mapping(actual, desired)
        if isinstance(actual, np.ndarray) and isinstance(desired, np.ndarray):
            return self._run_array(actual, desired)
        if isinstance(actual, Iterable) and isinstance(desired, Iterable):
            return self._run_iterable(actual, desired)

        raise TypeError(
            f"Both type(actual) and type(desired) should be one of "
            f"Mapping[str, np.ndarray | Iterable[np.ndarray]], np.ndarray and "
            f"Iterable[np.ndarray], but got {type(actual)} and {type(desired)}."
        )

    def _run_mapping(
        self,
        actual: Mapping[str, Union[np.ndarray, Iterable[np.ndarray]]],
        desired: Mapping[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> float:
        """Run the loss calculation for mapping."""
        assert set(actual.keys()) == set(desired.keys()), (
            f"Keys of actual and desired should be the same, "
            f"but got {set(actual.keys())} and {set(desired.keys())}."
        )

        # calculate loss for each key and return the average value
        loss_values = []
        for name in actual:
            if isinstance(actual[name], np.ndarray) and isinstance(
                desired[name], np.ndarray
            ):
                loss_values.append(self._run_array(actual[name], desired[name]))
            elif isinstance(actual[name], Iterable) and isinstance(
                desired[name], Iterable
            ):
                loss_values.append(self._run_iterable(actual[name], desired[name]))
            else:
                raise TypeError(
                    f"Both type(actual) and type(desired) should be "
                    f"Mapping of np.ndarray or Iterable[np.ndarray], but got "
                    f"Mapping of {type(actual[name])} and {type(desired[name])}."
                )
        assert loss_values, "Illegal empty mapping for loss calculation."
        return np.mean(loss_values)

    def _run_iterable(
        self, actual: Iterable[np.ndarray], desired: Iterable[np.ndarray]
    ) -> float:
        """Run the loss calculation for iterable."""
        # calculate loss for each ndarray and return the average value
        loss_values = []
        for actual_array, desired_array in zip(actual, desired):
            assert isinstance(actual_array, np.ndarray) and isinstance(
                desired_array, np.ndarray
            ), (
                f"Both actual and desired array should be np.ndarray, "
                f"but got {type(actual_array)} and {type(desired_array)}."
            )
            loss_values.append(self._run_array(actual_array, desired_array))
        assert loss_values, "Illegal empty iterable for loss calculation."
        return np.mean(loss_values)

    def _run_array(self, actual: np.ndarray, desired: np.ndarray) -> float:
        """Run the loss calculation for ndarray."""
        assert actual.shape == desired.shape, (
            f"Shape of actual and desired should be the same, "
            f"but got {actual.shape} and {desired.shape}."
        )
        return self._loss(actual.flatten(), desired.flatten())

    @abstractmethod
    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        """Calculate the loss value."""

    @property
    @abstractmethod
    def optimal(self) -> Callable[[Sequence[float]], int]:
        """Return the optimal function."""

    @staticmethod
    def create(name: str) -> "Loss":
        """Create a loss instance by name.

        Args:
            name: The name of the loss instance to create.

        Returns:
            The loss instance

        Raises:
            ValueError: If the loss instance name is not supported.
        """
        if name == "mse":
            return _MSE()
        if name == "mre":
            return _MRE()
        if name == "cosine-similarity":
            return _CosineSimilarity()
        if name == "sqnr":
            return _SQNR()
        if name == "chebyshev":
            return _Chebyshev()

        raise ValueError(f"Unsupported loss type: {name}.")


class _MSE(Loss):
    """Mean Squared Error loss function.

    The loss is calculated as the mean of the squared difference between the
    actual value and the desired value.

    The optimal value is the minimum value.

    Attributes:
        name: The name of the loss function.
    """

    def __init__(self) -> None:
        self.name = "mse"

    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        # When both actual and desired are empty, return the minimum mse.
        if actual.size == 0 and desired.size == 0:
            return 0.0
        return np.mean(np.square(actual - desired))

    @property
    def optimal(self) -> Callable[[Sequence[float]], int]:
        return np.argmin


class _MRE(Loss):
    """Mean Relative Error loss function.

    The loss is calculated as the mean of the absolute difference between the
    actual value and the desired value.

    The optimal value is the minimum value.

    Attributes:
        name: The name of the loss function.
    """

    def __init__(self) -> None:
        self.name = "mre"

    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        # When both actual and desired are empty, return the minimum mre.
        if actual.size == 0 and desired.size == 0:
            return 0.0
        return np.mean(np.abs(actual - desired))

    @property
    def optimal(self) -> Callable[[Sequence[float]], int]:
        return np.argmin


class _CosineSimilarity(Loss):
    """Cosine Similarity loss function.

    The loss is calculated as the cosine similarity between the actual value
    and the desired value.

    The optimal value is the maximum value.

    Attributes:
        name: The name of the loss function.
    """

    def __init__(self) -> None:
        self.name = "cosine-similarity"

    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        # When both actual and desired are empty, return the maximum cosine similarity.
        if actual.size == 0 and desired.size == 0:
            return 1.0

        actual_norm = np.linalg.norm(actual)
        desired_norm = np.linalg.norm(desired)
        # handle the case when both actual and desired are zero
        if actual_norm == 0.0 and desired_norm == 0.0:
            return 1.0
        # handle the case when either actual or desired is zero
        if actual_norm == 0.0 or desired_norm == 0.0:
            return 0.0

        return np.dot(actual, desired) / (actual_norm * desired_norm)

    @property
    def optimal(self) -> Callable[[Sequence[float]], int]:
        return np.argmax


class _SQNR(Loss):
    """Signal-to-Quantization-Noise Ratio loss function.

    The loss is calculated as the signal-to-quantization-noise ratio between the
    actual value and the desired value.

    The optimal value is the maximum value.

    Attributes:
        name: The name of the loss function.
    """

    def __init__(self) -> None:
        self.name = "sqnr"

    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        # When both actual and desired are empty, return the maximum sqnr.
        if actual.size == 0 and desired.size == 0:
            return float("+inf")

        signal = np.linalg.norm(actual)
        noise = np.linalg.norm(actual - desired)
        # handle the case when both signal and noise are zero
        if noise == 0.0 and signal == 0.0:
            return 0.0
        # handle the case when either signal or noise is zero
        if signal == 0.0:
            return float("-inf")
        if noise == 0.0:
            return float("+inf")

        return 20.0 * np.log10(np.sqrt(signal) / np.sqrt(noise))

    @property
    def optimal(self) -> Callable[[Sequence[float]], int]:
        return np.argmax


class _Chebyshev(Loss):
    """Chebyshev loss function.

    The loss is calculated as the maximum absolute difference between the actual
    value and the desired value.

    The optimal value is the minimum value.

    Attributes:
        name: The name of the loss function.
    """

    def __init__(self) -> None:
        self.name = "chebyshev"

    def _loss(self, actual: np.ndarray, desired: np.ndarray) -> float:
        # When both actual and desired are empty, return the minimum chebyshev.
        if actual.size == 0 and desired.size == 0:
            return 0.0
        return np.max(np.abs(actual - desired))

    @property
    def optimal(self) -> Callable[[Sequence[float]], int]:
        return np.argmin
