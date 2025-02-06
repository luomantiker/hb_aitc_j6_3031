from typing import List, Optional, Sequence, Tuple

import numpy as np


def random_data(
    shape: Sequence[int],
    dtype: np.dtype = np.dtype("float32"),
    rtype: str = "uniform",
    interval: Optional[Sequence[int]] = None,
    mean_std: Sequence[float] = (0.0, 1.0),
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random data with specific shape, datatype, distribution, and seed.

    Args:
        shape: The shape of the generated random data.
        dtype: The data type of the generated random data.
        rtype: The type of random distribution, can be 'normal' or 'uniform'.
            When rtype='normal', the dtype should be np.dtype('float16'),
            np.dtype('float32') or np.dtype('float64').
        interval: Valid only when rtype is 'uniform'.
            Consists of two numbers, all generated data will be in the "half-open"
            interval [interval[0], interval[1]).
            The numerical range of the dtype must totally contain the interval.
            Default values:
                1. For float16/32/64: [-1.0, 1.0]
                2. For bool: [0, 2]
                3. For uint/int: [dtype's min_value, dtype's max_value + 1]
        mean_std: Valid only when rtype is 'normal'.
            Consists of two numbers, which are the mean and std of
            the Gaussian distribution.
        seed: The random seed.

    Returns:
        Generated random data.
    """
    if seed is not None:
        np.random.seed(seed)

    if rtype == "uniform":
        return _generate_uniform_data(shape, dtype, interval)
    if rtype == "normal":
        return _generate_normal_data(shape, dtype, mean_std)
    raise ValueError(f"Unsupported random distribution type: {rtype}")


def _generate_uniform_data(
    shape: Sequence[int], dtype: np.dtype, interval: Optional[Sequence[int]]
) -> np.ndarray:
    assert dtype in [
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("uint8"),
        np.dtype("int8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("uint32"),
        np.dtype("int32"),
        np.dtype("uint64"),
        np.dtype("int64"),
        np.dtype("bool"),
    ], f"Unsupported dtype for uniform distribution: {dtype}"

    if interval is None:
        interval = _default_interval(dtype)
    _check_interval(interval, dtype)

    if dtype in [np.dtype("float16"), np.dtype("float32"), np.dtype("float64")]:
        data = np.random.uniform(interval[0], interval[1], shape).astype(dtype)
    else:
        data = np.random.randint(interval[0], interval[1], shape).astype(dtype)
    return data


def _default_interval(dtype: np.dtype) -> List[int]:
    if dtype in [np.dtype("float16"), np.dtype("float32"), np.dtype("float64")]:
        interval = [-1.0, 1.0]
    elif dtype in [
        np.dtype("uint8"),
        np.dtype("int8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("uint32"),
        np.dtype("int32"),
        np.dtype("uint64"),
        np.dtype("int64"),
    ]:
        interval = [np.iinfo(dtype).min, np.iinfo(dtype).max + 1]
    else:
        # dtype == np.dtype("bool")
        interval = [0, 2]
    return interval


def _check_interval(interval: Sequence[int], dtype: np.dtype) -> None:
    if len(interval) != 2 or interval[0] >= interval[1]:
        raise ValueError(
            f"Interval must consist of two numbers with interval[0] < interval[1], "
            f"but got {interval}."
        )
    # check interval for float16/32/64
    if dtype in [
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("float64"),
    ] and not (
        interval[0] >= np.finfo(dtype).min and interval[1] <= np.finfo(dtype).max
    ):
        raise ValueError(
            f"When dtype={dtype}, interval must be contained in "
            f"[{np.finfo(dtype).min}, {np.finfo(dtype).max}]."
        )
    # check interval for uint/int
    if dtype in [
        np.dtype("uint8"),
        np.dtype("int8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("uint32"),
        np.dtype("int32"),
        np.dtype("uint64"),
        np.dtype("uint64"),
    ] and not (
        interval[0] >= np.iinfo(dtype).min and interval[1] <= np.iinfo(dtype).max + 1
    ):
        raise ValueError(
            f"When dtype={dtype}, interval must be contained in "
            f"[{np.iinfo(dtype).min}, {np.iinfo(dtype).max + 1}]."
        )
    # check interval for bool
    if dtype == np.dtype("bool") and not (interval[0] == 0 and interval[1] == 2):
        raise ValueError(f"When dtype={dtype}, interval must be [0, 2].")


def _generate_normal_data(
    shape: Sequence[int], dtype: np.dtype, mean_std: Sequence[float]
) -> np.ndarray:
    assert dtype in [
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("float64"),
    ], f"Unsupported dtype for normal distribution: {dtype}"

    if len(mean_std) != 2 or mean_std[1] < 0.0:
        raise ValueError(
            f"The length of mean_std should be 2 and mean_std[1] >= 0, "
            f"but got {mean_std}."
        )

    return (
        np.random.normal(mean_std[0], mean_std[1], shape)
        .clip(np.finfo(dtype).min, np.finfo(dtype).max)
        .astype(dtype)
    )


def random_data_with_nv12(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random YUV444 data and corresponding NV12 data.

    Args:
        shape: The shape of the generated random data.
            The shape should be (batch, height, width, channel).
            The channel should be 1 or 3.

    Returns:
        Generated YUV444 data and corresponding NV12 data.
    """
    idx_c = 3
    if len(shape) != 4 or shape[idx_c] not in [1, 3]:
        raise ValueError("For pyramid input, channel must be 1 or 3.")
    # yuv444_128 for quantized, nv12 for hybrid
    nv12_data = random_data(shape, np.dtype("uint8"), interval=[0, 256])
    if shape[idx_c] == 3:
        nv12_data[:, 0::2, 1::2, 1:] = nv12_data[:, 0::2, 0:-1:2, 1:]
        nv12_data[:, 1::2, 0::2, 1:] = nv12_data[:, 0:-1:2, 0::2, 1:]
        nv12_data[:, 1::2, 1::2, 1:] = nv12_data[:, 0:-1:2, 0:-1:2, 1:]
    yuv444_data = (nv12_data - 128).astype(np.dtype("int8"))
    return yuv444_data, nv12_data.astype(np.dtype("int8"))
