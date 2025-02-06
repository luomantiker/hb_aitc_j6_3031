from typing import Sequence


def get_kernel_stride(input_size, output_size):
    if isinstance(input_size, Sequence):
        kernel = []
        stride = []
        for i, o in zip(input_size, output_size):
            k, s = get_kernel_stride(i, o)
            kernel.append(k)
            stride.append(s)
    else:
        if output_size is None:
            output_size = input_size

        if input_size % output_size != 0:
            raise ValueError(
                "Only support the case that input size can be "
                "divided equally by output size, but give "
                "input size {} and output size {}".format(
                    input_size, output_size
                )
            )
        else:
            stride = input_size // output_size
            kernel = input_size - (output_size - 1) * stride

    return kernel, stride
