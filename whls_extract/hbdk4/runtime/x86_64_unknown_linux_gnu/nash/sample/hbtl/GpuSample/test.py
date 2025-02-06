import sys
import ctypes
import torch
import numpy as np
from hbdk4.compiler import Module
from hbdk4.compiler._mlir_libs._hbdk import Dispatcher


def main():
    lib_hbtl_cuda_kernel = sys.argv[1]  # Path to libHBTLCudaKernel.so
    _mylib = ctypes.cdll.LoadLibrary(lib_hbtl_cuda_kernel)
    Dispatcher.get().load(lib_hbtl_cuda_kernel)

    mlir_text = """
    func.func @main(%arg0 : tensor<2x2x32xf32>) -> tensor<2x2x32xf32>{
        %0 = "hbir.sigmoid"(%arg0) : (tensor<2x2x32xf32>) -> tensor<2x2x32xf32>
        return %0 : tensor<2x2x32xf32>
    }
    """

    module = Module.parse(mlir_text)
    print(module)

    torch.manual_seed(0)

    input = torch.rand(2, 2, 32)
    print(input)
    func = module[0]
    func.session.backend = "UCP"
    result_dnn = func(input)
    np.testing.assert_equal(len(result_dnn), 1)
    print("result_dnn = ", result_dnn)

    # TODO(weiguozhao): Verify the result is correct
    func.session.backend = "SILVERC"
    result_silver = func(input)
    np.testing.assert_equal(len(result_silver), 1)
    print("result_silver = ", result_silver)
    np.testing.assert_allclose(result_dnn[0], result_silver[0], rtol=1e-05)


if __name__ == "__main__":
    main()
