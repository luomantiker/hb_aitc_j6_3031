#include "Kernel.h"
#include "cuda_runtime.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/LogicalResult.h"
#include "hbtl/Tensor.h"
#include <cstdint>
#include <iostream>

HBTL_NAMESPACE_BEGIN {

  LogicalResult Sigmoid(Tensor & fout, const Tensor &fin) {
    std::cout << "Running GPU Sigmoid\n"; // This is for test
    runByElementType<int8_t, int16_t, int32_t, int64_t, float>(fin.getType(), [&](auto iv) {
      using type = typename decltype(iv)::type;
      const auto *pFin = fin.getData<type>().data();
      auto *pFout = fout.getMutData<type>().data();
      auto byteSize = fin.getByteCapacity();
      void *pFinCuda;
      void *pFoutCuda;
      cudaMalloc((void **)&pFinCuda, byteSize);
      cudaMalloc((void **)&pFoutCuda, byteSize);
      cudaMemcpy(pFinCuda, pFin, byteSize, cudaMemcpyHostToDevice);
      auto n = fin.getElementSize();
      launchSigmoid<type>(reinterpret_cast<type *>(pFoutCuda), reinterpret_cast<type *>(pFinCuda), n);
      cudaMemcpy(pFout, pFoutCuda, byteSize, cudaMemcpyDeviceToHost);
    });
    return LogicalResult::success();
  }
}
HBTL_NAMESPACE_END
