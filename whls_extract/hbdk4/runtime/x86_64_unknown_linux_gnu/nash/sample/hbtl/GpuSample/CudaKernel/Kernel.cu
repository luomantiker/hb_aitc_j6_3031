#include "Kernel.h"
#include <iostream>

#include "cuda_runtime.h"

HBTL_NAMESPACE_BEGIN {

  template <typename T> __global__ void sigmoid(T * fout, const T *fin, int64_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += gridStride) {
      fout[i] = 1 / (1 + __expf(static_cast<float>(-fin[i])));
    }
  }

  template <typename T> void launchSigmoid(T * fout, const T *fin, int64_t n) {
    int numSMs = 2;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int blockSize = 32;
    // It's better to set it as a multiple of 32.
    dim3 block(blockSize);
    dim3 grid(numSMs * 2);

    // Explicitly submit kernel to stream matching cudaContext to avoid issues
    sigmoid<T><<<grid, block, 0, 0>>>(fout, fin, n);
  }

// just example for template, sigmoid only need float type
#define REGSIGMOID(T) template void launchSigmoid(T *fout, const T *fin, int64_t n)
  REGSIGMOID(int8_t);
  REGSIGMOID(int16_t);
  REGSIGMOID(int32_t);
  REGSIGMOID(int64_t);
  REGSIGMOID(float);
#undef REGSIGMOID
}
HBTL_NAMESPACE_END
