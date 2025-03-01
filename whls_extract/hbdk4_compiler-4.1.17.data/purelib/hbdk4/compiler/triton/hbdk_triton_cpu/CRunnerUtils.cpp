//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "CRunnerUtils.h"
#include "Msan.h"

#ifndef _WIN32
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#include <cstdlib>
#else
#include <alloca.h>
#endif
#include <sys/time.h>
#else
#include "malloc.h"
#endif // _WIN32

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

namespace {
template <typename V> void stdSort(uint64_t n, V *p) { std::sort(p, p + n); }

} // namespace

// Small runtime support "lib" for vector.print lowering.
// By providing elementary printing methods only, this
// library can remain fully unaware of low-level implementation
// details of our vectors. Also useful for direct LLVM IR output.
extern "C" void printI64(int64_t i) { fprintf(stdout, "%" PRId64, i); }
extern "C" void printU64(uint64_t u) { fprintf(stdout, "%" PRIu64, u); }
extern "C" void printF32(float f) { fprintf(stdout, "%g", f); }
extern "C" void printF64(double d) { fprintf(stdout, "%lg", d); }
extern "C" void printString(char const *s) { fputs(s, stdout); }
extern "C" void printOpen() { fputs("( ", stdout); }
extern "C" void printClose() { fputs(" )", stdout); }
extern "C" void printComma() { fputs(", ", stdout); }
extern "C" void printNewline() { fputc('\n', stdout); }

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg, UnrankedMemRefType<char> *dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;
  MLIR_MSAN_MEMORY_IS_INITIALIZED(src.sizes, rank * sizeof(int64_t));

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}

/// Prints GFLOPS rating.
extern "C" void printFlops(double flops) { fprintf(stderr, "%lf GFLOPS\n", flops / 1.0E9); }

/// Returns the number of seconds since Epoch 1970-01-01 00:00:00 +0000 (UTC).
extern "C" double rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

extern "C" void *mlirAlloc(uint64_t size) { return malloc(size); }

extern "C" void *mlirAlignedAlloc(uint64_t alignment, uint64_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#elif defined(__APPLE__)
  // aligned_alloc was added in MacOS 10.15. Fall back to posix_memalign to also
  // support older versions.
  void *result = nullptr;
  (void)::posix_memalign(&result, alignment, size);
  return result;
#else
  return aligned_alloc(alignment, size);
#endif
}

extern "C" void mlirFree(void *ptr) { free(ptr); }

extern "C" void mlirAlignedFree(void *ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

extern "C" void *rtsrand(uint64_t s) {
  // Standard mersenne_twister_engine seeded with s.
  return new std::mt19937(s);
}

extern "C" uint64_t rtrand(void *g, uint64_t m) {
  std::mt19937 *generator = static_cast<std::mt19937 *>(g);
  std::uniform_int_distribution<uint64_t> distrib(0, m);
  return distrib(*generator);
}

extern "C" void rtdrand(void *g) {
  std::mt19937 *generator = static_cast<std::mt19937 *>(g);
  delete generator;
}

#define IMPL_STDSORT(VNAME, V)                                                                                         \
  extern "C" void _mlir_ciface_stdSort##VNAME(uint64_t n, StridedMemRefType<V, 1> *vref) {                             \
    assert(vref);                                                                                                      \
    assert(vref->strides[0] == 1);                                                                                     \
    V *values = vref->data + vref->offset;                                                                             \
    stdSort(n, values);                                                                                                \
  }
IMPL_STDSORT(I64, int64_t)
IMPL_STDSORT(F64, double)
IMPL_STDSORT(F32, float)
#undef IMPL_STDSORT

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
