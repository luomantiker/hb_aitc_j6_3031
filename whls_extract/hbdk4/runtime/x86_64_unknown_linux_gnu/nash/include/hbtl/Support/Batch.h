// HBTL Shape checking

#pragma once
#ifndef HBTL_SUPPORT_BATCH_H_
#define HBTL_SUPPORT_BATCH_H_

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/Context.h"
#include <cstdint>

#include "hbtl/Core/TensorRef.h"
#include "hbtl/Support/ADTExtras.h"
#include "hbtl/Tensor.h"
#ifdef _OPENMP
#include <omp.h>
#endif
HBTL_NAMESPACE_BEGIN {

  // only use for generator coord for this cpp file, so no check
  HBTL_EXPORTED std::vector<int64_t> getCoord(int64_t cur, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides);

  /// `until` is the 1st non-batch axis. For example, until=-1 means the last axis is not batch, while all other axes
  /// are.
  template <typename Func, typename... Args>
  inline void runByVariadicBatch(int64_t until, const Tensor &t, Func func, Args &&...args) {
    if (until < 0) {
      until += t.getRank();
    }
    auto batchSizes = t.getSizes().take_front(until);
    auto strides = hbtl::getStrides(batchSizes, 1);
    for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
      func(getCoord(i, batchSizes, strides), std::forward<Args>(args)...);
    }
  }

  /// `until` is the 1st non-batch axis. For example, until=-1 means the last axis is not batch, while all other axes
  /// are.
  template <bool disableOMP, typename Func, typename... Args>
  inline void runByVariadicBatchOMP(int64_t until, const Tensor &t, Func func, Args &&...args) {
    if (until < 0) {
      until += t.getRank();
    }

    auto batchSizes = t.getSizes().take_front(until);
    auto strides = hbtl::getStrides(batchSizes, 1);
    if HBTL_CONSTEXPR_IF (disableOMP) {
      for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
        func(ArrayRef<int64_t>(getCoord(i, batchSizes, strides)), std::forward<Args>(args)...);
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
        func(ArrayRef<int64_t>(getCoord(i, batchSizes, strides)), std::forward<Args>(args)...);
      }
    }
  }

  /// `until` is the 1st non-batch axis. For example, until=-1 means the last axis is not batch, while all other axes
  /// are.
  template <bool disableOMP, typename T, size_t rank, typename Func, typename... Args>
  inline void runByVariadicBatchOMP(int64_t until, const TensorRef<T, rank> &t, Func func, Args &&...args) {
    if (until < 0) {
      until += t.getRank();
    }

    auto batchSizes = t.getSizes().take_front(until);
    auto strides = hbtl::getStrides(batchSizes, 1);
    if HBTL_CONSTEXPR_IF (disableOMP) {
      for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
        func(ArrayRef<int64_t>(getCoord(i, batchSizes, strides)), std::forward<Args>(args)...);
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
        func(ArrayRef<int64_t>(getCoord(i, batchSizes, strides)), std::forward<Args>(args)...);
      }
    }
  }

  /// `until` is the 1st non-batch axis. For example, until=-1 means the last axis is not batch, while all other axes
  /// are.
  template <bool disableOMP, typename Func, typename... Args>
  inline void runByVariadicBatchGroupOMP(int64_t until, const Tensor &t, Func func, Args &&...args) {
    if (until < 0) {
      until += t.getRank();
    }

    auto batchSizes = t.getSizes().take_front(until);
    auto strides = hbtl::getStrides(batchSizes, 1);
    if HBTL_CONSTEXPR_IF (disableOMP) {
      for (auto i = 0; i < vector::reduceMul(batchSizes); ++i) {
        func(ArrayRef<int64_t>(getCoord(i, batchSizes, strides)), std::forward<Args>(args)...);
      }
    } else {
      auto threadNums = Context::get()->getNumThreads();
      const auto numElems = vector::reduceMul(batchSizes);
      auto groupSize = (numElems - 1) / threadNums + 1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (auto i = 0; i < threadNums; ++i) {
        for (int j = 0; j < groupSize; ++j) {
          if (i * groupSize + j >= numElems) {
            break;
          }
          func(ArrayRef<int64_t>(getCoord(i * groupSize + j, batchSizes, strides)), std::forward<Args>(args)...);
        }
      }
    }
  }
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_BATCH_H_
