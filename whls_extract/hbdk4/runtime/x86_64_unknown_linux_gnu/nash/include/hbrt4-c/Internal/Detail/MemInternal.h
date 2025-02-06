/// Internal testing helper
/// Not part of public API
/// May change at any time

// IWYU pragma: private, include "hbrt4-c/Internal/AllInternal.h"

#pragma once

#if !defined(__clangd__) && !defined(HBRT4_INTERNAL_DETAIL_GUARD)
#error "This file should not be directly included. Include hbrt4-c/Internal/AllInternal.h instead"
#endif

#include "hbrt4-c/hbrt4-c.h"

// NOLINTNEXTLINE
typedef uint64_t Hbrt4BpuAddr;

struct Hbrt4MemIAPI {

  /// Allocate bpu memory with all bytes zero, which can be used by BPU
  Hbrt4BpuAddr (*allocBpuZero)(size_t size, bool enableCache);

  /// Flush cpu cached related to bpu memory, ensure all data on ddr
  void (*cleanBpuMemoryCache)(Hbrt4BpuAddr ptr, size_t size);

  /// Free bpu memory
  void (*freeBpuMemory)(Hbrt4BpuAddr ptr);
};

extern const struct Hbrt4MemIAPI HBRT4_MEM_IAPI;
