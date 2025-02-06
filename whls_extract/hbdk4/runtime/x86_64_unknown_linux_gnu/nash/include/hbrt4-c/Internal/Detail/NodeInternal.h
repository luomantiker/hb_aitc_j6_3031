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

struct Hbrt4NodeIAPI {
  Hbrt4Status (*getHardwarePerfMemspace)(Hbrt4Node node, Hbrt4Memspace *memspace);
};

extern const struct Hbrt4NodeIAPI HBRT4_NODE_IAPI;
