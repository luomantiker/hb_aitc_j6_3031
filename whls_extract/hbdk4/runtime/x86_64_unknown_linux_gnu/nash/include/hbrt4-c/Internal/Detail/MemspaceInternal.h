/// Internal testing helper
/// Not part of public API
/// May change at any time

// IWYU pragma: private, include "hbrt4-c/Internal/AllInternal.h"

#pragma once

#if !defined(__clangd__) && !defined(HBRT4_INTERNAL_DETAIL_GUARD)
#error "This file should not be directly included. Include hbrt4-c/Internal/AllInternal.h instead"
#endif

#include "hbrt4-c/hbrt4-c.h"

/// Function pointers to internal APIs
struct Hbrt4MemspaceIAPI {

  /// Create new data memspace
  Hbrt4Status (*newDataMemspace)(const char *name, size_t alignment, size_t size, Hbrt4Memspace *memspace);

  /// Destroy the memspace
  Hbrt4Status (*destroy)(Hbrt4Memspace *memspace);
};

extern const struct Hbrt4MemspaceIAPI HBRT4_MEMSPACE_IAPI;
