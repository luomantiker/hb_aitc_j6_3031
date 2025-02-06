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
struct Hbrt4TypeIAPI {

  /// Get a simple type by tag. User has no ownership of output
  Hbrt4Status (*simpleTypeByTag)(Hbrt4TypeTag tag, Hbrt4Type *type);
};

extern const struct Hbrt4TypeIAPI HBRT4_TYPE_IAPI;
