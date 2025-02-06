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
struct Hbrt4DescriptionIAPI {

  /// Create new string description
  /// `cstring` is NOT copied, and user should ensure it lives longer than `desc`
  Hbrt4Status (*newStringDesc)(const char *cstring, Hbrt4Description *desc);

  /// Create new binary description (laast byte must be NUL)
  /// `data` is NOT copied, and user should ensure it lives longer than `desc`
  Hbrt4Status (*newBinaryDesc)(Hbrt4ArrayRef data, Hbrt4Description *desc);

  /// Destroy the desc
  Hbrt4Status (*destroy)(Hbrt4Description *desc);
};

extern const struct Hbrt4DescriptionIAPI HBRT4_DESCRIPTION_IAPI;
