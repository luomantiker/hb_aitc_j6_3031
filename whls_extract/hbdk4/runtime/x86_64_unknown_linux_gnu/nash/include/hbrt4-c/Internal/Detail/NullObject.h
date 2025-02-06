/// Include all internal API headers here
/// Not part of public API. May change at any time

// IWYU pragma: private, include "hbrt4-c/Internal/AllInternal.h"

#pragma once

#if !defined(__clangd__) && !defined(HBRT4_INTERNAL_DETAIL_GUARD)
#error "This file should not be directly included. Include hbrt4-c/Internal/AllInternal.h instead"
#endif

#include "hbrt4-c/hbrt4-c.h"

#define HBRT4_PRIV_NULL_OBJECT                                                                                         \
  {                                                                                                                    \
    { 0, 0 }                                                                                                           \
  }

static inline Hbrt4Type hbrt4IAPITypeNullObj(void) {
  Hbrt4Type value = HBRT4_PRIV_NULL_OBJECT;
  return value;
}

static inline Hbrt4HbmHeader hbrt4IAPIHbmHeaderNullObj(void) {
  Hbrt4HbmHeader value = HBRT4_PRIV_NULL_OBJECT;
  return value;
}

static inline Hbrt4Instance hbrt4IAPIInstanceNullObj(void) {
  Hbrt4Instance value = HBRT4_PRIV_NULL_OBJECT;
  return value;
}

static inline Hbrt4PreInit hbrt4IAPIPreInitNullObj(void) {
  Hbrt4PreInit value = HBRT4_PRIV_NULL_OBJECT;
  return value;
}

#undef HBRT4_PRIV_NULL_OBJECT
