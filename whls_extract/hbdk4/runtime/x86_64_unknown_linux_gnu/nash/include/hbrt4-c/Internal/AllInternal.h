/// Include all internal API headers here
/// Not part of public API. May change at any time

#pragma once

#ifndef HBRT4_ENABLE_INTERNAL_API
#error "This file is for internal purpose. Not part of public API. Mainly used for testing purpose"
"No compatibility and API may change at any time. "
    "Define HBRT4_ENABLE_INTERNAL_API before use this. "
#endif

#define HBRT4_INTERNAL_DETAIL_GUARD

// IWYU pragma: begin_exports

#include "hbrt4-c/Internal/Detail/DescriptionInternal.h"
#include "hbrt4-c/Internal/Detail/Jit.h"
#include "hbrt4-c/Internal/Detail/MemInternal.h"
#include "hbrt4-c/Internal/Detail/MemspaceInternal.h"
#include "hbrt4-c/Internal/Detail/NodeInternal.h"
#include "hbrt4-c/Internal/Detail/NullObject.h"
#include "hbrt4-c/Internal/Detail/TypeInternal.h"

// IWYU pragma: end_exports

#undef HBRT4_INTERNAL_DETAIL_GUARD
