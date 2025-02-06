/// \file
/// Main header for HBRT C++ wrapper.
///
/// \warning This should be the only C++ HBRT header file to be directly
/// included by the users. Directly include any header file in the
/// \c Detail subdirectory is not allowed.
///
/// C++ APIs are header only.
/// Currently C++ API is only used internally by compiler team only
/// No ABI/API compatibility will be considered

#pragma once

#ifndef __cplusplus
#error "Should only include this file in C++ code"
#endif

#include "hbrt4-c/hbrt4-c.h"

// Must define this macro to include headers in the `detail` directory
#define HBRT4_DETAIL_GUARD

#include "hbrt4/Detail/Object.hpp"

#undef HBRT4_DETAIL_GUARD
