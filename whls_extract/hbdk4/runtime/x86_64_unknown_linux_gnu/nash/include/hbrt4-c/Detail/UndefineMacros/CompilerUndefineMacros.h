/// \file
/// Undefine all private macros in Compiler.h to avoid macro leaks
///
/// \warning
/// Should include this at the end of "hbrt/hbrt4-c.h"
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c
///
/// \private_api

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

// Include this to ensure user does not include this file directly
#include "hbrt4-c/Detail/Compiler.h"

#undef HBRT4_PRIV_C_STRICT_PROTOTYPES_BEGIN
#undef HBRT4_PRIV_C_STRICT_PROTOTYPES_END

#undef HBRT4_PRIV_C_EXTERN_C_BEGIN
#undef HBRT4_PRIV_C_EXTERN_C_END

// HBRT4_PRIV_DOXYGEN is for doxygen. Do not undefine it.

#undef HBRT4_PRIV_CAPI_EXPORTED

#undef HBRT4_PRIV_TYPEDEF_ENUM

#undef HBRT4_PRIV_TYPEDEF_STRUCT

#undef HBRT4_PRIV_WARN_UNUSED_RESULT

#undef HBRT4_DEPRECATED
