/// \file
/// \ref Profiling Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Profiling
/// @{

/// Enum used to indicate the exact struct used to parse the current profiling entry
enum Hbrt4ProfilingCategory {
  /// Corresponds to the \ref Hbrt4ProfilingSimpleTime struct
  HBRT4_PROFILING_CATEGORY_SIMPLE_TIME = 1,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4ProfilingCategory);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
