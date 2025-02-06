/// \file
/// \ref Description Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Description
/// @{

/// The category of the description
enum Hbrt4DescriptionCategory {
  /// Unknown category. Only used when there is an error
  HBRT4_DESCRIPTION_CATEGORY_UNKNOWN = 0,

  /// human-readable string
  HBRT4_DESCRIPTION_CATEGORY_STRING = 1,

  /// Not human-readable binary data
  HBRT4_DESCRIPTION_CATEGORY_BINARY = 2,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4DescriptionCategory);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
