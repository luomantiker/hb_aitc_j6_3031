/// \file
/// \ref Variable Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Variable
/// @{

/// The enum for operator that may require special handling
/// \since v4.0.17
enum Hbrt4SpecialOperator {

  /// Unknown operator. Only used when there is an error
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_UNKNOWN = 0,

  /// A operator that currently do not require special handling
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_NORMAL = 1,

  /// A filter operator
  /// \todo Add comment for this enum
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_FILTER = 2,

  /// A rle operator
  /// \todo Add comment for this enum
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_RLE = 3,

  /// A detection post processing operator
  /// \todo Add comment for this enum
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_DPP = 4,

  /// A reduce argmax operator
  /// \todo Add comment for this enum
  /// \since v4.0.17
  HBRT4_SPECIAL_OPERATOR_ARGMAX = 5,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4SpecialOperator);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
