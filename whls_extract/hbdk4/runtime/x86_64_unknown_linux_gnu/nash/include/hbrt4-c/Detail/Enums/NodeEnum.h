/// \file
/// \ref Node Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Node
/// @{

/// Tag to indicate node is pseudo
///
/// Pseudo node does not to be executed.
///
enum Hbrt4NodePseudoTag {
  /// Unknown tag. This value only outputted when API fails
  HBRT4_NODE_PSEUDO_TAG_UNKNOWN = 0,

  /// The function pack inputs into tuple
  /// This function has no side effect
  HBRT4_NODE_PSEUDO_TAG_TUPLE_PACK = 1,

  /// The function unpacks tuple
  /// This function has no side effect
  HBRT4_NODE_PSEUDO_TAG_TUPLE_UNPACK = 2,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4NodePseudoTag);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
