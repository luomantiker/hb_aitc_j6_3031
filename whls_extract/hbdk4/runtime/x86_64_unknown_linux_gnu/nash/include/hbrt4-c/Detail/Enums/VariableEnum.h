/// \file
/// \ref Type Module
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

/// How is variable used by \ref Hbrt4Node,
enum Hbrt4VariableNodeUsage {

  /// Unknown usage. This value is outputted only when API fails
  HBRT4_VARIABLE_NODE_USAGE_UNKNOWN = 0,

  /// Variable is used as input of the node
  HBRT4_VARIABLE_NODE_USAGE_INPUT = 1,

  /// Variable is used as output of the node
  HBRT4_VARIABLE_NODE_USAGE_OUTPUT = 2,

  /// Variable is used as constant of the node
  HBRT4_VARIABLE_NODE_USAGE_CONSTANT = 3,

  /// Non-constant, Non-input, Non-output data,
  /// Only used by this node, but not by other nodes
  HBRT4_VARIABLE_NODE_USAGE_TEMPORARY = 4,

  /// Same usage as \ref HBRT4_MEMSPACE_USAGE_NODE_CACHE
  HBRT4_VARIABLE_NODE_USAGE_NODE_CACHE = 6,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4VariableNodeUsage);

/// The semantic of variable,
/// when it is used as the input of some computation,
/// to ease data processing of user
enum Hbrt4VariableInputSemantic {
  /// Only use this value when error occurs
  HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN = 0,

  /// No special handling needed
  HBRT4_VARIABLE_INPUT_SEMANTIC_NORMAL = 1,

  /// Tuple for pyramid input
  HBRT4_VARIABLE_INPUT_SEMANTIC_PYRAMID = 2,

  /// Tuple for Resizer input
  HBRT4_VARIABLE_INPUT_SEMANTIC_RESIZER = 3,

  /// Y channel of Pyramid/resizer
  HBRT4_VARIABLE_INPUT_SEMANTIC_IMAGE_Y = 4,

  /// UV channel of Pyramid/Resizer
  HBRT4_VARIABLE_INPUT_SEMANTIC_IMAGE_UV = 5,

  /// ROI of image, for resizer input，a 1x4 int32 tensor
  /// such as [[start_w, start_h, end_w, end_h]]
  HBRT4_VARIABLE_INPUT_SEMANTIC_IMAGE_ROI = 6,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4VariableInputSemantic);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
