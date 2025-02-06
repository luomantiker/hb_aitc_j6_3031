/// \file
/// \ref Graph Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup GraphGroup
/// @{

/// Classification to show why graphs in the graph group are grouped together
enum Hbrt4GraphGroupClassification {

  /// Unknown category
  /// This value outputted only when \ref hbrt4TypeGetTag errors
  HBRT4_GRAPH_GROUP_CLASSIFICATION_UNKNOWN = 0,

  /// Graphs in group share same graph structure,
  /// and their only difference is the batch number
  HBRT4_GRAPH_GROUP_CLASSIFICATION_BATCH = 1,

  /// Group only has one single graph
  HBRT4_GRAPH_GROUP_CLASSIFICATION_SINGLE = 2,

};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4GraphGroupClassification);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
