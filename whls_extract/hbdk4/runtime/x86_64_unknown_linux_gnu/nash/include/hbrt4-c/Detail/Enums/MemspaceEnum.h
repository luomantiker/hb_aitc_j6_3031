/// \file
/// \ref Memspace Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Memspace
/// @{

/// How is the memspace used
///
/// \attention
/// It is very important to fully understand this to avoid undefined behavior. \n
/// Violating these rules could cause bad bug that is hard to DEBUG!
///
/// #### General Rules
/// - All buffers whose memspace has different usage, **MUST NOT** overlap
/// - All buffers used by the same \ref Hbrt4Command, **MUST NOT** overlap
enum Hbrt4MemspaceUsage {

  /// Unknown usage. This value only outputted when API fails
  HBRT4_MEMSPACE_USAGE_UNKNOWN = 0,

  /// Memspace for graph input
  HBRT4_MEMSPACE_USAGE_GRAPH_INPUT = 1,

  /// Memspace for graph output
  HBRT4_MEMSPACE_USAGE_GRAPH_OUTPUT = 2,

  /// Memspace that contains constant,
  /// The associate buffer is managed by \ref hbrt4
  /// Get the buffer (address) by \ref hbrt4MemspaceGetConstantBuffer
  HBRT4_MEMSPACE_USAGE_CONSTANT = 3,

  /// Memspace is used for tensors that will be only related to one node,
  /// excluding model input/model outputs
  HBRT4_MEMSPACE_USAGE_TEMPORARY = 4,

  /// Memspace is used for tensors that is related to multiple nodes,
  /// This does not include memspace for model input/model outputs
  HBRT4_MEMSPACE_USAGE_INTERMEDIATE = 5,

  /// Used as cache between different commands of the same node
  /// Can be used concurrently by multiple commands
  ///
  /// The data of the memspace must be ONE of:
  /// 1. ALL ZERO
  /// 2. Modified only by command(s) of this node
  HBRT4_MEMSPACE_USAGE_NODE_CACHE = 6,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4MemspaceUsage);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
