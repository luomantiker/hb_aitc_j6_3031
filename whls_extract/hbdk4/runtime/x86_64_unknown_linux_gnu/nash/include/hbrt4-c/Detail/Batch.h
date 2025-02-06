/// \file
/// \ref Batch Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \ingroup Graph
///
/// \defgroup Batch Batch
/// Batch related information of graph or tensor
///
/// @{

/// To specify which batch(es) the memspace or tensor correspond to.
///
/// #### Explanation
///
///
/// #### Default Value {#Hbrt4BatchRange_DefaultValue}
/// - All fields are set to zero
struct Hbrt4BatchRange {
  /// DOC TODO
  /// \internal `numpy` uses `size_t` for this.
  /// See <https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.NPY_AO.dimensions>
  size_t begin;
  /// DOC TODO
  size_t end;
};

/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4BatchRange);
/// \endcond

/// @}

HBRT4_PRIV_C_EXTERN_C_END
