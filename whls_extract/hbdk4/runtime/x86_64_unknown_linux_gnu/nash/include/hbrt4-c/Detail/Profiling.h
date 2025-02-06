/// \file
/// \ref Profiling Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/ProfilingEnum.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Profiling Pipeline
///
/// Code to parse profiling data
///
/// The profiling data consists of multiple profiling entries
///
/// Each entry consists of exactly one struct as below.
/// All entries are located consecutively inside memory
///
/// Each entry or struct begins with \ref Hbrt4ProfilingHeader
///
/// User should use `Hbrt4ProfilingHeader.category` to determine the
/// actual struct used to parse the profiling entry.
/// User should validate `Hbrt4ProfilingHeader.byte_size` to detect memory error,
/// which be equal to `sizeof(struct)`
///
/// For example, if `Hbrt4ProfilingHeader.category` equals to
/// \ref HBRT4_PROFILING_CATEGORY_SIMPLE_TIME,
/// user should validate `Hbrt4ProfilingHeader.byte_size` ==
/// sizeof(Hbrt4ProfilingSimpleTime), then use the semantic of
/// \ref Hbrt4ProfilingSimpleTime to parse the data
///
/// Then user should continue to parse other profiling entries,
/// assuming all entries are located consecutively inside memory
///
///
/// @{

/// Header to indicate which struct to use to parse the profiling entry
struct Hbrt4ProfilingHeader {
  Hbrt4ProfilingCategory
      category;     ///< Category which determines the exact struct used to parse the current profiling event
  size_t byte_size; ///< Should always equal to the byte size of current profiling event,
                    ///< including this \ref Hbrt4ProfilingHeader
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4ProfilingHeader);
/// \endcond

/// Simple profiling entry with basic timestamp and tag information
struct Hbrt4ProfilingSimpleTime {

  /// Expect `header.category == HBRT4_PROFILING_CATEGORY_SIMPLE_TIME`
  /// Expect `header.byte_size == sizeof(Hbrt4ProfilingSimpleTime)`
  Hbrt4ProfilingHeader header;

  /// Value in cpu `CYCLE` register
  uint64_t cpu_cycle;

  /// Value in cpu `TIME` register
  uint64_t cpu_timer;

  /// Value in cpu `INST_RET` register
  uint64_t cpu_num_insts;

  /// Whole second part of timestamp of profiling event
  uint64_t second;

  /// Sub seconde part of timestamp, in nanosecond
  uint32_t nano_second;

  /// String tag for the current event. This field does not need to be null terminated due to the existence of `nul`
  /// field
  char tag[27];

  /// This field always equal to 0
  char nul;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4ProfilingSimpleTime);
/// \endcond

/// @}

HBRT4_PRIV_C_EXTERN_C_END
