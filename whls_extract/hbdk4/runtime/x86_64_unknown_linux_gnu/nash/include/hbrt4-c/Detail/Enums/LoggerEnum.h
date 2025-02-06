/// \file
/// \ref Logger Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Logger
/// @{

/// The log level to control log verbosity.
/// This matches the log level of libhblog
enum Hbrt4LogLevel {

  /// Print all logs. The log will be very verbose. Expected only used by developer
  HBRT4_LOG_LEVEL_TRACE = 0,

  /// Use it only during debug. The log will be verbose.
  HBRT4_LOG_LEVEL_DEBUG = 1,

  /// Default log level. Use this in production
  HBRT4_LOG_LEVEL_INFO = 2,

  /// Errors and warnings are show
  HBRT4_LOG_LEVEL_WARN = 3,

  /// Errors are shown
  HBRT4_LOG_LEVEL_ERROR = 4,

  // Only errors that very bad are shown
  // \internal Removed because Rust `tracing` crate currently do not support this
  // HBRT4_LOG_LEVEL_CRITICAL = 5,

  /// Use this log level to suppress all logs
  HBRT4_LOG_LEVEL_OFF = 6,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4LogLevel);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
