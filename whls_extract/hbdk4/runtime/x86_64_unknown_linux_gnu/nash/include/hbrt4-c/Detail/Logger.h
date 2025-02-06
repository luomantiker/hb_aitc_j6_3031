/// \file
/// \ref Logger Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/LoggerEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"
#include <sys/types.h>

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Logger Logger
/// Logging utilities to print running status and information, support multiple log levels
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4Logger to be alive in the entire process
///
/// # List of Features
/// - Provide API to receive logging message for the logging system
///
/// # Task Sequence
///
/// User should register the logger in the following sequence:
/// - Initialize \ref Hbrt4LoggerInitializer struct,
///   set the handling function and log level inside it
/// - Create logger with \ref Hbrt4LoggerInitializer, by \ref hbrt4LoggerCreate
/// - Register the logger, while initializing \ref Hbrt4Instance, by
///   \ref hbrt4InstanceBuilderRegisterLogger
/// - When logger is no longer used, it should be destroyed by \ref hbrt4LoggerDestroy.
///   This should be done after the destruction of \ref Hbrt4Instance
///
/// @{

/// Handler to receive the log message
///
/// \warning
/// This handler may be called in multiple threads
/// It is the user's responsibility to ensure this handler is thread-safe
///
/// \param[in] message Message to write in this function call \n
/// This is **NOT** null-terminated
/// \param[in] moduleName The null-terminated module name of the log. This is never `NULL` pointer
/// \param[in] level: The log level of the current message
/// \param[in] messageId Message may be sent with multiple calls of this API \n
/// Message with the same message id correspond to the same message
/// \param[in] logData Reserved for future extension
/// \param[in] userData The userdata set in \ref Hbrt4LoggerInitializer, for user's custom purpose
/// \return
/// - On success, return number of bytes processed
/// - On failure, return negative number whose absolute value corresponding to a Unix error code
/// \note
/// HBRT **NEVER** abort or panic regardless this handler returns success or not
typedef ssize_t (*Hbrt4LoggerHandler)(Hbrt4ArrayRef message, const char *moduleName, Hbrt4LogLevel level,
                                      uint64_t messageId, Hbrt4LoggerData logData, void *userData);

/// Struct to initialize the logger
struct Hbrt4LoggerInitializer {
  /// Minimum log level to filter the log
  ///
  /// Recommend to set this to \ref HBRT4_LOG_LEVEL_INFO for normal usage
  Hbrt4LogLevel level;
  /// User define handler to process log data
  Hbrt4LoggerHandler handler;
  /// User custom data to be used in `handler`
  void *handlerUserdata;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4LoggerInitializer);
/// \endcond

/// Create a logger from initializer structure
///
/// \param[in] initializer to create the logger
/// \param[out] logger Newly created logger object
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4LoggerDestroy}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4Logger to be alive in the entire process
///
/// \returns
/// - \ref HBRT4_STATUS_INVALID_ARGUMENT `handler` in `initializer` not set,
/// or `level` in `initializer` is invalid
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_SUPPORTED An \ref Hbrt4Logger already exists before this API completes
/// - \ret_ok
Hbrt4Status hbrt4LoggerCreate(Hbrt4LoggerInitializer initializer, Hbrt4Logger *logger) HBRT4_PRIV_CAPI_EXPORTED;

/// Destroy the logger
///
/// \param_dtor{logger}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION \ref Hbrt4Instance using the logger is still alive
/// - \ret_ok
Hbrt4Status hbrt4LoggerDestroy(Hbrt4Logger *logger) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
