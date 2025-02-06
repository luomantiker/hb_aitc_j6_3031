/// \file
/// \ref Status Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/StatusEnum.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Status Status
/// Simple enum to indicate the status of the HBRT API.
/// If the status is abnormal, the category of status is returned for debug
/// This status enum is the returned type of any HBRT API
///
/// # List of Features
/// - Simple enum for API error status
/// - Provide string representation of status
///
/// # Task Sequence
/// This enum can be used for simple error handling
/// - Get the status enum from the return value of any HBRT API,
/// the enum value can be used for error handling.
/// - The the string representation of the enum by \ref hbrt4StatusGetCString
/// for logging or debugging purpose
///
/// See the API of \ref Error Module if more detailed error message is needed
///
///
///
/// @{

/// Get the printable cstring representation of the \ref Hbrt4Status
/// \param[in] status The status enum
///
/// \invariant
/// Guarantee return value is never null or empty cstring
///
/// \test
/// Test the output for all enum values defined in \ref Hbrt4Status
/// \test
/// Test the output for values not defined in \ref Hbrt4Status, such as -1, `INT32_MAX`, `INT32_MIN`
///
/// #### Examples
/// - "HBRT4_STATUS_IO_ERROR" <- \ref hbrt4StatusGetCString(\ref HBRT4_STATUS_IO_ERROR)
/// - "HBRT4_STATUS_UNKNOWN" <- \ref hbrt4StatusGetCString(\ref HBRT4_STATUS_UNKNOWN)
/// - "HBRT4_STATUS_NOT_RECOGNIZED" <- \ref hbrt4StatusGetCString(\ref Hbrt4Status(12345678 /* Some random int */))
///
/// \returns
/// - The string form of the enum key, if `status` is a recognized enum value defined in \ref Hbrt4Status
/// - Otherwise, return "HBRT4_STATUS_NOT_RECOGNIZED"
const char *hbrt4StatusGetCString(Hbrt4Status status) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
