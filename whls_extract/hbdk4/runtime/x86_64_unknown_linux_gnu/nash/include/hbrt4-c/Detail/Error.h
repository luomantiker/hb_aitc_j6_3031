/// \file
/// \ref Error Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Error Error
/// Get detailed error information,
/// user can check the status when an error occurs,
/// as well as the string description.
/// It can help user better locate error message.
///
/// # List of Features
/// - Provide detailed error message of the previous error
/// - The detail error message of any \ref hbrt4 API can be retrieved
///
/// # Task Sequence
/// - Whenever an HBRT4 APIs fails (not returning \ref HBRT4_STATUS_OK),
///   a \ref Hbrt4Error object can be retrieved by \ref hbrt4ErrorTakeLast
/// - The detailed message error can be retrieved from \ref Hbrt4Error by
///   \ref hbrt4ErrorGetCString
/// - When \ref Hbrt4Error is no longer used, it needs to deallocated by
///   \ref hbrt4ErrorDestroy
///
/// @{

/// Take the detail error of last failed \ref hbrt4 API called in the current thread,
/// which not returning \ref HBRT4_STATUS_OK
///
/// \do_not_test_in_j5_toolchain
/// \param[out] error The detailed error of most recent error in the current thread.
///
/// \on_err_out_null_obj
/// - If this API fails, the internal error stored in the thread will **NOT** be modified
///
/// \lifetime_ctor{hbrt4ErrorDestroy}
///
/// \mt_safe_atomic_mut
//
/// \note
/// - Use this API once, whenever an API error occurs to get the detail error \n
/// User can also choose to not call this API, which will **NOT** causing a memory leak
///
/// \invariant
/// - Any thread stores exactly one or zero detailed error, at any time,
/// \n which is updated whenever an API called in that thread
/// does not return \ref HBRT4_STATUS_OK
/// \n Including this API \ref hbrt4ErrorTakeLast itself
/// - When API failure occurs when there is already an error stored in the current thread,
/// the older error is discarded,
/// \n and then it is impossible to retrieve the older error by this API
/// - This API takes the ownership the most recent error stored in the current thread
/// \n So if this API succeeds, the current thread no longer store any error.
///
/// \internal
/// - If this API fails, store an error that wrap the previous error in the current thread.
///
/// \warning
/// - Error must be destructed by \ref hbrt4ErrorDestroy after use,
/// Memory leak otherwise
///
/// See examples below
///
/// #### Examples: Normal usage
/// Take error whenever the APIs fail
///
/// \code
/// // Code to acquired `Hbrt4Hbm hbm` omitted
/// Hbrt4Graph graph;
/// Hbrt4Status code = hbrt4HbmGetGraph(hbm, 9999 /* Out of bound */, &graph);
/// if (code != HBRT4_STATUS_OK) {
///   Hbrt4Error error;
///   code = hbrt4ErrorTakeLast(&error);
///   assert (code == HBRT4_STATUS_OK);
///   const char* error_string;
///   code = hbrt4ErrorGetCString(error, &error_string);
///   assert (code == HBRT4_STATUS_OK);
///   printf("Error message: %s\n", error_string);
///   hbrt4ErrorDestroy(&error);
/// }
/// // Calling hbrt4ErrorTakeLast again, when another error occurs
/// Hbrt4Status code = hbrt4HbmGetGraph(hbm, 9998 /* Out of bound */, &graph);
/// if (code != HBRT4_STATUS_OK) {
///   // hbm related API `hbrt4HbmGetGraph` fails.
///   // Get the detailed error from the `hbm` object
///   Hbrt4Error error;
///   code = hbrt4ErrorTakeLast(&error);
///   assert (code == HBRT4_STATUS_OK);
///   const char* error_string;
///   code = hbrt4ErrorGetCString(error, &error_string);
///   assert (code == HBRT4_STATUS_OK);
///   printf("Error message: %s\n", error_string);
///   hbrt4ErrorDestroy(&error);
/// }
///
/// \endcode
///
/// #### Examples: Call this API after success API calls, take most recent error
///
/// \code
/// // Code to acquired `Hbrt4Hbm hbm` omitted
/// Hbrt4Graph graph;
/// Hbrt4Status code = hbrt4HbmGetGraph(hbm, 9999 /* Out of bound */, &graph);
/// // First call to `hbrt4HbmGetGraph` fails
/// if (code != HBRT4_STATUS_OK) {
///   Hbrt4Status code2 = hbrt4HbmGetGraph(hbm, 0, &graph);
///   // Second call to `hbrt4HbmGetGraph` succeeds
///   assert (code2 == HBRT_STATUS_OK);
///   // Successful API calls operating on the object, does not affect the internal error stored in object
///   Hbrt4Error error;
///   code = hbrt4ErrorTakeLast(&error);
///   // Detail error caused by the failure of first `hbrt4HbmGetGraph` call is taken here
///   assert (code == HBRT4_STATUS_OK);
///   const char* error_string;
///   code = hbrt4ErrorGetCString(error, &error_string);
///   assert (code == HBRT4_STATUS_OK);
///   printf("Error message: %s\n", error_string);
///   hbrt4ErrorDestroy(&error);
/// }
/// \endcode
///
/// #### Examples: Call this API TWICE consecutively without other failed API in between, the second call of fails
///
/// \code
/// // Code to acquired `Hbrt4Hbm hbm` omitted
/// Hbrt4Graph graph;
/// Hbrt4Status code = hbrt4HbmGetGraph(hbm, 9999 /* Out of bound */, &graph);
/// if (code != HBRT4_STATUS_OK) {
///   Hbrt4Error error;
///   code = hbrt4ErrorTakeLast(&error);
///   // Detail error caused by the failure of `hbrt4HbmGetGraph` is taken here
///   assert (code == HBRT4_STATUS_OK);
///   hbrt4ErrorDestroy(&error);
///
///   Hbrt4Error error2;
///   code = hbrt4ErrorTakeLast(&error2);
///   // The second call to `hbrt4ErrorTakeLast` fails
///   // Because the detail error has been taken by the first `hbrt4ErrorTakeLast` call
///   assert (code == HBRT4_STATUS_NOT_FOUND);
/// }
/// \endcode
/// \returns
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_FOUND No detailed error stored internally in `object`
/// - \ret_ok
Hbrt4Status hbrt4ErrorTakeLast(Hbrt4Error *error) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the error status stored inside of the detail error
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{error}
/// \param[out] status The status enum code which is the return value of the API that introduces the detailed error.
/// See example for meaning
///
/// \on_err_out_set_to{HBRT4_STATUS_UNKNOWN}
/// \mt_safe
///
/// #### Example
/// \code
/// // Code to acquired `Hbrt4Hbm hbm` omitted
/// Hbrt4Graph graph;
/// Hbrt4Status code = hbrt4HbmGetGraph(hbm, 9999 /* Out of bound */, &graph);
/// if (code != HBRT4_STATUS_OK) {
///   Hbrt4Error error;
///   code = hbrt4ErrorTakeLast(&error);
///   // Detail error caused by the failure of `hbrt4HbmGetGraph` is taken here
///   assert (code == HBRT4_STATUS_OK);
///   hbrt4ErrorDestroy(&error);
///
///   Hbrt4Status error_code;
///   hbrt4ErrorGetStatus(error, &error_code);
///
///   // This `error_code` always equal to the code of APIs that introduces the detailed error
///   // In this case, the returned value of the failed `hbrt4HbmGetGraph` call
///   assert(error_code == code);
/// }
/// \endcode
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4ErrorGetStatus(Hbrt4Error error, Hbrt4Status *status) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the string representation of the detailed error
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{error}
/// \param[out] cstring Null terminated error string
///
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4ErrorGetCString(Hbrt4Error error, const char **cstring) HBRT4_PRIV_CAPI_EXPORTED;

/// Release the memory associate with the detailed error
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{error}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4ErrorDestroy(Hbrt4Error *error) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
