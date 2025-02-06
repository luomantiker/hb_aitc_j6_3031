/// \file
/// \ref Buffer Module
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

/// \defgroup Buffer Buffer
/// Interface to create and destroy actual memory buffer,
/// as well as buffer address and size
///
/// # List of Features
/// - Provide user the address and size of \ref hbrt4 allocated buffer
/// - Allow user to tell \ref hbrt4 the address and size of user allocated buffer
///
/// # Task Sequence
///
/// To provide BPU address of user allocated buffer to hbrt4,
/// in order to execute BPU model, user should:
/// - Query the memory requirement using the APIs listed in \ref Memspace module
/// - Allocate the memory buffer according to the requirement
/// - Record the memory address, and size if needed in \ref Hbrt4Buffer,
/// by \ref hbrt4BufferCreate or \ref hbrt4BufferCreateWithSize
/// Then bind the \ref Hbrt4Buffer to \ref Hbrt4Command in one of the following way:
/// 1. Register the buffer to \ref Hbrt4Command by \ref hbrt4CommandBuilderBindBuffer
/// 2. Register the buffer to \ref Hbrt4Value by \ref hbrt4ValueBuilderSetBuffer,
///   and register \ref Hbrt4Value to \ref Hbrt4Command by \ref hbrt4CommandBuilderBindValue
/// - When buffer is no longer used, it should be destructed by \ref hbrt4BufferDestroy
///
/// ---
///
/// For memory region allocated by hbrt4, which usually stores constant data.
/// User can get information of its \ref Hbrt4Buffer in the following way:
/// - Get \ref Hbrt4Buffer from \ref Hbrt4Memspace by \ref hbrt4MemspaceGetConstantBuffer
/// Its information get be retrievd by getter APIs such as:
/// 1. \ref hbrt4BufferGetAddress
/// 2. \ref hbrt4BufferGetSize
/// 3. \ref hbrt4BufferGetMemspace
/// - The lifetime of this \ref Hbrt4Buffer is managed by hbrt4,
/// user must *NOT* deallocated it for \ref hbrt4BufferDestroy
///
/// @{

/// Create a new buffer by a simple address
///
/// Cannot be used if memspace does not specify the size
///
/// \internal
/// HBRT internally cache buffer inside memspace, \n
/// so no heap allocation for `Hbrt4Buffer` is needed, \n
/// unless too many buffer are created with the same memspace
///
/// \param_in_obj{memspace}
/// \param[in] address Address used by buffer
/// \param[out] buffer Newly created buffer
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4BufferDestroy}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION `address` not satisfy alignment requirement of \ref
/// hbrt4MemspaceGetAlignment
/// - \ret_ok
Hbrt4Status hbrt4BufferCreate(Hbrt4Memspace memspace, void *address, Hbrt4Buffer *buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Create a buffer by address and size
///
/// Use if memspace does not specify the size, such as pyramid input and resizer input
///
/// \internal
/// HBRT internally cache buffer inside memspace, \n
/// so no heap allocation for `Hbrt4Buffer` is needed, \n
/// unless too many buffer are created with the same memspace
///
/// \param_in_obj{memspace}
/// \param[in] address Address used by buffer
/// \param[in] size Size used by buffer
/// \param[out] buffer Newly created buffer
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4BufferDestroy}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION `address` not satisfy alignment requirement of \ref
/// hbrt4MemspaceGetAlignment
/// - \ret_ok
Hbrt4Status hbrt4BufferCreateWithSize(Hbrt4Memspace memspace, void *address, size_t size,
                                      Hbrt4Buffer *buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Destroy the buffer
///
/// \param_dtor{buffer}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION `buffer` is still hold by one alive \ref Hbrt4Command
/// - \ret_ok
Hbrt4Status hbrt4BufferDestroy(Hbrt4Buffer *buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Get memspace of buffer, which describes the memory allocation requirement
///
/// \param_in_obj{buffer}
/// \param[out] memspace Memspace of buffer
///
/// \on_err_out_null_obj
/// \lifetime_getter_other{Hbrt4Hbm}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4BufferGetMemspace(Hbrt4Buffer buffer, Hbrt4Memspace *memspace) HBRT4_PRIV_CAPI_EXPORTED;

/// Get address of buffer
///
/// \param_in_obj{buffer}
/// \param[out] address Address of buffer
///
/// \on_err_out_zero
/// \lifetime_getter_other{Hbrt4Hbm}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4BufferGetAddress(Hbrt4Buffer buffer, void **address) HBRT4_PRIV_CAPI_EXPORTED;

/// Get size of buffer
///
/// \param_in_obj{buffer}
/// \param[out] size Size of buffer
///
/// \on_err_out_zero
/// \lifetime_getter_other{Hbrt4Hbm}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4BufferGetSize(Hbrt4Buffer buffer, size_t *size) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
