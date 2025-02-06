/// \file
/// \ref Memspace Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DeviceEnum.h"
#include "hbrt4-c/Detail/Enums/MemspaceEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Memspace Memspace
/// Get memory allocation requirement, used to create \ref Buffer to bind \ref Command
///
/// # List of Features
/// - Name of memspace
/// - Size and alignment requirement of memspace
/// - Usage of memspace
///
/// In \ref hbrt4, the memory allocation requirement to run node,
/// is no longer directly one-to-one associated with tensors as \ref hbrt3
///
/// Instead, the memory requirement is designed to be a independent \ref Memspace module.
///
/// This allows the following:
/// - Two tensors can share the same chunk of memory chunk arbitrarily,
/// even with overlap
/// - The memory size or alignment can be adjusted without considering the tensor memory layout,
/// for example, add red-zone guard page during runtime, for memory out of bound check
/// - There are less ambiguity regarding how much memory needs to be allocated,
/// especially important when \ref hbrt4 supports dynamic shaped tensor
///
/// \invariant
/// - Each \ref Hbrt4Variable is always associated with one or zero \ref Hbrt4Memspace
/// - All data of the \ref Hbrt4Variable is inside the associate \ref Hbrt4Memspace
///
/// # Task Sequence
///
/// - Get the memspaces to describe the memory allocation,
///   by \ref hbrt4VariableGetMemspace or \ref hbrt4NodeGetMemspace or \ref hbrt4GraphGetMemspace \n
///   The most important one is \ref hbrt4NodeGetMemspace , this is the memory allocation requirement
///   in order to run the node
///
/// If the memspace is not constant, its buffer should be allocated by user:
/// - Get size and alignment requirement by \ref hbrt4MemspaceGetAlignment and \ref hbrt4MemspaceGetSize
/// - Allocate the memory using BSP driver APIs
/// - Use the address and size of allocated buffer to create \ref Hbrt4Buffer. See \ref Buffer module for detail
/// - Fill the input data into buffer according to tensor APIs listed in \ref Variable module
///
/// If the memspace is constants, its buffer is managed by hbrt:
/// - Get buffer by \ref hbrt4MemspaceGetConstantBuffer
///

/// \attention
/// \ref Hbrt4MemspaceUsage is very important. Read the document of that
///
/// @{

/// Get the usage of memspace.
///
/// \attention
/// This concept is very important to avoid undefined behavior.
/// See \ref Hbrt4MemspaceUsage for detail
///
/// \param_in_obj{memspace}
/// \param[out] usage Memspace usage
///
/// \on_err_out_set_to{HBRT4_MEMSPACE_USAGE_UNKNOWN}
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4MemspaceGetUsage(Hbrt4Memspace memspace, Hbrt4MemspaceUsage *usage) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the alignment of memspace.
/// The buffer allocated according to this memspace
/// must have its address divisible by `alignment`
///
/// \param_in_obj{memspace}
/// \param[out] alignment Alignment requirement
///
/// \on_err_out_zero
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4MemspaceGetAlignment(Hbrt4Memspace memspace, size_t *alignment) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the size of memspace.
/// The buffer allocated according to this memspace must uses its size
///
/// \param_in_obj{memspace}
/// \param[out] size Size requirement
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_DYNAMIC_VALUE Size is not statically known
/// - \ret_ok
Hbrt4Status hbrt4MemspaceGetSize(Hbrt4Memspace memspace, size_t *size) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the buffer associated with a constant memspace
/// (With usage of \ref HBRT4_MEMSPACE_USAGE_CONSTANT)
///
/// No need to destroy the output `buffer`, which is managed by \ref hbrt4
///
/// \param_in_obj{memspace}
/// \param[out] buffer Constant buffer
///
/// \on_err_out_zero
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4MemspaceGetConstantBuffer(Hbrt4Memspace memspace, Hbrt4Buffer *buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the unique name of memspace
///
/// \param_in_obj{memspace}
/// \param[out] name Null terminated memspace name
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
Hbrt4Status hbrt4MemspaceGetName(Hbrt4Memspace memspace, const char **name) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
