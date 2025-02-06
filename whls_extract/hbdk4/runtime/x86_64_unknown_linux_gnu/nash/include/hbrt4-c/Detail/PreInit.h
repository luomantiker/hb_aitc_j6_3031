/// \file
/// \ref Instance Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DeviceEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN
/// @{
/// Initialize the \ref Hbrt4InstanceBuilder
///
/// \param[in,out] builder The point of builder
///
/// \lifetime_ctor{hbrt4PreInitBuilderInto}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4PreInit or \ref Hbrt4PreInitBuilder to be alive in the
/// entire process
///
/// \returns
/// - \ret_null_out
/// - \ref HBRT4_STATUS_INVALID_ARGUMENT `builder` is invalid
/// - \ret_ok
Hbrt4Status hbrt4PreInitBuilderCreate(Hbrt4PreInitBuilder *builder) HBRT4_PRIV_CAPI_EXPORTED;

/// Register the logger to the preInit builder
///
///
/// \param_in_obj{builder}
/// \param[in] logger The logger
///
/// \lifetime_no_transfer
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_ALREADY_EXISTS This `builder` has already registered a `logger`
/// - \ret_ok
Hbrt4Status hbrt4PreInitBuilderRegisterLogger(Hbrt4PreInitBuilder builder, Hbrt4Logger logger) HBRT4_PRIV_CAPI_EXPORTED;

/// Create the preinit and destroy the preinit builder
///
/// \param_in_obj{builder}
/// \param_builder_into_out{preInit}
///
/// \on_err_out_null_obj
/// \lifetime_builder{hbrt4PreInitDestroy}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4PreInit or \ref Hbrt4PreInitBuilder to be alive in the
/// entire process
///
/// \note
/// Set `preInit` to nullptr, if you want to destroy the builder without creating `preInit`
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// \ref Hbrt4PreInitBuilder or \ref Hbrt4PreInit already exists before
/// this API completes
/// - \ret_ok
Hbrt4Status hbrt4PreInitBuilderInto(Hbrt4PreInitBuilder *builder, Hbrt4PreInit *preInit) HBRT4_PRIV_CAPI_EXPORTED;

/// Destroy preInit and reclaim its space
///
/// \param[in,out] preInit The preInit to be destroyed
///
/// \lifetime_dtor
///
/// \test
/// Ensure this API fails if any related \ref Hbrt4PreInit object is still alive
///
/// \returns\
/// - \ret_ok
Hbrt4Status hbrt4PreInitDestroy(Hbrt4PreInit *preInit) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
