/// \file
/// \ref Command Module
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

/// \defgroup Command Command
/// Bind the actual buffer address to a specified node for node running
///
/// # List of Features
/// - Bind actual address to be used by BPU
/// - Provide actual tensor dim and stride, for dynamic shaped tensor
/// - Generate \ref Hbrt4Command which has all information needed to run instructions on BPU
/// - Provide memory buffers used by \ref Hbrt4Command
///
/// # Task Sequence
///
/// - Create \ref Hbrt4CommandBuilder from \ref Hbrt4Node by \ref hbrt4CommandBuilderCreate
/// - If the shape and stride of all paramater variables are statically known,
///   bind all \ref Hbrt4Buffer needed to run the \ref Hbrt4Node by \ref hbrt4CommandBuilderBindBuffer,
///   otherwise bind all \ref Hbrt4Value needed to run the \ref Hbrt4Node by \ref hbrt4CommandBuilderBindValue,
/// - Create \ref Hbrt4Command and destroy \ref Hbrt4CommandBuilder by \ref hbrt4CommandBuilderInto
/// - Bind the command into \ref Hbrt4Pipeline, and use APIs in \ref Pipeline module to execute BPU
/// - When \ref Hbrt4Command is no longer used, it should be destroyed by \ref hbrt4CommandDestroy
///
/// @{

/// Create a new command builder object from node
///
/// \note
/// \ref Hbrt4CommandBuilder is used to create \ref Hbrt4Command
///
/// \param_in_obj{node}
/// \param[out] builder Builder to create command
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4CommandBuilderInto}
///
/// \mt_safe
///
/// \remark
/// User can create arbitrary numbers of \ref Hbrt4Command from \ref Hbrt4Node \n
/// Each command represents a different "frame"
///
/// \test
/// Ensure this API is multi thread safe, by running this API in multiple thread on the same node
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4CommandBuilderCreate(Hbrt4Node node, Hbrt4CommandBuilder *builder) HBRT4_PRIV_CAPI_EXPORTED;

/// Bind memspace buffers for command execution
///
/// \param_in_obj{builder}
/// \param[in] buffer Buffer to bind, corresponds to \ref Hbrt4Memspace provided by \ref hbrt4NodeGetMemspace
///
/// \remark This does not include constant buffer
///
/// \mt_unsafe_mut \n
/// \n
/// Calling this API multi-thread on the same `buffer` object is **ALLOWED**, \n
/// but on the same `builder` object is **NOT** allowed
///
/// \note
/// `buffer` is still alive after this API returns \n
/// This API does not take the ownership of `buffer`
/// Note that destroy \ref Hbrt4Buffer when it is hold by \ref Hbrt4Command is not allowed
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_oob
/// - \ref HBRT4_STATUS_BAD_DATA The memspace of `buffer` is not as expected
/// - \ret_ok
Hbrt4Status hbrt4CommandBuilderBindBuffer(Hbrt4CommandBuilder builder, Hbrt4Buffer buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Bind arguments by \ref Hbrt4Value for input variable(s) and create command
///
/// The \ref Hbrt4Value corresponds the parameter variables provided by \ref hbrt4NodeGetParameterVariable
Hbrt4Status hbrt4CommandBuilderBindValue(Hbrt4CommandBuilder builder, Hbrt4Value value) HBRT4_PRIV_CAPI_EXPORTED;

/// Create a new command and destroy the command builder
///
/// \param_in_obj{builder}
/// \param_builder_into_out{command}
///
/// \lifetime_builder{hbrt4CommandDestroy}
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Not all buffers required have been bind
/// - \ref HBRT4_STATUS_BAD_DATA Bad data in buffer with usage of \ref HBRT4_MEMSPACE_USAGE_NODE_CACHE
/// - \ret_ok
Hbrt4Status hbrt4CommandBuilderInto(Hbrt4CommandBuilder *builder,
                                    Hbrt4Command *command) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Destroy the command object
///
/// \param_dtor{command}
///
/// \lifetime_dtor
///
/// \note
/// If this \ref Hbrt4Command has been bind to a \ref Hbrt4Pipeline,
/// by \ref hbrt4PipelineBuilderPushCommand,
/// this \ref Hbrt4Command should be destroyed after
/// the destruction of \ref Hbrt4Pipeline
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION `command` is still hold by one \ref Hbrt4Pipeline
/// - \ret_ok
Hbrt4Status hbrt4CommandDestroy(Hbrt4Command *command) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
