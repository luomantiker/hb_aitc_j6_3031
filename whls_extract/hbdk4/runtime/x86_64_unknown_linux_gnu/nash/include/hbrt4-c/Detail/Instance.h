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

/// \defgroup Instance Instance
/// Top level instance to store global configuration information.
/// It should be created first in \ref hbrt4
///
/// # List of Features
/// - Initialize all internal states of \ref hbrt4,
///   including all states that cannot specific to Hbm model
/// - Provide hbrt4 the BPU march information
/// - Register the logger used by hbrt4
/// - Provide BPU node to do BPU context switch
///
///
/// \internal
/// There are no relationship between two different \ref Hbrt4Instance
/// This has root ownership of all HBRT objects \n
/// \ref hbrt4 design does not use true global variables \n
/// Any state must be directly or directly hold by one \ref Hbrt4Instance
///
/// # Task Sequence
///
/// - Create an instance builder with unknown bpu march
///   (with \ref HBRT4_BPU_MARCH_UNKNOWN),
///   by \ref hbrt4InstanceBuilderCreate,
/// - Then using \ref hbrt4InstanceBuilderInto create the instance wit
///   unknown bpu march
/// - Get the bpu march of hbm by \ref hbrt4HbmHeaderCreateByFilename
///   and \ref hbrt4HbmHeaderGetBpuMarch , as listed in \ref Hbm module
/// - Create an instance builder with known bpu march
///   by \ref hbrt4InstanceBuilderCreate
/// - Register the logger using \ref hbrt4InstanceBuilderRegisterLogger
/// - Register HBTL kernel using \ref hbrt4InstanceBuilderRegisterJitKernel
/// - Create an instance with known march using \ref hbrt4InstanceBuilderCreate
/// - Create hbm using \ref hbrt4HbmCreateByFilename
/// - Continue processing using APIs listed in \ref Hbm module
/// - When \ref Hbrt4Instance is no longer used,
///   usually at the time when the entire processing is being exited,
///   it should be destroyed by \ref hbrt4InstanceDestroy
///
///
/// # Purpose
///
/// Why need to have this \ref Instance. Why not just create hbm without instance?
///
/// Answer: \n
/// We want the global configuration, such as setting loggers,
/// to be done, only before the hbm creation. This is more secure.
///
/// The \ref Hbrt4InstanceBuilder -> \ref Hbrt4Instance -> \ref Hbrt4Hbm creation sequence, \n
/// ensure that it is impossible for user to modify the global configuration,
/// once that is set, \n
/// and it is impossible to set it after hbm creation
///
/// Previously in \ref hbrt3, `hbrtSetGlobalConfig` is a freestanding function. \n
/// It is not enforcing user to call it before hbm loading, which is problematic,
/// so many locking is done under the hood to prevent race condition
///
/// @{

/// Initialize the \ref Hbrt4InstanceBuilder
///
/// \param[in] march The bpu march \n
/// Note that this can be \ref HBRT4_BPU_MARCH_UNKNOWN, \n
/// in this case, instance can only be used on APIs in \ref HbmHeader module,
/// but not \ref Hbm module
/// \param[out] builder The builder
///
/// \lifetime_ctor{hbrt4InstanceBuilderInto}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4Instance or \ref Hbrt4InstanceBuilder to be alive in the
/// entire process
///
/// \note
/// Use bpu driver API or \ref hbrt4HbmHeaderGetBpuMarch in \ref HbmHeader module to get the bpu march
///
/// \returns
/// - \ret_null_out
/// - \ref HBRT4_STATUS_INVALID_ARGUMENT `march` is invalid or not supported
/// - \ref HBRT4_STATUS_NOT_SUPPORTED An \ref Hbrt4InstanceBuilder or \ref Hbrt4InstanceBuilder already exists before
/// this API completes
/// - \ret_ok
Hbrt4Status hbrt4InstanceBuilderCreate(Hbrt4BpuMarch march, Hbrt4InstanceBuilder *builder) HBRT4_PRIV_CAPI_EXPORTED;
/// Initialize the \ref Hbrt4InstanceBuilder
///
/// \param[in] bpuMarchName The bpu march \n
/// in this case, instance can only be used on APIs in \ref HbmHeader module,
/// but not \ref Hbm module
/// \param [in] preInit The instance without marchName, used to set the known marchName
/// instance's logger
/// \param[out] builder The builder
///
/// \lifetime_ctor{hbrt4InstanceBuilderInto}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4Instance or \ref Hbrt4InstanceBuilder to be alive in the
/// entire process
///
/// \note
/// Use bpu driver API or \ref hbrt4HbmHeaderGetBpuMarch in \ref HbmHeader module to get the bpu march
///
/// \returns
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NULL_OBJECT `preInit` or `preInit's logger` is null
/// - \ref HBRT4_STATUS_INVALID_ARGUMENT `march` is invalid or not supported
/// - \ref HBRT4_STATUS_NOT_SUPPORTED An \ref Hbrt4InstanceBuilder or \ref Hbrt4InstanceBuilder already exists before
/// this API completes
/// - \ret_ok
Hbrt4Status hbrt4InstanceBuilderCreate2(const char *bpuMarchName, Hbrt4PreInit preInit,
                                        Hbrt4InstanceBuilder *builder) HBRT4_PRIV_CAPI_EXPORTED;
/// Register the logger to the instance builder
///
/// \todo (hehaoqian) Describe the default logger, if this API not called
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
Hbrt4Status hbrt4InstanceBuilderRegisterLogger(Hbrt4InstanceBuilder builder,
                                               Hbrt4Logger logger) HBRT4_PRIV_CAPI_EXPORTED;

/// Handler to the register kernel
///
/// \warning
/// This handler may be called in multiple threads
/// It is the user's responsibility to ensure this handler is thread-safe
///
/// \param[in] kernel hbtl kernel pointer
/// \return
/// - null
/// \note
/// HBRT **NEVER** abort or panic regardless this handler returns success or not
typedef void (*registerFuncHandler)(void *kernel);

/// Jit Kernel register to the instance builder
///
///
/// \param_in_obj{builder}
/// \param[in] registerFunc used to register jit kernel
///
/// \lifetime_no_transfer
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4InstanceBuilderRegisterJitKernel(Hbrt4InstanceBuilder builder,
                                                  registerFuncHandler registerFunc) HBRT4_PRIV_CAPI_EXPORTED;

/// Create the instance and destroy the instance builder
///
/// \param_in_obj{builder}
/// \param_builder_into_out{instance}
///
/// \on_err_out_null_obj
/// \lifetime_builder{hbrt4InstanceDestroy}
///
/// \mt_unsafe_unique
///
/// \remark
/// Currently implementation only allow **ONE** \ref Hbrt4Instance or \ref Hbrt4InstanceBuilder to be alive in the
/// entire process
///
/// \note
/// Set `instance` to nullptr, if you want to destroy the builder without creating `instance`
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_NOT_SUPPORTED An \ref Hbrt4InstanceBuilder or \ref Hbrt4InstanceBuilder already exists before
/// this API completes
/// - \ret_ok
Hbrt4Status hbrt4InstanceBuilderInto(Hbrt4InstanceBuilder *builder, Hbrt4Instance *instance) HBRT4_PRIV_CAPI_EXPORTED;

/// Destroy instance and free all resources associate with it
///
/// \param[in,out] instance The instance to be destroyed
///
/// \lifetime_dtor
///
/// \test
/// Ensure this API fails if any related \ref Hbrt4Hbm object is still alive
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Any related \ref Hbrt4Hbm object still alive
/// - \ret_ok
Hbrt4Status hbrt4InstanceDestroy(Hbrt4Instance *instance) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
