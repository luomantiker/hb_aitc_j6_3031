/// \file
/// \ref Pipeline Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/External/NewDriverApi.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Pipeline Pipeline
/// Fuse multiple \ref Command together and generate data for BPU driver.
/// This operating mechanism can effectively utilize BPU hardware resource
///
/// # List of Features
/// - Fuse multiple commands together
/// - Generate BPU tasks from pipeline, which is to be used for BPU driver
/// - Set interrupt number used by BPU driver (J5 only)
///
/// \see For funccall preemption functionality, read doc of \ref ContextSwitch module
///
/// # Task Sequence
///
/// - Create \ref Hbrt4PipelineBuilder by \ref hbrt4PipelineBuilderCreate
/// - Add command by \ref hbrt4PipelineBuilderPushCommand
///
/// ---
///
/// For J5:
/// - Set interrupt number by \ref hbrt4PipelineBuilderSetInterruptNumber
/// - Create \ref Hbrt4Pipeline and destroy \ref Hbrt4PipelineBuilder by hbrt4PipelineBuilderInto
/// - Get \ref Hbrt4Funccall from \ref Hbrt4Pipeline by \ref hbrt4PipelineGetBpuFunccall
/// - Submit funccall by bpu driver API such as `cnn_core_set_fc`.
///   See description of \ref Hbrt4Funccall for detail
/// - Wait for the completion for funccall by BSP driver API
///   `cnn_core_wait_fc_done` or `cnn_core_check_fc_done`
/// - Destroy \ref Hbrt4Pipeline by \ref hbrt4PipelineDestroy if \ref Hbrt4Pipeline is no longer use. \n
///   Note that repeated calling `cnn_core_set_fc` and wait for funccall completion without
///   destroying \ref Hbrt4Funccall is allowed
/// \see <https://horizonrobotics.feishu.cn/docs/doccnv0VzNVLFrHor5HWcyu0H8b> for J5 BSP API usage
///
/// ---
///
/// For J6:
/// - Create \ref Hbrt4Pipeline and destroy \ref Hbrt4PipelineBuilder by hbrt4PipelineBuilderInto
/// - deprecated method: {
/// - Get \ref Hbrt4BpuTask from \ref Hbrt4Pipeline by \ref hbrt4PipelineGetBpuTask
/// - Get task handle directly used by driver API from \ref Hbrt4BpuTask by \ref hbrt4BpuTaskGetDriverHandle
/// - }
/// - new method for speed optimization: {
/// - Alloc `hb_bpu_task_t` from `hb_bpu_task_alloc` and free it from `hb_bpu_task_free`
/// - Get bpu task number from \ref hbrt4PipelineGetNumBpuTasks
/// - Get bpu task data with the iteration of \ref hbrt4PipelineGetBpuTask2
/// - Config bpu task from \ref hbrt4BpuTaskConfig
/// - }
/// - Submit task by bpu driver API such as `hb_bpu_core_process`.
/// - Wait for the completion for task by BSP driver API `hb_bpu_task_wait`
/// - Destroy \ref Hbrt4Pipeline by \ref hbrt4PipelineDestroy if \ref Hbrt4Pipeline is no longer use. \n
///   Note that repeated calling `hb_bpu_core_process` and wait for bpu task completion without
///   destroying \ref Hbrt4Pipeline is allowed
/// \see <https://horizonrobotics.feishu.cn/wiki/wikcnWbBmUrdb23OxJX22csXiXb> for J6 BSP API usage
///
/// @{

/// Create a new pipeline builder to store command
///
/// \param_in_obj{instance}
/// \param[out] builder Builder to create pipeline
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4PipelineBuilderInto}
///
/// \mt_safe
///
/// \test
/// Ensure this API is multi thread safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4PipelineBuilderCreate(Hbrt4Instance instance, Hbrt4PipelineBuilder *builder) HBRT4_PRIV_CAPI_EXPORTED;

/// Push a command to the end of queue in the pipeline builder
///
/// \param_in_obj{builder}
/// \param[in] command Builder to create pipeline
///
/// \mt_unsafe_mut \n
/// \n
/// Calling this API multi-thread on the same `command` object is **ALLOWED**, \n
/// but on the same `builder` object is **NOT** allowed
///
/// \note
/// It is **ALLOWED** to have a command in multiple pipeline \n
/// but running these pipelines at the same time in \ref undefined_behavior
///
/// \note
/// It is also **ALLOWED** to have a command to be pushed back
/// to the same pipeline multiple times
///
/// \test
/// Test this API is multi thread safe
/// \test
/// Test push the commands multiple times to the same builder is ok
/// \returns
/// - \ret_null_in_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Commands from multiple devices
/// categories have been pushed.
/// BPU and CPU commands cannot be in the same pipeline
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4PipelineBuilderPushCommand(Hbrt4PipelineBuilder builder,
                                            Hbrt4Command command) HBRT4_PRIV_CAPI_EXPORTED;

/// Set the interrupt number of BPU driver to meet the function `hb_bpu_core_check_fc_done`
///
/// Calling this API is optional
///
/// \deprecated
/// This function exists because requested by libdnn \n
/// The author of \ref hbrt4 does not believe interrupt number should be exposed to the application,
/// because it is not needed, and error prone \n
/// This API will be deprecated when OS team releases new driver API which does not use interrupt number
///
/// \warning
/// Use this function in one of the following ways:
/// - In the entire process, **NOT** calling this API. \ref hbrt4 will automatically assign interrupt number
/// - Call this APIs for every \ref Hbrt4PipelineBuilder \n
/// Caller must ensure all live \ref Hbrt4Pipeline or \ref Hbrt4PipelineBuilder have different interrupt number \n
/// \ref undefined_behavior otherwise
///
/// \param_in_obj{builder}
/// \param[in] interruptNumber Non-zero interrupt number
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_INVALID_ARGUMENT `interruptNumber` is zero
/// - \ret_ok
Hbrt4Status hbrt4PipelineBuilderSetInterruptNumber(Hbrt4PipelineBuilder builder,
                                                   uint32_t interruptNumber) HBRT4_PRIV_CAPI_EXPORTED;

/// Enable profiling of pipeline.
/// Call \ref hbrt4PipelineGetProfilingData to get the profiling data
///
/// \param_in_obj{builder}
///
/// \note
/// This is for the profiling purpose, and not recommended to be used in production.
/// This API will cost a bit of performance
///
/// \mt_unsafe_mut
///
/// \note
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4PipelineBuilderEnablingProfiling(Hbrt4PipelineBuilder builder) HBRT4_PRIV_CAPI_EXPORTED;

/// Create a new pipeline and destroy the pipeline builder
///
/// \param_in_obj{builder}
/// \param_builder_into_out{pipeline}
///
/// \lifetime_builder{hbrt4PipelineDestroy}
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Not all buffers required have been bind
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION \ref hbrt4PipelineBuilderSetInterruptNumber has been called at least once,
/// but not called for the current \ref Hbrt4PipelineBuilder
/// - \ret_ok
Hbrt4Status hbrt4PipelineBuilderInto(Hbrt4PipelineBuilder *builder,
                                     Hbrt4Pipeline *pipeline) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Destroy the pipeline object
///
/// \param_dtor{pipeline}
///
/// \lifetime_dtor
///
/// \warning
/// The lifetime of \ref Hbrt4Funccall ends after the destruction of \ref Hbrt4Pipeline
///
/// \returns
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4PipelineDestroy(Hbrt4Pipeline *pipeline) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Doc TODO
///
/// \j5_only
typedef void (*Hbrt4BpuFunccallCallback)(uint32_t id, int32_t err);

/// structure expression of BPU funccall for `cnn_core_set_fc` \n
/// Modification of funccalls by any means is not allowed
///
/// \j5_only
///
/// # Default Value {#Hbrt4Funccall_DefaultValue}
/// - All valid funccall will be returned that can be safely executed on bpu
/// - The returned funccall has no side effect, which does not write any data
struct Hbrt4Funccall {
  /// Callback function used in `cnn_core_set_fc`
  Hbrt4BpuFunccallCallback fcDoneCallback;

  /// funccalls used in `cnn_core_set_fc`
  /// \warning
  /// It is not allowed to modify any data inside this,
  /// \ref undefined_behavior otherwise

  void *funccalls;

  /// Number of funccalls used in `cnn_core_set_fc`
  uint32_t numFunccalls;

  /// DOC TODO
  uint32_t numCores;

  /// Interrupt number used to wait for the completion of funccalls
  uint32_t interruptNumber;

  /// Internal data used by \ref hbrt4
  void *opaque;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4Funccall);
/// \endcond

/// Get the bpu funccalls, to start actual bpu execution using BPU driver API `cnn_core_set_fc`
///
/// \j5_only
///
/// \warning
/// Modifying the data in returned \ref Hbrt4Funccall by any means is \ref undefined_behavior
///
/// \param_in_obj{pipeline}
/// \param[out] funccall Funccalls to be submitted by `cnn_core_set_fc`
///
/// \on_err_out_set_to{Hbrt4Funccall_DefaultValue}
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4PipelineGetBpuFunccall(Hbrt4Pipeline pipeline, Hbrt4Funccall *funccall) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu task, used to start actual bpu execution using BPU driver API `hb_bpu_core_process`
///
/// \deprecated
/// This function is deprecated because the optimization of frequent bpu task allocation/free,
/// as UCP team want to manually make bpu task allocation rather than hbrt4.
/// and API \ref hbrt4BpuTaskGetDriverHandle is also deprecated.
/// Please use \ref hbrt4PipelineGetBpuTask2 for speed optimization.
///
/// \param_in_obj{pipeline}
/// \param[in] taskType The task type as defined by driver API. See doc of driver API for detail
/// \param[out] bpuTask HBRT4 object which contains the driver API handle.
/// This returned `task` cannot be used directly by the driver API such as `hb_bpu_core_process`.
/// Acquire the real driver handle by \ref hbrt4BpuTaskGetDriverHandle
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \warning
/// User should never call the driver API `hb_bpu_task_free`,
/// This will be done internally by hbrt4, during \ref hbrt4PipelineDestroy
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4PipelineGetBpuTask(Hbrt4Pipeline pipeline, Hbrt4__hb_task_type_t taskType,
                                    Hbrt4BpuTask *bpuTask) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Get the number of bpu task in pipeline for multiple cores running
///
/// \param_in_obj{pipeline}
/// \param[out] num Pipeline bpu task data num
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4PipelineGetNumBpuTasks(Hbrt4Pipeline pipeline, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get bpu task serializing data with corresponding position in pipeline, serializing data is used to config bpu task
/// object `hb_bpu_task_t` alloced with `hb_bpu_task_alloc`.
/// Meanwhile, the output parameter of bpuTask no longer has the ownership of `hb_bpu_task_t`,
/// The difference of this function and \ref hbrt4PipelineGetBpuTask : \n
/// This function: only get bpu task data. User should manually alloc/free `hb_bpu_task_t` and config it through \ref
/// hbrt4BpuTaskConfig for speed optimization \n \ref hbrt4PipelineGetBpuTask : automatically alloc/free/config
/// `hb_bpu_task_t` inside it, get task handle from \ref hbrt4BpuTaskGetDriverHandle . But task allocation is time
/// consuming.
///
/// \param_in_obj{pipeline}
/// \param[in] pos Index in range of [0, \ref hbrt4PipelineGetNumBpuTasks )
/// \param[out] bpuTask HBRT4 object which does not contain the driver API handle.
/// This returned `task` cannot be used directly by the driver API such as `hb_bpu_core_process`.
/// User should config `bpuTask` bind with `hb_bpu_task_t` by \ref hbrt4BpuTaskConfig
///
/// \exp_api
/// \do_not_test_in_j5_toolchain

/// \warning
/// User should never call the HBRT4 API \ref hbrt4BpuTaskGetDriverHandle,
/// because the return value `bpuTask` no longer has the ownership of `hb_bpu_task_t` driver handle,
/// and it is only used to config or bind to `hb_bpu_task_t` with \ref hbrt4BpuTaskConfig
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4PipelineGetBpuTask2(Hbrt4Pipeline pipeline, size_t pos,
                                     Hbrt4BpuTask *bpuTask) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the profiling data from the Pipeline
///
/// \param_in_obj{pipeline}
/// \param[out] address The address of the profiling data.
/// \param[out] size The size of the profiling data. This only contains valid data of profiling data
///
/// \warning
/// The returned address is bpu memory,
/// and it is currently user's responsibility to invalidate cpu cache before read it,
/// otherwise wrong data may be read.
///
/// \note
/// It is allowed to call this function while the pipeline is running.
///
/// The output `size` of two \ref hbrt4PipelineGetProfilingData
/// to the same pipeline, may be different depending on
/// whether the pipeline is running or finished
///
/// \warning
/// User should not modify the profiling the data in any way, which is undefined behavior
///
/// \note
/// For the format of profiling data, \see the documentation of \ref Profiling module
///
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Profiling is not enabled by the pipeline
Hbrt4Status hbrt4PipelineGetProfilingData(Hbrt4Pipeline pipeline, const void **address,
                                          size_t *size) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
