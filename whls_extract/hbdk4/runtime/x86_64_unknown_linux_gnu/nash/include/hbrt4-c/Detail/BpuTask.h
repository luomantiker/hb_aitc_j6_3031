/// \file
/// \ref BpuTask Module
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

/// \ingroup Pipeline
/// \defgroup BpuTask BpuTask
///
/// Wrapper for new bpu driver API based on task
///
/// \see <https://horizonrobotics.feishu.cn/wiki/wikcnWbBmUrdb23OxJX22csXiXb> for J6 BSP API usage
///
/// @{

/// Get the handle directly used by driver API, such as `hb_bpu_core_process`
///
/// \deprecated
/// This function is deprecated because the optimization of frequent bpu task allocation/free,
/// as UCP team want to manually make bpu task allocation rather than hbrt4.
/// Please use \ref hbrt4BpuTaskConfig for speed optimization.
///
/// \param_in_obj{bpuTask}
/// \param[out] driverHandle The handle which can be directly used by
/// driver API such as `hb_bpu_core_process`.
/// This param can be safely type casted from `hb_bpu_task_t**`
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \warning
/// This function is mutually exclusive with \ref hbrt4BpuTaskConfig
/// User should never call the driver API `hb_bpu_task_free`,
/// This will be done internally by hbrt4, during \ref hbrt4PipelineDestroy
///
/// # Example code
///
/// ```C
/// // Get driver handle from `Hbrt4BpuTask`
/// // Error handling is omitted here
/// void submit_bpu_task_to_bpu(Hbrt4BpuTask bpuTask, hb_bpu_core_t core) {
///   hb_bpu_task_t* driverHandle;
///   hbrt4BpuTaskGetDriverHandle(bpuTask, (hbrt4__hb_bpu_task_t**)(&driverHandle));
///   hb_bpu_core_process(core, *driverHandle);
/// }
/// ```
///
/// \on_err_out_nullptr
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4BpuTaskGetDriverHandle(Hbrt4BpuTask bpuTask,
                                        hbrt4__hb_bpu_task_t **driverHandle) HBRT4_PRIV_CAPI_EXPORTED;

/// Config bpu task created by user, used to optimize Hbrt4BpuTask alloc/free before using BPU driver API
/// `hb_bpu_core_process`
///
/// \param[in] bpuTask HBRT4 object which has none the driver API handle `hb_bpu_task_t`
/// \param[in] hbTask The object which can be directly used by
/// driver API such as `hb_bpu_core_process`.
/// This param can be safely type casted from `hb_bpu_task_t`
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \warning
/// This function is mutually exclusive with \ref hbrt4BpuTaskGetDriverHandle
/// User should call the driver API `hb_bpu_task_alloc`,
/// This should be done externally by UCP, during `hb_bpu_task_free`
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4BpuTaskConfig(Hbrt4BpuTask bpuTask, hbrt4__hb_bpu_task_t hbTask) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
