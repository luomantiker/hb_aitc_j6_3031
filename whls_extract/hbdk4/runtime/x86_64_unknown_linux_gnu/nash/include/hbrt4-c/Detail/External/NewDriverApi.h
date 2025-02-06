/// \file
/// \ref NewDriverApi Module
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

/// \ingroup BpuTask
/// \defgroup NewDriverApi NewDriverApi
///
/// Defined wrapper types for new BPU driver API based on task
/// This module exists to avoid inclusion of header file of bpu driver API
///
/// \see <https://horizonrobotics.feishu.cn/wiki/wikcnWbBmUrdb23OxJX22csXiXb> for J6 BSP API usage
///
/// @{

/// Equivalent to enum `hb_task_type_t` used in new bpu driver API
/// Control the type of `hb_bpu_task_t` used in new bpu driver API.
///
/// \internal This type exists to avoid inclusion of header file of bpu driver API
///
/// \todo static assert the value of each variant is equivalent to that of `hb_task_type_t`
///
/// \internal This enum is not defined in `hbrt4-c/Detail/Enums`
/// because we do not want to generate Rust style enum for this enum
enum Hbrt4__hb_task_type_t {

  /// Equivalent to `TASK_TYPE_SYNC` in driver API.
  /// See doc of driver API for detail
  HBRT4__TASK_TYPE_SYNC = 0x0000,

  /// Equivalent to `TYPE_TRIG_TASK` in driver API.
  /// See doc of driver API for detail
  HBRT4__TYPE_TRIG_TASK = 0x0001,

  /// Equivalent to `TYPE_TRIG_CORE` in driver API.
  /// See doc of driver API for detail
  HBRT4__TASK_TYPE_CORE = 0x0002,

  /// Equivalent to `TYPE_TRIG_GRAPH` in driver API.
  /// See doc of driver API for detail
  HBRT4__TASK_TRIGGER_GRAPH = 0x0004,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4__hb_task_type_t);

/// Equivalent to `hb_bpu_task_t` used in new bpu driver API
///
/// \internal This type exists to avoid inclusion of header file of bpu driver API
///
/// \todo static assert `sizeof(hb_bpu_task_t) == sizeof(hbrt4__hb_bpu_task_t)`
struct hbrt4__hb_bpu_task_t {
  /** \private_api */
  uint64_t opaque;
};
typedef struct hbrt4__hb_bpu_task_t hbrt4__hb_bpu_task_t;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
