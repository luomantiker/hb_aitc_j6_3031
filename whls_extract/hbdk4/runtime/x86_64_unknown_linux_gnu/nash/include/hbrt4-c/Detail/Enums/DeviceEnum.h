/// \file
/// \ref Device Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Device Device
/// Device related enums
///
/// # Task Sequence
/// - \ref Hbrt4DeviceCategory of node can be retrieved by \ref hbrt4NodeGetDeviceCategory
///
/// ---
///
/// - \ref Hbrt4BpuMarch of hbm can be retrieved by
///   \ref hbrt4HbmGetBpuMarch or \ref hbrt4HbmHeaderGetBpuMarch
/// - \ref Hbrt4BpuMarch should be used by \ref hbrt4InstanceBuilderCreate,
/// to initialize the entire hbrt4 by creating \ref Hbrt4Instance.
/// See \ref Instance module for more detail.
///
/// @{

/// Device Category
enum Hbrt4DeviceCategory {
  /// Unknown device. This value is outputted only when API fails
  HBRT4_DEVICE_CATEGORY_UNKNOWN = 0,

  /// Node is run on BPU
  HBRT4_DEVICE_CATEGORY_BPU = 1,

  /// Node is run on CPU
  HBRT4_DEVICE_CATEGORY_CPU = 2,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4DeviceCategory);

/// BPU March, to differentiate BPU models
/// \since v4.0.1
enum Hbrt4BpuMarch {

  /// Unknown BPU march.
  /// \since v4.0.1
  ///
  /// \note
  /// This value is outputted only when API fails
  /// \remark
  /// This can also used in \ref hbrt4InstanceBuilderCreate
  HBRT4_BPU_MARCH_UNKNOWN = 0,

  /// For J5, same enum value as \ref hbrt3
  /// \since v4.0.1
  HBRT4_BPU_MARCH_BAYES2 = 3486274,
  /// For J6B, not supported by \ref hbrt3
  /// \since \CURRENT_HBRT4_VERSION
  HBRT4_BPU_MARCH_BPU31 = 4338498,

  /// For J5E, same enum value as \ref hbrt3
  /// \since v4.0.1
  HBRT4_BPU_MARCH_BPU25E = 4534850,

  /// For J6E, partially supported by \ref hbrt3
  ///
  /// \internal Called B30g internally in compiler team
  /// \since \CURRENT_HBRT4_VERSION
  HBRT4_BPU_MARCH_BPU30G = 4535106,

  /// For J6M, partially supported by \ref hbrt3
  ///
  /// \internal Called B30g2 internally in compiler team
  /// \since \CURRENT_HBRT4_VERSION
  HBRT4_BPU_MARCH_BPU30G2 = 5059394,

  /// For J6P, not supported by \ref hbrt3
  /// \since \CURRENT_HBRT4_VERSION
  HBRT4_BPU_MARCH_BPU30P = 5256002,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4BpuMarch);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
