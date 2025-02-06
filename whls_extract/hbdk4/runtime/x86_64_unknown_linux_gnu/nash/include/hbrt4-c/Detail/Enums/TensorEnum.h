/// \file
/// \ref TensorType Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup TensorType
/// @{

/// Tensor encoding to describe whether tensor uses some special layout
///
/// such as RLE which compresses the data
enum Hbrt4TensorEncoding {

  /// Unknown encoding. This value is outputted only when API fails
  HBRT4_TENSOR_ENCODING_UNKNOWN = 0,

  /// Standard tensor encoding that memory layout is determined by \ref hbrt4TypeGetTensorStrides
  HBRT4_TENSOR_ENCODING_DEFAULT = 1,

  /// \todo (hehaoqian): Add doc when this is implemented
  HBRT4_TENSOR_ENCODING_RLE = 2,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4TensorEncoding);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
