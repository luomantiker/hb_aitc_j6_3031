/// \file
/// \ref Type Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \addtogroup Type
/// @{

/// Tag to represent the basic type information
///
/// \internal
/// The enum values should match that of HBTL, if exists
/// There is static check in HBTL source code
///
/// \internal
/// The enum value was previously encoded with macro,
/// but that is hard to read by tester.
/// Now changed to plain number.
/// See StaticCheck/TypeEnum.cpp
enum Hbrt4TypeTag {

  /// Unknown type
  /// This value outputted only when \ref hbrt4TypeGetTag errors
  HBRT4_TYPE_TAG_UNKNOWN = -1,

  /// Two int4 values, which shares a byte storage
  ///
  /// Used by J5 BPU
  HBRT4_TYPE_TAG_SI4x2 = 65793,

  /// Signed int8 value with 1 byte storage
  HBRT4_TYPE_TAG_SI8 = 65794,

  /// Little endian, signed int16 value with 2 bytes storage
  HBRT4_TYPE_TAG_SI16 = 131331,

  /// Little endian, signed int32 value with 4 bytes storage
  HBRT4_TYPE_TAG_SI32 = 262404,

  /// Big endian, signed int32 value with 4 bytes storage
  ///
  /// This type is only used by XJ2 and XJ3 bpu,
  /// not used by later BPU such as J5
  HBRT4_TYPE_TAG_SI32_BIG_ENDIAN = 262405,

  /// Little endian, signed int64 value with 8 bytes storage
  ///
  /// This type is not used by BPU
  HBRT4_TYPE_TAG_SI64 = 524549,

  /// Unsigned int8 value with 1 byte storage
  HBRT4_TYPE_TAG_UI8 = 66049,

  /// Bool value with 1 byte storage
  /// Can only have value of 0 or 1
  ///
  /// This type is for ONNX
  HBRT4_TYPE_TAG_BOOL = 66053,

  /// Unsigned int16 value with 2 bytes storage
  HBRT4_TYPE_TAG_UI16 = 131586,

  /// Unsigned int32 value with 4 bytes storage
  HBRT4_TYPE_TAG_UI32 = 262659,

  /// Unsigned int64 value with 8 bytes storage
  HBRT4_TYPE_TAG_UI64 = 524804,

  /// Standard IEEE 754 float16, with 2 bytes storage
  ///
  /// 11 bits for significand and 5 bits for exponent
  HBRT4_TYPE_TAG_F16 = 131074,

  /// Standard IEEE 754 float32, with 4 bytes storage
  ///
  /// 24 bits for significand and 8 bits for exponent
  HBRT4_TYPE_TAG_F32 = 262147,

  /// Standard IEEE 754 float64, with 8 bytes storage
  ///
  /// 53 bits for significand and 11 bits for exponent
  HBRT4_TYPE_TAG_F64 = 524292,

  /// non IEEE standard bfloat16 implemented by J6 VPU, with 2 bytes storage
  ///
  /// 8 bits for significand and 8 bits for exponent
  /// \warning
  /// Some IEEE feature is not supported, such as NaN and denormal number
  HBRT4_TYPE_TAG_VPU_BF16 = 131077,

  /// non IEEE standard float32 implemented by J6 VPU, with 2 bytes storage
  ///
  /// 24 bits for significand and 8 bits for exponent
  /// \warning
  /// Some IEEE feature is not supported, such as NaN and denormal number
  HBRT4_TYPE_TAG_VPU_F32 = 262150,

  /// Tensor type
  HBRT4_TYPE_TAG_TENSOR = 65536,

  /// Pack of elements
  HBRT4_TYPE_TAG_TUPLE = 65537,

  /// Memory type which corresponds to a memspace
  /// Variable of this type do not have data that can be analyzed by user
  HBRT4_TYPE_TAG_MEMORY = 65538,

  /// Array of values. Corresponds to `std::vector` in C++
  HBRT4_TYPE_TAG_ARRAY = 65539,

  /// String of values, Corresponds to `std::string` in C++
  HBRT4_TYPE_TAG_STRING = 65540,
};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4TypeTag);

/// The quantization method
enum Hbrt4QuantizationMethod {
  /// Unknown method
  /// This value outputted only when \ref hbrt4TypeGetQuantizationMethod errors
  HBRT4_QUANTIZATION_METHOD_UNKNOWN = 0,

  /// No quantization is used
  HBRT4_QUANTIZATION_METHOD_NONE = 1,

  /// Default method
  ///
  /// Use \ref hbrt4TypeGetDefaultQuantizationInfo to get the quantization info
  HBRT4_QUANTIZATION_METHOD_DEFAULT = 2,

};
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4QuantizationMethod);

/// @}

HBRT4_PRIV_C_EXTERN_C_END
