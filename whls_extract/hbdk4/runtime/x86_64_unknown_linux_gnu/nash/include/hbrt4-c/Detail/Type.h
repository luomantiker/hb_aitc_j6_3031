/// \file
/// \ref Type Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/TypeEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Type Type
/// Get type enum of tensor, and quantization information is optional
///
/// Mainly used to get the element type and quantization information of tensors
///
/// # List of Features
/// - The graph information is read-only, and cannot be modified
///
/// Provides the following information of type:
/// - Type category (Tensor, array, string, etc)
/// - Quantization information, including scales, zero points
/// - Type dims and strides
/// - Number of elements in array
///
/// # Task Sequence
///
/// - Get \ref Hbrt4Type from hbrt4VariableGetType
/// - Check if type is tensor, tuple, array, etc, by \ref hbrt4TypeGetTag
/// - For tensor or array type, get the element type by \ref hbrt4TypeGetElementType
/// - For array type, get number of elements by \ref hbrt4TypeGetArrayNumElements
/// - Quantization info can be acquired by \ref hbrt4TypeGetDefaultQuantizationInfo
///
/// @{

/// Structure to express quantization information
///
/// #### Formula
/// TODO: To be described by zhaozhuoran
/// #### Default Value {#Hbrt4DefaultQuantizationInfo_DefaultValue}
/// - floatType is set the the quantized type itself
/// - All array ref fields are set to empty
/// - quantizedChannelIndex is set to `INT32_MIN`
/// - Other intger fields are set to 0
struct Hbrt4DefaultQuantizationInfo {
  /// The type before quantization, usually a F32 type
  Hbrt4Type expressedType;

  /// The scale value(s)
  ///
  /// Multiple values if per channel quantization, one value otherwise
  Hbrt4Float32ArrayRef scales;

  /// The zero point value(s)
  ///
  /// Multiple values if per channel quantization, one value otherwise
  Hbrt4Int32ArrayRef zeroPoints;

  /// The channel for per channel quantization
  ///
  /// \invariant
  /// This is set to `INT32_MIN` if per channel quantization not used
  int32_t quantizedChannelAxis;

  /// The minimum value when quantized to integer
  int32_t storageMin;

  /// The maximum value when quantized to integer
  int32_t storageMax;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4DefaultQuantizationInfo);
/// \endcond

/// Get the type tag
///
/// \param_in_obj{type}
/// \param[out] tag Type tag
///
/// \on_err_out_set_to{HBRT4_TYPE_TAG_UNKNOWN}
///
/// \mt_safe
///
/// \see \ref Hbrt4TypeTag for detail
///
/// \test_disas3{"element type name"}
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4TypeGetTag(Hbrt4Type type, Hbrt4TypeTag *tag) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the quantization method of tensor from `type` structure
///
/// \param_in_obj{type}
/// \param[out] method Quantization method
///
/// \on_err_out_set_to{HBRT4_QUANTIZATION_METHOD_UNKNOWN}
///
/// \mt_safe
///
/// \test_disas3_todo
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4TypeGetQuantizationMethod(Hbrt4Type type, Hbrt4QuantizationMethod *method) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the quantization info by default quantization method.
/// Call this only if the output of \ref hbrt4TypeGetQuantizationMethod is \ref HBRT4_QUANTIZATION_METHOD_DEFAULT
///
/// \param_in_obj{type}
/// \param[out] info Quantization info
///
/// \on_err_out_set_to{Hbrt4DefaultQuantizationInfo_DefaultValue}
/// \lifetime_getter
///
/// \mt_safe
///
/// \test_disas3_todo
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Output of \ref hbrt4TypeGetQuantizationMethod is not \ref
/// HBRT4_QUANTIZATION_METHOD_DEFAULT
/// - \ret_ok
Hbrt4Status hbrt4TypeGetDefaultQuantizationInfo(Hbrt4Type type,
                                                Hbrt4DefaultQuantizationInfo *info) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the element type of tensor/array type
///
/// \param_in_obj{type}
/// \param[out] elementType Element type
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
Hbrt4Status hbrt4TypeGetElementType(Hbrt4Type type, Hbrt4Type *elementType) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of elements in an array, only available if the `typetag` is array
///
/// \param_in_obj{type}
/// \param[out] num Array element number
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \test_disas3_todo
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4TypeGetArrayNumElements(Hbrt4Type type, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
