/// \file
/// \ref Value Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Value Value
/// Value of variable, used to register user provided information to \ref Command
///
/// # List of Features
/// - Bind buffer of variable
/// - Bind tensor dims and strides
///
/// # Task Sequence
///
/// - Get \ref Hbrt4Variable according to doc in \ref Variable module;
/// - Create \ref Hbrt4ValueBuilder with \ref Hbrt4Variable by \ref hbrt4ValueBuilderCreate
/// - If the variable is of tuple types, then first create \ref Hbrt4Value
///   for its sub variables acquired by \ref hbrt4VariableGetTupleChild, \n
///   then set them by \ref hbrt4ValueBuilderSetTupleSubValue
/// - Otherwise, Set buffer by \ref hbrt4ValueBuilderSetBuffer
/// - If variable is of tensor type, and
///   the tensor dims or strides of variable is not statically known, set them by
///   \ref hbrt4ValueBuilderSetTensorDims and \ref hbrt4ValueBuilderSetTensorStrides
/// - Then create \ref Hbrt4Value from \ref Hbrt4ValueBuilder by
///   \ref hbrt4ValueBuilderInto
/// - Bind \ref Hbrt4Value to \ref Hbrt4Command by \ref hbrt4CommandBuilderBindValue
///   See \ref Command module for detail
/// - When \ref Hbrt4Value is no longer used that
///   all \ref Hbrt4Command that uses it has been destroyed,
///   it should be destroyed with \ref hbrt4ValueDestroy
///
/// @{

/// Init ValueBuilder from variable
///
/// \param_in_obj{variable}
/// \param[out] valueBuilder Builder to set value
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4ValueBuilderInto}
///
/// \mt_safe
///
/// \test
/// Ensure this API is multi thread safe, by running this API in multiple thread on the same node
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4ValueBuilderCreate(Hbrt4Variable variable, Hbrt4ValueBuilder *valueBuilder) HBRT4_PRIV_CAPI_EXPORTED;

/// Bind buffer of Value. Use this only if this type needs buffer
Hbrt4Status hbrt4ValueBuilderSetBuffer(Hbrt4ValueBuilder builder, Hbrt4Buffer buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Add sub Value for value of tuple type
///
/// \param_in_obj{builder}
/// \param[in] pos Position of sub Value
/// \param[in] value sub Value
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
///
/// \deprecated
/// This function has been deprecated in after version 4.0.7
/// Use `hbrt4ValueBuilderSetSubValue` instead
Hbrt4Status hbrt4ValueBuilderSetTupleSubValue(Hbrt4ValueBuilder builder, size_t pos,
                                              Hbrt4Value value) HBRT4_PRIV_CAPI_EXPORTED
    HBRT4_DEPRECATED("This function has been deprecated in versions after 4.0.7", "hbrt4ValueBuilderSetSubValue");

/// Add sub Value for value of tuple type
///
/// \param_in_obj{builder}
/// \param[in] pos Position of sub Value
/// \param[in] value sub Value
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4ValueBuilderSetSubValue(Hbrt4ValueBuilder builder, size_t pos,
                                         Hbrt4Value value) HBRT4_PRIV_CAPI_EXPORTED;

/// Set tensor dims, used for dynamic shaped tensor.
///
/// \param_in_obj{builder}
/// \param[in] dims new tensor dims
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4ValueBuilderSetTensorDims(Hbrt4ValueBuilder builder,
                                           Hbrt4PtrdiffTArrayRef dims) HBRT4_PRIV_CAPI_EXPORTED;

/// Set tensor strides, used for dynamic shaped tensor.
///
/// \param_in_obj{builder}
/// \param[in] strides Tensor strides
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4ValueBuilderSetTensorStrides(Hbrt4ValueBuilder builder,
                                              Hbrt4PtrdiffTArrayRef strides) HBRT4_PRIV_CAPI_EXPORTED;

/// Get buffer of Value. Use this only if value has no sub value
Hbrt4Status hbrt4ValueGetBuffer(Hbrt4Value value, Hbrt4Buffer *buffer) HBRT4_PRIV_CAPI_EXPORTED;

/// Get num of sub values in value of tuple type
Hbrt4Status hbrt4ValueGetNumSubValues(Hbrt4Value value, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get sub value in value of tuple type
Hbrt4Status hbrt4ValueGetSubValue(Hbrt4Value value, size_t pos, Hbrt4Value *child) HBRT4_PRIV_CAPI_EXPORTED;

/// Get dimensions of the value
Hbrt4Status hbrt4ValueGetTensorDims(Hbrt4Value value, Hbrt4PtrdiffTArrayRef *dims) HBRT4_PRIV_CAPI_EXPORTED;

/// Get strides of the value
Hbrt4Status hbrt4ValueGetTensorStrides(Hbrt4Value value, Hbrt4PtrdiffTArrayRef *strides) HBRT4_PRIV_CAPI_EXPORTED;

/// Convert ValueBuilder to Value
///
/// \param_in_obj{valueBuilder}
/// \param_builder_into_out{value}
///
/// \lifetime_builder{hbrt4ValueDestroy}
///
/// \mt_unsafe_mut
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4ValueBuilderInto(Hbrt4ValueBuilder *valueBuilder, Hbrt4Value *value) HBRT4_PRIV_CAPI_EXPORTED;

/// Destroy Value
///
/// \param_dtor{value}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION `value` is still hold by one alive
/// - \ret_ok
Hbrt4Status hbrt4ValueDestroy(Hbrt4Value *value) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
