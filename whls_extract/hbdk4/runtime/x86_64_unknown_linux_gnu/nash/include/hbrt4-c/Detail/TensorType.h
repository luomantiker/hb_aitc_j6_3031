/// \file
/// \ref TensorType Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Description.h"
#include "hbrt4-c/Detail/Enums/TensorEnum.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \ingroup Type
/// \defgroup TensorType TensorType
/// A competent to express tensor dim and layout information
///
/// # Usage
///
/// ## Use
/// - Get the element type by \ref hbrt4TypeGetTensorElementType
/// - Get the quantization info by running \ref hbrt4TypeGetQuantizationMethod and
///   \ref hbrt4TypeGetDefaultQuantizationInfo on the output \ref hbrt4TypeGetTensorElementType
/// - Get dimensions by \ref hbrt4TypeGetTensorDims
/// - Get the tensor memory layout by \ref hbrt4TypeGetTensorStrides and \ref hbrt4TypeGetTensorEncoding
/// ## Destroy
/// - No need to destroy
///
/// @{

/// Get the element type of tensor type
///
/// \param_in_obj{tensorType}
/// \param[out] elementType Tensor element type
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test_disas3{"element type name"}
///
/// \note
/// Get quantization information by \ref hbrt4TypeGetQuantizationMethod
/// and \ref hbrt4TypeGetDefaultQuantizationInfo on the output `elementType`
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4TypeGetTensorElementType(Hbrt4Type tensorType, Hbrt4Type *elementType) HBRT4_PRIV_CAPI_EXPORTED;

/// Get encoding method of tensor
///
/// \do_not_test_in_j5_toolchain
/// \todo
/// (hehaoqian): Explain this
///
/// \param_in_obj{tensorType}
/// \param[out] encoding The tensor encoding
///
/// \on_err_out_set_to{HBRT4_TENSOR_ENCODING_UNKNOWN}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4TypeGetTensorEncoding(Hbrt4Type tensorType, Hbrt4TensorEncoding *encoding) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the dimensions of tensor, the return dimension can be static and dynamic
///
/// \param_in_obj{tensorType}
/// \param[out] dims The tensor dimensions
///
/// \attention
/// - If the number of dims is not statically known, return \ref HBRT4_STATUS_DYNAMIC_VALUE \n
/// User must check the return value to determine whether the num of dimensions is dynamic or 0
/// - If the number of dims is known, output is given normally.
/// If any single dimension in output array is not statically known,
/// it will have the value of `PTRDIFF_MIN` in the output array
///
/// \remarks
/// This dims also contains the dim that represent the batch
///
/// \on_err_out_empty_array
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_DYNAMIC_VALUE The num of dims is not statically known
/// - \ret_ok
Hbrt4Status hbrt4TypeGetTensorDims(Hbrt4Type tensorType, Hbrt4PtrdiffTArrayRef *dims) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the stride of tensor to determine the memory layout method
///
/// \invariant
/// - The number of strides is always the same as number of dims (\ref hbrt4TypeGetTensorDims)
/// - The relationship between dim and stride is one-to-one
///
/// \remarks
/// - This is a standard strides as defined in numpy ndarray
/// - This effectively the same stride of `hbrt4TensorTypeGetDims` in \ref hbrt3
///
/// \remarks
/// These strides also **CONTAINS** batch stride as returned by
/// \ref hbrt4VariableGetBatchStride, if batch stride exists
///
/// \see <https://numpy.org/doc/stable/reference/arrays.ndarray.html> \n
/// In the section of "Internal memory layout of an ndarray"
///
/// \param_in_obj{tensorType}
/// \param[out] strides The tensor strides
///
/// \attention
/// - If the number of dims is not statically known, return \ref HBRT4_STATUS_DYNAMIC_VALUE \n
/// User must check the return value to determine whether the num of dimensions is dynamic or 0
/// - If the number of dims is known, output is given normally.
/// If any single stride in output array is not statically known,
/// it will have the value of `PTRDIFF_MIN` in the output array
///
/// \on_err_out_empty_array
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_DYNAMIC_VALUE The num of dims is not statically known
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION The graph is using deprecated old layout
/// and cannot represent memory layout by \ref hbrt4TypeGetTensorStrides.
/// - \ret_ok
Hbrt4Status hbrt4TypeGetTensorStrides(Hbrt4Type tensorType, Hbrt4PtrdiffTArrayRef *strides) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
