/// \file
/// \ref Variable Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Batch.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/OperatorEnum.h"
#include "hbrt4-c/Detail/Enums/VariableEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Variable Variable
///
/// Each variable represent a
/// [Static Single Assignment](https://en.wikipedia.org/wiki/Static_single-assignment_form) variable
/// \since v4.0.1
///
/// # List of Features
/// - The variable information is read-only, and cannot be modified
/// Provides the following information of graph:
/// - Variable number of batches
/// - Variable name
/// - Variable description
/// - Memspace information
/// - Child variable of variable
///
/// # Task Sequence
///
/// - Get \ref Hbrt4Variable from APIs in \ref Graph module,
///   such as \ref hbrt4GraphGetInputVariable, \ref hbrt4GraphGetOutputVariable, \n
///   or get it from APIs in \ref Node module,
///   such as \ref hbrt4NodeGetInputVariable, \ref hbrt4NodeGetInputVariable
///
/// Many information may be retrieved from the variable.
/// The followings are the most frequently used variables:
///
/// - Name from \ref hbrt4VariableGetName
/// - Memspace from \ref hbrt4VariableGetMemspace. \n
///   This specifies the memory allocation requirement for the variable. \n
///   User should use this to allocate memory.
///   See \ref Memspace module for more detail
/// - Memspace offset from \ref hbrt4VariableGetOffsetInMemspace \n
///   This is where the data of variable should be located inside memspace
/// - Variable type from \ref hbrt4VariableGetType \n
///   Whether variable is tensor, array, tuple, etc should be checked here.
///   See \ref Type module for more detail
///
/// Some other modules need to use \ref Hbrt4Variable
/// - \ref Hbrt4Value is created from \ref Hbrt4Variable
///   See \ref Value module for more detail
///
/// @{

/// Get type from variable
/// \since v4.0.1
Hbrt4Status hbrt4VariableGetType(Hbrt4Variable variable, Hbrt4Type *type) HBRT4_PRIV_CAPI_EXPORTED;

/// Check if variable is constant
/// \since v4.0.1
Hbrt4Status hbrt4VariableIsConstant(Hbrt4Variable variable, bool *isConstant) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the name of variable in string format
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param[out] name The variable name
///
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetName(Hbrt4Variable variable, const char **name) HBRT4_PRIV_CAPI_EXPORTED;

/// Get description of variable
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param[out] description The variable description
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
/// - \ref HBRT4_STATUS_NOT_FOUND No description for variable
/// - \ret_ok
Hbrt4Status hbrt4VariableGetDescription(Hbrt4Variable variable, Hbrt4Description *description) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the \ref Memspace which this variable belongs to, which provides buffer allocation information,
/// see \ref Memspace for more detail.
/// \since v4.0.1
///
/// Note that this variable may not have associated memspace,
/// if the type of variable is a container type
///
/// \param_in_obj{variable}
/// \param[out] memspace Variable memspace for memory allocation requirement
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \invariant
/// Every variable only completely inside in one \ref Hbrt4Memspace,
/// or not in any \ref Hbrt4Memspace
///
/// \remark
/// Each memspace may contain 0, 1, or multiple variables
///
/// \see \ref hbrt4VariableGetOffsetInMemspace
///
/// \test_disas3_todo
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_FOUND No associated \ref Hbrt4Memspace for variable
/// - \ret_ok
Hbrt4Status hbrt4VariableGetMemspace(Hbrt4Variable variable, Hbrt4Memspace *memspace) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the address offset inside memspace, where variable should be located in memory
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param[out] offsetInMemspace Variable offset in byte, to the beginning of memspace
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \see \ref hbrt4VariableGetMemspace
///
/// \test_disas3_todo
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetOffsetInMemspace(Hbrt4Variable variable, size_t *offsetInMemspace) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the batch range, which represents which batch this variable belongs to
/// \since v4.0.1
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{variable}
/// \param[out] batchRange Batch range of the variable
///
/// \on_err_out_set_to{Hbrt4BatchRange_DefaultValue}
///
/// \mt_safe
///
/// \see \ref Hbrt4BatchRange for the explanation of batch range
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetBatchRange(Hbrt4Variable variable, Hbrt4BatchRange *batchRange) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the batch stride value of variable
/// \since v4.0.1
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{variable}
/// \param[out] batchStride Batch stride of the variable (unit: byte)
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \note
/// If this API return \ref HBRT4_STATUS_OK, and output 0, this means the batch dimension is broadcasted
///
/// \warning
/// The return value of this API checked to differ with broadcasted batc,
/// and the batch stride does not exist (because the batch dimension is reshaped, reordered during compilation, etc)
///
/// \terminology_batch
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_FOUND Batch stride does not exist
/// - \ret_ok
Hbrt4Status hbrt4VariableGetBatchStride(Hbrt4Variable variable,
                                        ptrdiff_t *batchStride) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Get the number of child variables in `variable`, only if typetag of `variable` is tuple type
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param[out] num child variables number.
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
Hbrt4Status hbrt4VariableGetTupleNumChildren(Hbrt4Variable variable, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get child variable of `variable`, only if typetag of `variable` is tuple type
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param_pos{hbrt4VariableGetTupleNumChildren}
/// \param[out] child child variable
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test
/// See API documentation in \ref Variable module
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4VariableGetTupleChild(Hbrt4Variable variable, size_t pos,
                                       Hbrt4Variable *child) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the semantic of variable, when it is used as the input of some computation requirement
/// \since v4.0.1
///
/// \param_in_obj{variable}
/// \param[out] inputSemantic Input semantic of variable
///
/// \on_err_out_set_to{HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN}
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetInputSemantic(Hbrt4Variable variable,
                                          Hbrt4VariableInputSemantic *inputSemantic) HBRT4_PRIV_CAPI_EXPORTED;

/// Get \ref Value of const \ref Variable
/// This Value should not be destroyed
///
/// \not_impl_api
/// \do_not_test_in_j5_toolchain
///
/// \param_in_obj{variable}
/// \param[out] value Const value of variable
///
/// \on_err_out_zero
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetConstValue(Hbrt4Variable variable, Hbrt4Value *value) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the destination dims for resizer input model
/// \since v4.0.7
///
/// This API only succeeds if \ref hbrt4VariableGetInputSemantic gives \ref HBRT4_VARIABLE_INPUT_SEMANTIC_RESIZER
///
/// \param_in_obj{variable}
/// \param[out] dims destination dims of variable for resizer input model
///
/// \on_err_out_zero
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetResizerDestDims(Hbrt4Variable variable,
                                            Hbrt4PtrdiffTArrayRef *dims) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the special operator that defines the value of this variable,
/// for operators that requires special handling
///
/// \param_in_obj{variable}
/// \param[out] specialOperator The category of the operator that defines the value of variable,
/// that requires special handling
///
/// \do_not_test_in_j5_toolchain
/// \on_err_out_set_to{HBRT4_SPECIAL_OPERATOR_UNKNOWN}
/// \mt_safe
///
/// \note
/// Output \ref HBRT4_SPECIAL_OPERATOR_NORMAL, if the defining operator do not
/// require special handling
///
/// \returns
/// - \ref HBRT4_STATUS_NOT_FOUND This operator has no defining operator (This is the input of the graph)
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VariableGetDefiningSpecialOperator(Hbrt4Variable variable,
                                                    Hbrt4SpecialOperator *specialOperator) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
