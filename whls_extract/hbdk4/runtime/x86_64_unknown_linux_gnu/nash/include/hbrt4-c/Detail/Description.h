/// \file
/// \ref Description Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DescriptionEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Description Description
/// User defined description
/// This is used by the user to add custom information to
/// hbm, graph or tensor, to help post processing the data.
///
/// # List of Features
/// - Encode text or binary description of graph
/// - Encode text or binary description of tensor
///
/// The user usually choose to embed some Json string,
/// but some custom binary data is also possible
///
/// # Task Sequence
///
/// - Get description by \ref hbrt4HbmGetDescription, or \ref hbrt4GraphGetDescription or \ref
/// hbrt4VariableGetDescription
/// - Then check the description category by \ref hbrt4DescriptionGetCategory
/// - Get the data by \ref hbrt4DescriptionGetData
/// - Do something according to the data
///
/// @{

/// Get the data stored inside description,
/// data result includes pointer and length
///
/// \param_in_obj{description}
/// \param[out] data Data inside description
///
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// \remark
/// You can get the data of description regardless of the output of
/// \ref hbrt4DescriptionGetCategory
///
/// \mt_safe
///
/// \invariant
/// The output is always null terminated string
///
/// \warning
/// If the output of \ref hbrt4DescriptionGetCategory is
/// \ref HBRT4_DESCRIPTION_CATEGORY_BINARY,
/// null character may appear in the middle of string
///
/// \test
/// Check if the output is always null terminated with correct length,
/// regardless whether error or not. \n
/// See other tests in
/// \ref hbrt4HbmGetDescription, or \ref hbrt4GraphGetDescription or \ref hbrt4VariableGetDescription
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4DescriptionGetData(Hbrt4Description description, Hbrt4CStringRef *data) HBRT4_PRIV_CAPI_EXPORTED;

/// Get category of description, the category result may be string or binary
///
/// \param_in_obj{description}
/// \param[out] category Description Category
///
/// \on_err_out_set_to{HBRT4_DESCRIPTION_CATEGORY_UNKNOWN}
/// \invariant
/// The output is NEVER \ref HBRT4_DESCRIPTION_CATEGORY_UNKNOWN
/// if this API returns \ref HBRT4_STATUS_OK
///
/// \mt_safe
///
/// \test
/// If output is \ref HBRT4_DESCRIPTION_CATEGORY_STRING,
/// then data from \ref hbrt4DescriptionGetData
/// should be valid UTF-8 string without null character in the middle. \n
/// Search on the internet how to the UTF-8 validation.
/// \test
/// If output is \ref HBRT4_DESCRIPTION_CATEGORY_BINARY,
/// the output should not be a valid UTF-8 string
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4DescriptionGetCategory(Hbrt4Description description,
                                        Hbrt4DescriptionCategory *category) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
