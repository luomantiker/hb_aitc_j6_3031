/// \file
/// \ref Graph Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/GraphEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \ingroup Graph
/// \defgroup GraphGroup GraphGroup
///
/// Group of graphs, where graphs share similarities
///
/// @{

/// Get number of graphs inside graph group
///
/// \param_in_obj{graphGroup}
/// \param[out] num Number of graphs inside graph group
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGroupGetNumGraphs(Hbrt4GraphGroup graphGroup, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get graph inside graph group
///
/// \param_in_obj{graphGroup}
/// \param_pos{hbrt4GraphGroupGetNumGraphs}
/// \param[out] graph Graph inside graph group
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
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
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4GraphGroupGetGraph(Hbrt4GraphGroup graphGroup, size_t pos, Hbrt4Graph *graph) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the classification of graph group to show why graphs are grouped together
///
/// \param_in_obj{graphGroup}
/// \param[out] classification Graph group classification
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \on_err_out_set_to{HBRT4_GRAPH_GROUP_CLASSIFICATION_UNKNOWN}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGroupGetClassification(Hbrt4GraphGroup graphGroup,
                                             Hbrt4GraphGroupClassification *classification) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the graph group name in string format, only support string, no binary
///
/// \param_in_obj{graphGroup}
/// \param[out] name Null terminated graph group name
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
Hbrt4Status hbrt4GraphGroupGetName(Hbrt4GraphGroup graphGroup, const char **name) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
