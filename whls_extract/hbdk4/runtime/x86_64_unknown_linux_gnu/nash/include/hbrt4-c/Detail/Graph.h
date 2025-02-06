/// \file
/// \ref Graph Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DeviceEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Graph Graph
/// A computational graph to express model information,
/// such as toolkit version, description, graph name, node,
/// memspace, input and output variables
///
/// # Multiple batch graphs support
///
/// Users can compile models with multiple batch counts,
/// to supports models with dynamic batch during actual execution.
/// For example, user can compile models with 1,2,3 batch counts,
/// and there will be 3 graphs inside hbm, which are inside same graph group
///
/// # List of Features
/// - The graph information is read-only, and cannot be modified
/// Provides the following information of graph:
/// - Graph toolkit version
/// - Graph number of batches
/// - Graph Name
/// - List of input/output variables
/// - List of memspaces
/// - List of nodes
/// - Get information of graph group, which groups similar graphs together
///
/// # Task Sequence
///
/// - Get graph group (\ref Hbrt4GraphGroup) by \ref hbrt4HbmGetGraphGroup and \ref hbrt4HbmGetNumGraphGroups,
/// or by \ref hbrt4HbmGetGraphGroupByName
/// - Check the number of graphs in graph group by \ref hbrt4GraphGroupGetNumGraphs
///
/// If there is only 1 graph in graph group:
/// - Get \ref Hbrt4Graph in group by \ref hbrt4GraphGroupGetGraph
/// - Get the information of graphs by its getter functions such as
///   \ref hbrt4GraphGetMemspace, \ref hbrt4GraphGetNode, \ref hbrt4GraphGetInputVariable
/// - Continue to process using APIs in other modules.
///   See \ref Memspace module, \ref Node module, \ref Variable module for detail
///
/// Otherwise, if there are multiple graphs in graph group:
///
/// If there are multiple graphs in graph groups,
/// Check the output of \ref hbrt4GraphGroupGetClassification
///
/// If the output is \ref HBRT4_GRAPH_GROUP_CLASSIFICATION_BATCH
/// then these graphs have same structure, but different batch number.
/// This is used to support the multiple batch feature.
/// - Get the graphs by \ref hbrt4GraphGroupGetGraph
/// - Get their batches by \ref hbrt4GraphGetNumBatches
/// - Select the graph to execute according to the batch number
///
/// @{

/// The version of HBDK toolkit that compiles the graph
///
/// \param_in_obj{graph}
/// \param[out] version The toolchain version that compiles the graph
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test_disas3{"hbdk-cc version"}
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetToolkitVersion(Hbrt4Graph graph, Hbrt4Version *version) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the graph name in string format, only support string, no binary
///
/// \param_in_obj{graph}
/// \param[out] name Null terminated graph name
///
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// \mt_safe
///
/// \test_disas3{"model name"}
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetName(Hbrt4Graph graph, const char **name) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the graph description in string or binary format
///
/// \param_in_obj{graph}
/// \param[out] description Graph description
///
/// \on_err_out_null_obj
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
/// - \ref HBRT4_STATUS_NOT_FOUND No description associated with the graph
/// - \ret_ok
Hbrt4Status hbrt4GraphGetDescription(Hbrt4Graph graph, Hbrt4Description *description) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of nodes in graph
///
/// \param_in_obj{graph}
/// \param[out] num Graph node number
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \test_disas3{"segment num"}
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetNumNodes(Hbrt4Graph graph, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a node with corresponding position in graph
///
/// \param_in_obj{graph}
/// \param[in] pos Index in range of [0, \ref hbrt4GraphGetNumNodes )
/// \param[out] node Graph node
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test
/// See API document in \ref Node module
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4GraphGetNode(Hbrt4Graph graph, size_t pos, Hbrt4Node *node) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of memspaces in graph
///
/// \param_in_obj{graph}
/// \param[out] num Graph memspace number
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
Hbrt4Status hbrt4GraphGetNumMemspaces(Hbrt4Graph graph, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a memspace with corresponding position in graph
///
/// \param_in_obj{graph}
/// \param[in] pos Index in range of [0, \ref hbrt4GraphGetNumMemspaces )
/// \param[out] memspace memspace for memory allocation information. See \ref Memspace module for detail
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test
/// See API documentation in \ref Memspace module
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4GraphGetMemspace(Hbrt4Graph graph, size_t pos, Hbrt4Memspace *memspace) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of input variables in graph
///
/// \param_in_obj{graph}
/// \param[out] num Graph input variable number
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \test
/// The sum of array size of "input features"
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetNumInputVariables(Hbrt4Graph graph, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a input variable with corresponding position in graph
///
/// \param_in_obj{graph}
/// \param[in] pos Index in range of [0, \ref hbrt4GraphGetNumInputVariables )
/// \param[out] variable Input variable in graph
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test
/// - Variables are in the fields of "input features"
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4GraphGetInputVariable(Hbrt4Graph graph, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of output variables in graph
///
/// \param_in_obj{graph}
/// \param[out] num Graph output variable number
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \test
/// The sum of array size of "output features"
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetNumOutputVariables(Hbrt4Graph graph, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a output variable with corresponding position in graph
///
/// \param_in_obj{graph}
/// \param[in] pos Index in range of [0, \ref hbrt4GraphGetNumOutputVariables )
/// \param[out] variable Output variable in graph
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \test
/// - Variables are in the fields of "output features"
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4GraphGetOutputVariable(Hbrt4Graph graph, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of batches in graph
///
/// \param_in_obj{graph}
/// \param[out] num Batch number
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \test_disas3_todo
///
/// \see The explanation of batch in \ref batch_definition
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GraphGetNumBatches(Hbrt4Graph graph, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
