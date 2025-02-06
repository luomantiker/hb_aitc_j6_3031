/// \file
/// \ref Node Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DeviceEnum.h"
#include "hbrt4-c/Detail/Enums/NodeEnum.h"
#include "hbrt4-c/Detail/Enums/VariableEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Node Node
/// Get static computational node information in graph
///
/// # List of Features
/// - The node information is read-only, and cannot be modified
/// Provides the following information of node:
/// - Node name
/// - Node operation name
/// - List of input/output/parameter variables, and all other variables
/// - Usages of variables
/// - List of memspaces
/// - Node dependencies
/// - Estimated latency of node
///
/// # Task Sequence
/// To execute node:
/// - Get \ref Hbrt4Node from \ref Hbrt4Graph by \ref hbrt4GraphGetNode
/// - Get \ref Hbrt4Memspace by \ref hbrt4NodeGetMemspace
/// - Allocate memory buffer according to memspaces and creates \ref Hbrt4Buffer
///   See \ref Memspace module and \ref Buffer module for more detail
/// - Get list of inputs variable by \ref hbrt4NodeGetInputVariable,
///   fill in the input data if needed.
/// - Check \ref Hbrt4DeviceCategory by \ref hbrt4NodeGetDeviceCategory
///
/// ---
///
/// If device category is BPU ( \ref HBRT4_DEVICE_CATEGORY_BPU )
/// - Use \ref hbrt4NodeGetParameterVariable to get list of variables,
///   which are needed for bpu execution.
/// - Use APIs in \ref Command module and \ref Pipeline module.
///   Read those modules for detail
///
/// ---
///
/// If device category is CPU ( \ref HBRT4_DEVICE_CATEGORY_CPU )
/// - Use \ref hbrt4NodeGetParameterVariable to get list of variables,
///   which are needed for bpu execution.
/// - Call the APIs of \ref HBTL
///
/// @{

/// Get the node name in string format
///
/// \param_in_obj{node}
/// \param[out] name Null terminated node name
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
Hbrt4Status hbrt4NodeGetName(Hbrt4Node node, const char **name) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the node operation name, used to identify which function to call for CPU operator
///
/// \note
/// This operation always exists, even for BPU operation.
/// libdnn team and compiler team should discussion for a naming convention here
///
/// \param_in_obj{node}
/// \param[out] operationName Null terminated operation name
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
Hbrt4Status hbrt4NodeGetOperationName(Hbrt4Node node, const char **operationName) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the pseudo tag of node, which indicates this node does not need to be executed
/// This node exists to help graph analysis
/// \do_not_test_in_j5_toolchain
Hbrt4Status hbrt4NodeGetPseudoTag(Hbrt4Node node, Hbrt4NodePseudoTag *pseudoTag) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of memspaces in graph
///
/// \param_in_obj{node}
/// \param[out] num Node memspace number
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
Hbrt4Status hbrt4NodeGetNumMemspaces(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a memspace in node with corresponding position,
/// it is used for the memory allocation requirement
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumMemspaces}
/// \param[out] memspace memspace for memory allocation information. See \ref Memspace module for detail \n
/// This is guaranteed to be one of the memspaces in \ref hbrt4GraphGetMemspace of the corresponding \ref
/// Hbrt4Graph
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
Hbrt4Status hbrt4NodeGetMemspace(Hbrt4Node node, size_t pos, Hbrt4Memspace *memspace) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of variables in node
///
/// \param_in_obj{node}
/// \param[out] num Node variables number.
/// This does **NOT** include constant variable
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
Hbrt4Status hbrt4NodeGetNumVariables(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a variable with corresponding position in node
///
/// \remark
/// Use the order of variables listed in this API to bind CPU function
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumVariables}
/// \param[out] variable An variable. See \ref Variable module for detail
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
Hbrt4Status hbrt4NodeGetVariable(Hbrt4Node node, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the usage of variable
Hbrt4Status hbrt4NodeGetVariableUsage(Hbrt4Node node, Hbrt4Variable variable,
                                      Hbrt4VariableNodeUsage *usage) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of input variables in node, without constant variable
///
/// \param_in_obj{node}
/// \param[out] num Node input variable number.
/// This does **NOT** include constant variable
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
Hbrt4Status hbrt4NodeGetNumInputVariables(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a input variable with corresponding position in node,
/// used for variable data information analysis
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumInputVariables}
/// \param[out] variable Input variable for data information. See \ref Variable module for detail
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
Hbrt4Status hbrt4NodeGetInputVariable(Hbrt4Node node, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of output variables in graph
///
/// \param_in_obj{node}
/// \param[out] num Node output variable number
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
Hbrt4Status hbrt4NodeGetNumOutputVariables(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get a output variable with corresponding position in node,
/// used for variable data information analysis
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumOutputVariables}
/// \param[out] variable Output variable for data information. See \ref Variable module for detail
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
Hbrt4Status hbrt4NodeGetOutputVariable(Hbrt4Node node, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// Get number of ancestor nodes in current node,
/// which needs to be run before the current node
///
/// \param_in_obj{node}
/// \param[out] num Number of ancestor nodes
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
Hbrt4Status hbrt4NodeGetNumAncestors(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get ancestor node of the current node with corresponding position,
/// which needs to be run before the current node TODO(wurudiong: current node?)
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumAncestors}
/// \param[out] ancestor The ancestor node
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
Hbrt4Status hbrt4NodeGetAncestor(Hbrt4Node node, size_t pos, Hbrt4Node *ancestor) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the device category of node, to determine whether the node runs on CPU or BPU
///
/// \param_in_obj{node}
/// \param[out] category Device category
///
/// \on_err_out_set_to{HBRT4_DEVICE_CATEGORY_UNKNOWN}
///
/// \mt_safe
///
/// \test
/// Check "funccalls" field appears in "segment" of json output of \ref hbdk3-disas
/// to determine whether the node runs on bpu or cpu
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4NodeGetDeviceCategory(Hbrt4Node node, Hbrt4DeviceCategory *category) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of cores used by node
///
/// \param_in_obj{node}
/// \param[out] num Number of cores
///
/// \on_err_out_zero
///
/// \note
/// Output zero means HBRT does not care the number of cores used \n
/// This is usually the case for cpu node
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4NodeGetNumCores(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the estimated latency in microsecond of node
///
/// \param_in_obj{node}
/// \param[out] latencyInMicro The estimated latency in microsecond
///
/// \on_err_out_zero
///
/// \mt_safe
///
/// \remark
/// - Will only implement this API for BPU node \n
/// All cpu node will return error for this API
/// - This latency is fixed value, and determined during compilation, \n
/// will never change according to the input data
/// - For dual BPU core models, this latency is the maximum latency of two cores, **NOT** the sum.
///
/// \warning
/// - The estimation is calculated, assuming that there are enough bandwidth available
/// to BPU on the system, so BPU can reach its maximum theoretical latency \n
/// If non-BPU devices uses up too much latency, the estimation will largely bias
/// - The latency of some node will vary largely depending on the input value. \n
/// For these node, the estimation will be determined by experience \n
/// If the input data is out of the compiler team's expectation, \n
/// the estimation will largely bias
///
///
/// #### Further ideas
/// This is just ideas. May or may not implemented. \n
/// Add API to get the user the min, max and standard deviation? \n
/// This will introduce another object type
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_FOUND The estimated latency not recorded for the node
/// - \ret_ok
Hbrt4Status hbrt4NodeGetEstimatedLatencyMicros(Hbrt4Node node, uint64_t *latencyInMicro) HBRT4_PRIV_CAPI_EXPORTED;

/// Get number of parameters in node
///
/// \param_in_obj{node}
/// \param[out] num Number of parameter variables
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
Hbrt4Status hbrt4NodeGetNumParameterVariables(Hbrt4Node node, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get parameters of node with corresponding position
///
/// \param_in_obj{node}
/// \param_pos{hbrt4NodeGetNumParameterVariables}
/// \param[out] variable The parameter variable
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
Hbrt4Status hbrt4NodeGetParameterVariable(Hbrt4Node node, size_t pos, Hbrt4Variable *variable) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
