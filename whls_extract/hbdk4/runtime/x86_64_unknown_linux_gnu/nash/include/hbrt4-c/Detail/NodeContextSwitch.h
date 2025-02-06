/// \file
/// \ref ContextSwitch Module
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

/// \ingroup Node
///
/// \defgroup ContextSwitch Context Switch Nodes
///
/// Special node to backup and restore bpu L1m,
/// to ensure the correct behavior when using the funccall preemption functionality
///
/// This means to the corresponding bpu \ref Hbrt4Command are to submitted with high priority, \n
/// which may be run when low priority bpu \ref Hbrt4Command are running in the middle
///
/// # Usage
///
/// ## Acquire
/// - Get special nodes by \ref hbrt4InstanceGetDumpBpuContextNode and \ref hbrt4InstanceGetRestoreBpuContextNode
/// ## Use
/// - Get memspaces info of these node using APIs in \ref Memspace module
/// - Allocate buffer using APIs in \ref Buffer module. \n
/// Note that each of above two nodes should use the same buffer
/// - Create command by APIs in \ref Command
/// - In \ref Pipeline APIs, push \ref Command corresponding to \ref hbrt4InstanceGetDumpBpuContextNode
/// - Then push all other commands that need to be submitted with high priority
/// - Then push \ref Command corresponding to \ref hbrt4InstanceGetRestoreBpuContextNode
///
/// ## Destroy
/// - No need to destroy
///
/// @{

/// Get the node to dump bpu context
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{instance}
/// \param[out] node Node to dump bpu context
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \see \ref ContextSwitch Module for usage
///
/// \internal
/// In the current implementation, this node has one memspace and one output tensor \n
/// The memspace and tensor is the shared with \ref hbrt4InstanceGetRestoreBpuContextNode
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4InstanceGetDumpBpuContextNode(Hbrt4Instance instance, Hbrt4Node *node) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the node to restore bpu context
///
/// \do_not_test_in_j5_toolchain
/// \param_in_obj{instance}
/// \param[out] node Node to restore bpu context
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \see \ref ContextSwitch Module for usage
///
/// \internal
/// In the current implementation, this node has one memspace and one output tensor \n
/// The memspace and tensor is the shared with \ref hbrt4InstanceGetDumpBpuContextNode
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4InstanceGetRestoreBpuContextNode(Hbrt4Instance instance, Hbrt4Node *node) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
