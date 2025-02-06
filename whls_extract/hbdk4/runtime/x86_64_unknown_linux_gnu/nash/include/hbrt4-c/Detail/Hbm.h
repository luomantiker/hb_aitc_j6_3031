/// \file
/// \ref Hbm Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/ArrayRef.h"
#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Enums/DeviceEnum.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Hbm Hbm
/// Hbm file, which is a compiled model.
/// It includes model information such as BPU march, model description,
/// multiple graphs and toolkit version
///
/// # List of Features
/// - Load Hbm into memory
/// - Unload hbm and release all memory
/// - Get the toolkit version and Bpu march of Hbm, without loading it
/// - Get the list of graphs in Hbm
/// - Get the list of graph groups in Hbm
/// - Get one graph in Hbm by name
///
/// # Task Sequence
/// To get limited information of hbm file, without loading the entire file
/// into memory, use the \ref HbmHeader APIs.
/// These APIs are allowed to be called with \ref Hbrt4Instance with unknown march
/// - Get \ref Hbrt4HbmHeader by \ref hbrt4HbmHeaderCreateByFilename,
///   or by \ref hbrt4HbmHeaderCreateByAddress
/// - When get its hbdk toolkit version by \ref hbrt4HbmHeaderGetToolkitVersion
/// - When get its Bpu march by \ref hbrt4HbmHeaderGetBpuMarch
///
/// ---
///
/// To load the entire hbm file into file, to get entire information, and to execute hbm model
/// - Get \ref Hbrt4Hbm by \ref hbrt4HbmCreateByFilename,
///   or by \ref hbrt4HbmCreateByAddress
/// - Get the information of graph groups by
///   \ref hbrt4HbmGetGraphGroup, and \ref hbrt4HbmGetNumGraphGroups,
///   or by \ref hbrt4HbmGetGraphGroupByName
/// - Continue processing using APIs listed in \ref Graph module
/// - When \ref Hbrt4Hbm is no longer used, it should be destroyed by
///   \ref hbrt4HbmDestroy
///
/// @{

/// Bit flag to control Hbm loading behavior.
/// \internal
/// This was declared by struct with bit fields
/// But arm gcc9.3 warns with
/// parameter passing for argument of type changed in GCC 9.1
/// I concerns with the portability
enum Hbrt4HbmCreateFlag {
  /// Default behavior to load hbm
  HBRT4_HBM_CREATE_FLAG_DEFAULT = 0,

  /// Skip checksum verification
  HBRT4_HBM_CREATE_FLAG_NO_VERIFY = 1,
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_ENUM(Hbrt4HbmCreateFlag);
/// \endcond

/// Load \ref hbm_file from DDR memory with bit flag, and return hbm object
///
/// \param[in] instance Instance that related to hbm \n
/// Instance created with \ref HBRT4_BPU_MARCH_UNKNOWN is **NOT** supported.
/// \param[in] data Data of hbm in memory \n
/// HBRT does **NOT** hold any the memory in `data` after this API returns \n
/// User can safely free those memory after this API returns.
/// \param[in] flag to control the loading behavior. Unused bits must be set to zero
/// \param[out] hbm The handle to represent the hbm file loaded into the memory
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmDestroy}
///
/// \mt_safe
///
/// \note
/// This function will validate data integrity by internal checksum
/// if flag \ref HBRT4_HBM_CREATE_FLAG_NO_VERIFY is not specified
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm, \n
/// even if same \ref Hbrt4ArrayRef is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4Hbm object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4Hbm object
///
/// \remark
/// To get limited information without loading the entire hbm into memory, \n
/// Use APIs in \ref HbmHeader module
///
/// \returns
/// - \ret_empty_data
/// - \ret_null_out
/// - \ret_bad_flag
/// - \ref HBRT4_STATUS_NOT_SUPPORTED The bpu march of hbm does not match that of `instance`
/// - \ref HBRT4_STATUS_BAD_DATA `size` is not big enough to hold all data,
/// OR \ref HBRT4_HBM_CREATE_FLAG_NO_VERIFY not
/// specified, and checksum validation fails
/// - \ret_oom
/// - \ret_ok
Hbrt4Status hbrt4HbmCreateByAddress(Hbrt4Instance instance, Hbrt4ArrayRef data, Hbrt4HbmCreateFlag flag,
                                    Hbrt4Hbm *hbm) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Load \ref hbm_file by filename with bit flag, and return hbm object
///
/// \param[in] instance Instance that related to hbm \n
/// Instance created with \ref HBRT4_BPU_MARCH_UNKNOWN is **NOT** supported.
/// \param[in] filename Null-terminated filename
/// \param[in] flag to control the loading behavior. Unused bits must be set to zero
/// \param[out] hbm The handle to represent the hbm file loaded into the memory
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmDestroy}
///
/// \mt_safe
///
/// \note
/// This function will validate data integrity by internal checksum
/// if flag \ref HBRT4_HBM_CREATE_FLAG_NO_VERIFY is not specified
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm, \n
/// even if same filename is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4Hbm object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4Hbm object
///
/// \remark
/// To get limited information without loading the entire hbm into memory, \n
/// Use APIs in \ref HbmHeader module
///
/// \returns
/// - \ret_null_filename
/// - \ret_null_out
/// - \ret_bad_flag
/// - \ref HBRT4_STATUS_NOT_SUPPORTED The bpu march of hbm does not match that of `instance`
/// - \ref HBRT4_STATUS_BAD_DATA \ref HBRT4_HBM_CREATE_FLAG_NO_VERIFY not
/// specified, and checksum validation fails
/// - \ref HBRT4_STATUS_IO_ERROR File with `filename` does not exist or IO error occurs while reading file
/// - \ret_oom
/// - \ret_ok
Hbrt4Status hbrt4HbmCreateByFilename(Hbrt4Instance instance, const char *filename, Hbrt4HbmCreateFlag flag,
                                     Hbrt4Hbm *hbm) HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Destroy loaded hbm object and free all memory resources associated with it
///
/// \param_dtor{hbm}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ref HBRT4_STATUS_FAILED_PRECONDITION Any related \ref Hbrt4Command object still alive
/// - \ret_ok
Hbrt4Status hbrt4HbmDestroy(Hbrt4Hbm *hbm) HBRT4_PRIV_CAPI_EXPORTED;

/// Get toolkit version of hbm
///
/// \param_in_obj{hbm}
/// \param[out] version Toolkit version of hbm \n
/// This is the version of program that **DIRECTLY** creates the hbm, \n
/// at the last step on hbm creation \n
/// If the hbm is created by `hbdk-pack`, then it is the version of `hbdk-pack`
/// If the hbm is created by `hbdk-cc`, then it is the version of `hbdk-cc`
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
Hbrt4Status hbrt4HbmGetToolkitVersion(Hbrt4Hbm hbm, Hbrt4Version *version) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu march of hbm
///
/// \param_in_obj{hbm}
/// \param[out] march The bpu march
///
/// \on_err_out_set_to{HBRT4_BPU_MARCH_UNKNOWN}
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4HbmGetBpuMarch(Hbrt4Hbm hbm, Hbrt4BpuMarch *march) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu march name of hbm
///
/// \param_in_obj{hbm}
/// \param[out] bpuMarchName The bpu march name
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
Hbrt4Status hbrt4HbmGetBpuMarchName(Hbrt4Hbm hbm, const char **bpuMarchName) HBRT4_PRIV_CAPI_EXPORTED;

/// Get description associated with hbm,
/// the description category can be string and binary
///
/// \param_in_obj{hbm}
/// \param[out] description Description associated with the hbm file
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
/// - \ref HBRT4_STATUS_NOT_FOUND No description is associated with hbm
/// - \ret_ok
Hbrt4Status hbrt4HbmGetDescription(Hbrt4Hbm hbm, Hbrt4Description *description) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of graphs in hbm
///
/// \param_in_obj{hbm}
/// \param[out] num Number of graphs
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
Hbrt4Status hbrt4HbmGetNumGraphs(Hbrt4Hbm hbm, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get names of graph(s) in hbm, return multiple graph names in array
///
/// \param_in_obj{hbm}
/// \param[out] names The array of graph names
///
/// \on_err_out_empty_array
/// \lifetime_getter
///
/// \mt_safe
///
/// \note
/// The order of graph names is the same as \ref hbrt4HbmGetGraph
///
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4HbmGetGraphNames(Hbrt4Hbm hbm, Hbrt4CStringArrayRef *names) HBRT4_PRIV_CAPI_EXPORTED;

/// Get one graph in hbm by index
///
/// \param_in_obj{hbm}
/// \param_pos{hbrt4HbmGetNumGraphs}
/// \param[out] graph The graph object
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
/// - \ret_oob
/// - \ret_ok
Hbrt4Status hbrt4HbmGetGraph(Hbrt4Hbm hbm, size_t pos, Hbrt4Graph *graph) HBRT4_PRIV_CAPI_EXPORTED;

/// Get one graph in hbm by name
///
/// \param_in_obj{hbm}
/// \param[in] name The graph name
/// \param[out] graph The graph object
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
/// - \ref HBRT4_STATUS_NOT_FOUND Graph with specified `name` not found in this hbm
/// - \ret_ok
Hbrt4Status hbrt4HbmGetGraphByName(Hbrt4Hbm hbm, const char *name, Hbrt4Graph *graph) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the number of graph groups in hbm
///
/// \param_in_obj{hbm}
/// \param[out] num Number of graph groups
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
Hbrt4Status hbrt4HbmGetNumGraphGroups(Hbrt4Hbm hbm, size_t *num) HBRT4_PRIV_CAPI_EXPORTED;

/// Get names of graph group(s) in hbm, return multiple graph group names in array
///
/// \param_in_obj{hbm}
/// \param[out] names The array of graph group names
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \on_err_out_empty_array
/// \lifetime_getter
///
/// \mt_safe
///
/// \note
/// The order of graph names is the same as \ref hbrt4HbmGetGraphGroup
///
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4HbmGetGraphGroupNames(Hbrt4Hbm hbm, Hbrt4CStringArrayRef *names) HBRT4_PRIV_CAPI_EXPORTED;

/// Get one graph group in hbm by index
///
/// \param_in_obj{hbm}
/// \param_pos{hbrt4HbmGetNumGraphGroups}
/// \param[out] graphGroup The graph group object
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
Hbrt4Status hbrt4HbmGetGraphGroup(Hbrt4Hbm hbm, size_t pos, Hbrt4GraphGroup *graphGroup) HBRT4_PRIV_CAPI_EXPORTED;

/// Get one graph group in hbm by name
///
/// \param_in_obj{hbm}
/// \param[in] name The graph group name
/// \param[out] graphGroup The graph group object
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \exp_api
/// \do_not_test_in_j5_toolchain
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ref HBRT4_STATUS_NOT_FOUND Graph group with specified `name` not found in this hbm
/// - \ret_ok
Hbrt4Status hbrt4HbmGetGraphGroupByName(Hbrt4Hbm hbm, const char *name,
                                        Hbrt4GraphGroup *graphGroup) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
