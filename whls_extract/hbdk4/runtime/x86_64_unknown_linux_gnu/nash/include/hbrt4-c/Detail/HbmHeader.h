/// \file
/// \ref HbmHeader Module
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

/// \ingroup Hbm
///
/// \defgroup HbmHeader Hbm Header
/// Hbm header to represent some limited information of hbm,
/// which can be created without loading the entire hbm into the memory.
/// The information includes toolkit version and BPU march
///
/// \remark
/// Create \ref Hbrt4Instance with \ref HBRT4_BPU_MARCH_UNKNOWN
/// in \ref hbrt4InstanceBuilderCreate if bpu march is unknown
///
/// @{

/// Create a new hbm header by data without loading entire hbm content into memory
///
/// \param[in] instance Instance that related to hbm \n
/// Instance created with \ref HBRT4_BPU_MARCH_UNKNOWN is supported.
/// \param[in] data Data of hbm in memory
/// \param[out] hbmHeader Newly created hbm header
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmHeaderDestroy}
///
/// \mt_safe
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm or hbm header, \n
/// even if same \ref Hbrt4ArrayRef is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4HbmHeader object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4HbmHeader object
///
/// \remark
/// Create \ref Hbrt4Instance with \ref HBRT4_BPU_MARCH_UNKNOWN
/// in \ref hbrt4InstanceBuilderCreate if bpu march is unknown
///
/// \warning
/// Always pass in \ref Hbrt4ArrayRef with all of hbm data \n
/// Passing in \ref Hbrt4ArrayRef without full data, this API may succeed or fail, \n
/// there is no guarantee
///
/// \warning
/// This function does not loaded entire hbm into the memory.
/// Only read the minimum necessary data. Checking checksum of hbm
/// will not be done
///
/// \returns
/// - \ret_empty_data
/// - \ret_null_out
/// - \ref HBRT4_STATUS_BAD_DATA Invalid header in hbm
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderCreateByAddress(Hbrt4Instance instance, Hbrt4ArrayRef data, Hbrt4HbmHeader *hbmHeader)
    HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Create a new hbm header by data without loading entire hbm content into memory
///
/// \param[in] preInit Instance that without march and related to hbm \n
/// \param[in] data Data of hbm in memory
/// \param[out] hbmHeader Newly  created hbm header
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmHeaderDestroy}
///
/// \mt_safe
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm or hbm header, \n
/// even if same \ref Hbrt4ArrayRef is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4HbmHeader object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4HbmHeader object
/// This API differs from \ref hbrt4HbmHeaderCreateByAddress in that it introduces an instance 'preinit' \n
/// representing an unknown march
/// This API primarily validates the legality of preInit's values and checks whether a series of API \n
/// call sequences are correct.
///
/// \warning
/// Always pass in \ref Hbrt4ArrayRef with all of hbm data \n
/// Passing in \ref Hbrt4ArrayRef without full data, this API may succeed or fail, \n
/// there is no guarantee
///
/// \warning
/// This function does not loaded entire hbm into the memory.
/// Only read the minimum necessary data. Checking checksum of hbm
/// will not be done
///
/// \returns
/// - \ret_empty_data
/// - \ref HBRT4_STATUS_BAD_DATA Invalid header in hbm
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderCreateByAddress2(Hbrt4PreInit preInit, Hbrt4ArrayRef data, Hbrt4HbmHeader *hbmHeader)
    HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Create a new hbm header by hbm filename,
/// without loading the entire hbm into the memory
///
/// \param[in] instance Instance that related to hbm \n
/// Instance created with \ref HBRT4_BPU_MARCH_UNKNOWN is supported.
/// \param[in] filename Null-terminated filename of hbm
/// \param[out] hbmHeader Newly created hbm header
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmHeaderDestroy}
///
/// \mt_safe
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm or hbm header, \n
/// even if same filename is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4HbmHeader object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4HbmHeader object
///
/// \remark
/// Create \ref Hbrt4Instance with \ref HBRT4_BPU_MARCH_UNKNOWN
/// in \ref hbrt4InstanceBuilderCreate if bpu march is unknown
///
/// \warning
/// Always pass in `filename` with all of hbm data \n
/// Passing in `filename` without full data, this API may succeed or fail, \n
/// there is no guarantee
///
/// \warning
/// This function does not loaded entire hbm into the memory.
/// Only read the minimum necessary data. Checking checksum of hbm
/// will not be done
///
/// \returns
/// - \ret_null_filename
/// - \ret_null_out
/// - \ref HBRT4_STATUS_BAD_DATA Invalid header in hbm
/// - \ref HBRT4_STATUS_IO_ERROR File with `filename` does not exist or IO error occurs while reading file
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderCreateByFilename(Hbrt4Instance instance, const char *filename, Hbrt4HbmHeader *hbmHeader)
    HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Create a new hbm header by hbm filename,
/// without loading the entire hbm into the memory
///
/// \param[in] preInit  Instance that without march and related to hbm  \n
/// \param[in] filename Null-terminated filename of hbm
/// \param[out] hbmHeader Newly created hbm header
///
/// \on_err_out_null_obj
/// \lifetime_ctor{hbrt4HbmHeaderDestroy}
///
/// \mt_safe
///
/// \attention
/// This API **NEVER** reuse any data of already loaded hbm or hbm header, \n
/// even if same filename is loaded twice consecutively by this API \n
/// This API always different completely new \ref Hbrt4HbmHeader object in every calls to this API \n
/// This API **NEVER** returns previously returned and still alive \ref Hbrt4HbmHeader object
/// This API differs from \ref hbrt4HbmHeaderCreateByAddress in that it introduces an instance 'preinit' \n
/// representing an unknown march
/// This API primarily validates the legality of preInit's values and checks whether a series of API \n
/// call sequences are correct.
///
/// \warning
/// Always pass in `filename` with all of hbm data \n
/// Passing in `filename` without full data, this API may succeed or fail, \n
/// there is no guarantee
///
/// \warning
/// This function does not loaded entire hbm into the memory.
/// Only read the minimum necessary data. Checking checksum of hbm
/// will not be done
///
/// \returns
/// - \ret_empty_data
/// - \ref HBRT4_STATUS_BAD_DATA Invalid header in hbm
/// - \ref HBRT4_STATUS_IO_ERROR File with `filename` does not exist or IO error occurs while reading file
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderCreateByFilename2(Hbrt4PreInit preInit, const char *filename, Hbrt4HbmHeader *hbmHeader)
    HBRT4_PRIV_CAPI_EXPORTED HBRT4_PRIV_WARN_UNUSED_RESULT;

/// Destroy the hbm header object
///
/// \param_dtor{hbmHeader}
///
/// \lifetime_dtor
///
/// \returns
/// - \ret_bad_obj
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderDestroy(Hbrt4HbmHeader *hbmHeader) HBRT4_PRIV_CAPI_EXPORTED;

/// Get toolkit version of hbm header
///
/// \param_in_obj{hbmHeader}
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
Hbrt4Status hbrt4HbmHeaderGetToolkitVersion(Hbrt4HbmHeader hbmHeader, Hbrt4Version *version) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu march of hbm header
///
/// \param_in_obj{hbmHeader}
/// \param[out] march The bpu march
///
/// \on_err_out_set_to{HBRT4_BPU_MARCH_UNKNOWN}
/// \lifetime_getter
///
/// \mt_safe
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4HbmHeaderGetBpuMarch(Hbrt4HbmHeader hbmHeader, Hbrt4BpuMarch *march) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu march of hbm header
///
/// \param_in_obj{hbmHeader}
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
Hbrt4Status hbrt4HbmHeaderGetBpuMarchName(Hbrt4HbmHeader hbmHeader, const char **bpuMarchName) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the bpu buildid of hbm header
///
/// \param_in_obj{hbmHeader}
/// \param[out] cstring The buildid string
///
/// \on_err_out_null_obj
/// \lifetime_getter
///
/// \mt_safe
///
/// \brief  Extracts the BuildId of a hbm file.
///
/// \details BuildId is a checksum computed for the full content of each hbm file
/// (excluding the BuildId section in the header), as generated by the HBDK.
/// This function extracts the BuildId, which serves as the file's unique identifier,
/// ensuring that identical files share the same checksum and different files have
/// distinct checksums.
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
/// - \ref HBRT4_STATUS_NOT_FOUND Hbm does not have `build id` recorded
Hbrt4Status hbrt4HbmHeaderGetBuildId(Hbrt4HbmHeader hbmHeader, const char **cstring) HBRT4_PRIV_CAPI_EXPORTED;
/// @}

HBRT4_PRIV_C_EXTERN_C_END
