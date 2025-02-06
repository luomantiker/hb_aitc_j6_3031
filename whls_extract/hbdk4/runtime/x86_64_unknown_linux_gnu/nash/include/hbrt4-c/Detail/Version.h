/// \file
/// \ref Version Module
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#include "hbrt4-c/Detail/Compiler.h"
#include "hbrt4-c/Detail/Object.h"
#include "hbrt4-c/Detail/Status.h"

HBRT4_PRIV_C_EXTERN_C_BEGIN

/// \defgroup Version Version
/// Get version information of HBRT, Hbm and Graph
///
/// # List of Features
/// Version information including:
/// - Major, minor and patch number
/// - Git commit hash
/// - CString of version
///
/// # Task Sequence
/// The version information can be retrieved in the following sequence:
/// - Get the \ref Hbrt4Version object in one of the following ways:
/// 1. hbrt4 version by \ref hbrt4GetToolkitVersion,
/// 2. \ref Hbrt4Graph toolkit version by \ref hbrt4GraphGetToolkitVersion,
/// 3. \ref Hbrt4Hbm toolkit version by \ref hbrt4HbmGetToolkitVersion,
/// - Get the version information from \ref Hbrt4Version by
/// \ref hbrt4VersionGetSemantic,
/// \ref hbrt4VersionGetCString, and
/// \ref hbrt4VersionGetCommitHash
/// - Two versions can be compared using \ref hbrt4VersionCompare
///
/// @{

/// Hbrt4 version semantic information
/// \since v4.0.1
///
/// # Default Value {#Hbrt4VersionSemantic_DefaultValue}
/// - All integer fields are set to 0
/// - All `const char*` fields are set to empty string
struct Hbrt4VersionSemantic {

  /// Major version
  /// \since v4.0.1
  ///
  /// For \ref hbrt4GetToolkitVersion, this must be equal to 4
  uint32_t major;

  /// Minor version
  /// \since v4.0.1
  ///
  /// \invariant
  /// For \ref hbrt4GetToolkitVersion, this equal to the minor version
  /// of the corresponding HBDK compiler version
  uint32_t minor;

  /// Patch version
  /// \since v4.0.1
  ///
  /// \invariant
  /// For \ref hbrt4GetToolkitVersion, this equal to the minor version
  /// of the corresponding HBDK compiler version
  uint32_t patch;

  /// Reserved for future use
  /// \since v4.0.1
  uint32_t reserved;

  /// Pre-release version, such as "a0", "rc1"
  /// \since v4.0.1
  ///
  /// \invariant
  /// This is empty string if this is not a pre-release version
  ///
  /// \invar_non_null_cstring
  const char *pre_release;

  /// Other version string, which is not represented by stuffs above.
  /// \since v4.0.1
  ///
  /// Such as the commit hash in a daily build
  ///
  /// \invar_non_null_cstring
  const char *extra;
};
/// \cond hidden
HBRT4_PRIV_TYPEDEF_STRUCT(Hbrt4VersionSemantic);
/// \endcond

/// Get the toolkit version of HBRT, which is same as HBDK version
/// \since v4.0.1
///
/// \param[out] version The toolkit version
/// \lifetime_static
///
/// \note
/// In \ref hbrt4, the compiler version and the HBRT version are the same \n
/// which is called `Toolkit Version` here \n
/// It is unlikely to separate two versions in the future
///
/// #### Examples
/// \code
/// Hbrt4Version version;
/// hbrt4GetToolkitVersion(&version);
/// \endcode
///
/// \returns
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4GetToolkitVersion(Hbrt4Version *version) HBRT4_PRIV_CAPI_EXPORTED;

/// Get the version semantic information, such as major/minor/major number
/// \since v4.0.1
///
/// \param_in_obj{version}
/// \param[out] semantic The version semantic
/// \on_err_out_set_to{Hbrt4VersionSemantic_DefaultValue}
/// \lifetime_getter
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VersionGetSemantic(Hbrt4Version version, Hbrt4VersionSemantic *semantic) HBRT4_PRIV_CAPI_EXPORTED;

/// Get version in string form
/// \since v4.0.1
///
/// \param_in_obj{version}
/// \param[out] cstring Version string
/// May contain additional information not shown in \ref hbrt4VersionGetSemantic
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// #### Examples
/// \code
/// // Code to acquire `version` omitted
/// const char* version_string;
/// hbrt4VersionGetCString(version, &version_string);
/// \endcode
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VersionGetCString(Hbrt4Version version, const char **cstring) HBRT4_PRIV_CAPI_EXPORTED;

/// Get git commit hash string of version
/// \since v4.0.1
///
/// \param_in_obj{version}
/// \param[out] commitHash HBRT git commit hash
/// \on_err_out_empty_str
/// \lifetime_getter
///
/// \invariant
/// Output should be null terminated 40 character long SHA1 hash,
/// consist of digit `0-9` and lowercase letters `a-f` only
///
/// #### Examples
/// \code
/// // Code to acquire `version` omitted
/// const char* hash;
/// hbrt4VersionGetCommitHash(version, &hash);
/// \endcode
///
/// \lifetime_getter
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VersionGetCommitHash(Hbrt4Version version, const char **commitHash) HBRT4_PRIV_CAPI_EXPORTED;

/// Compare two versions to check if they are same
/// \since v4.0.1
///
/// \param_in_obj{lhs}
/// \param_in_obj{rhs}
/// \param[out] result The comparison result \n
/// Set to `-1` if `lhs` is older than `rhs` \n
/// Set to `1` if `lhs` is newer than `rhs` \n
/// Otherwise set to `0`
///
/// \on_err_out_zero
///
/// \note
/// This API does NOT care how the input versions are got, \n
/// and whether they come from the same API \n
///
/// For example, comparing versions between the output of
/// \ref hbrt4GetToolkitVersion and \ref hbrt4GraphGetToolkitVersion is allowed
///
/// \returns
/// - \ret_null_in_obj
/// - \ret_bad_obj
/// - \ret_null_out
/// - \ret_ok
Hbrt4Status hbrt4VersionCompare(Hbrt4Version lhs, Hbrt4Version rhs, int32_t *result) HBRT4_PRIV_CAPI_EXPORTED;

/// @}

HBRT4_PRIV_C_EXTERN_C_END
