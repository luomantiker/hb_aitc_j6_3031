/// \file
/// Private APIs of HBRT. User must **NOT** use them directly. May change at any time
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved
/// \header_detail_c

// IWYU pragma: private, include "hbrt4-c/hbrt4-c.h"

#pragma once

#if !defined(__clangd__) && !defined(HBRT4_DETAIL_GUARD)
#error "This file should not be directly included. Include hbrt4-c/hbrt4-c.h instead"
#endif

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

/// \if hidden
/// \defgroup Hbrt4Private Private APIs
/// \endif
///
/// @{

#ifdef __clang__
#define HBRT4_PRIV_C_STRICT_PROTOTYPES_BEGIN                                                                           \
  _Pragma("clang diagnostic push") _Pragma("clang diagnostic error \"-Wstrict-prototypes\"")
#define HBRT4_PRIV_C_STRICT_PROTOTYPES_END _Pragma("clang diagnostic pop")
#else
/// To disable -Wstrict-prototypes
/// \private_api
#define HBRT4_PRIV_C_STRICT_PROTOTYPES_BEGIN
/// To disable -Wstrict-prototypes
/// \private_api
#define HBRT4_PRIV_C_STRICT_PROTOTYPES_END
#endif

#ifdef __cplusplus
#define HBRT4_PRIV_C_EXTERN_C_BEGIN                                                                                    \
  extern "C" {                                                                                                         \
  HBRT4_PRIV_C_STRICT_PROTOTYPES_BEGIN
#define HBRT4_PRIV_C_EXTERN_C_END                                                                                      \
  HBRT4_PRIV_C_STRICT_PROTOTYPES_END                                                                                   \
  }
#else
/// For extern "C", depending on whether current language is C or C++
/// \private_api
#define HBRT4_PRIV_C_EXTERN_C_BEGIN HBRT4_PRIV_C_STRICT_PROTOTYPES_BEGIN
/// For extern "C", depending on whether current language is C or C++
/// \private_api
#define HBRT4_PRIV_C_EXTERN_C_END HBRT4_PRIV_C_STRICT_PROTOTYPES_END
#endif

#ifdef HBRT4_PRIV_DOXYGEN
/// To make the function have public visibility
/// \private_api
#define HBRT4_PRIV_CAPI_EXPORTED
#else
/// To make the function have public visibility
/// \private_api
#define HBRT4_PRIV_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
/// Macro to typedef enum
/// \private_api
#define HBRT4_PRIV_TYPEDEF_ENUM(enumName)                                                                              \
  /** \cond hidden */                                                                                                  \
  using enumName = enum enumName /** \endcond */
#else
/// Macro to typedef enum
/// \private_api
#define HBRT4_PRIV_TYPEDEF_ENUM(enumName)                                                                              \
  /** \cond hidden */                                                                                                  \
  typedef enum enumName enumName /** \endcond */
#endif

#ifdef __cplusplus
/// Macro to typedef struct
/// \private_api
#define HBRT4_PRIV_TYPEDEF_STRUCT(structName)                                                                          \
  /** \cond hidden */                                                                                                  \
  using structName = struct structName /** \endcond */
#else
/// Macro to typedef struct
/// \private_api
#define HBRT4_PRIV_TYPEDEF_STRUCT(structName)                                                                          \
  /** \cond hidden */                                                                                                  \
  typedef struct structName structName /** \endcond */
#endif

/// Warn if the returned status of the API not checked
/// Add this to API, which not checking the returned status must be bug
#define HBRT4_PRIV_WARN_UNUSED_RESULT __attribute__((warn_unused_result))

/// Warn something has been deprecated
#if defined(__clang__)
#define HBRT4_DEPRECATED(MSG, FIX) __attribute__((deprecated(MSG, FIX)))
#else
#define HBRT4_DEPRECATED(MSG, FIX) __attribute__((deprecated(MSG)))
#endif

// NOTE(hehaoqian)
// When add new macro, undefine it in \file CompilerUndefineMacros.h

/// @}
