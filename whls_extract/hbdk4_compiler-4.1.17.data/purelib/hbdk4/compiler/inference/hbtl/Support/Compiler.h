#pragma once

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#ifndef __has_extension
#define __has_extension(x) 0
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#define HBTL_CPP14_OR_GREATER (__cplusplus >= 201402L)
#define HBTL_CPP17_OR_GREATER (__cplusplus >= 201703L)

#if !HBTL_CPP14_OR_GREATER
#error "HBTL requires at least C++14 for compiling header, C++17 for compiling entire code base. C++11 is unsupported"
#endif

// set kernel define visible in shared library
#define HBTL_EXPORTED __attribute__((visibility("default")))

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define HBTL_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define HBTL_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define HBTL_LIKELY(EXPR) (EXPR)
#define HBTL_UNLIKELY(EXPR) (EXPR)
#endif

#ifdef __cplusplus
#define HBTL_ASSUME_USED(x) static_cast<void>(x)
#else
#define HBTL_ASSUME_USED(x) (void)(x)
#endif

/// \macro HBTL_GNUC_PREREQ
/// Extend the default __GNUC_PREREQ even if glibc's features.h isn't
/// available.
#ifndef HBTL_GNUC_PREREQ
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define HBTL_GNUC_PREREQ(maj, min, patch)                                                                              \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >= ((maj) << 20) + ((min) << 10) + (patch))
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define HBTL_GNUC_PREREQ(maj, min, patch) ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) >= ((maj) << 20) + ((min) << 10))
#else
#define HBTL_GNUC_PREREQ(maj, min, patch) 0
#endif
#endif

// clang-format off

/// Use to ignore warnings in the 3rdparty header file
/// Should be paired with HBTL_POP_IGNORE_3RDPARTY_WARNING,
/// and put the code or #include between them
/// Example:
/// #include "hbtl/Support/Compiler.h"
/// HBTL_PUSH_IGNORE_3RDPARTY_WARNING
/// #include "llvm/xxxxx.h"
/// HBTL_POP_IGNORE_3RDPARTY_WARNING
#ifdef __clang__
#define HBTL_PUSH_IGNORE_3RDPARTY_WARNING \
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic ignored \"-Wcast-qual\"") \
_Pragma("GCC diagnostic ignored \"-Wconversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-equal\"") \
_Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-compare\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wstrict-overflow\"") \
_Pragma("GCC diagnostic ignored \"-Wsuggest-override\"") \
_Pragma("GCC diagnostic ignored \"-Wzero-as-null-pointer-constant\"") \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"") \
_Pragma("GCC diagnostic ignored \"-Wunused-parameter\"") \
_Pragma("GCC diagnostic ignored \"-Wcovered-switch-default\"") \
_Pragma("GCC diagnostic ignored \"-Wself-assign\"")\
_Pragma("GCC diagnostic ignored \"-Wsometimes-uninitialized\"")\
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic pop")
// Useless _Pragma at the end,
// so developer less likely to forget to add backslash ("\") at the end
// when ignoring more warnings
#elif __GNUC__ >= 7
#define HBTL_PUSH_IGNORE_3RDPARTY_WARNING \
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic ignored \"-Wunused-but-set-variable\"") \
_Pragma("GCC diagnostic ignored \"-Wcast-qual\"") \
_Pragma("GCC diagnostic ignored \"-Wconversion\"") \
_Pragma("GCC diagnostic ignored \"-Wduplicated-branches\"")
_Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-equal\"") \
_Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-compare\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wstrict-overflow\"") \
_Pragma("GCC diagnostic ignored \"-Wsuggest-override\"") \
_Pragma("GCC diagnostic ignored \"-Wzero-as-null-pointer-constant\"") \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"") \
_Pragma("GCC diagnostic ignored \"-Wnoexcept\"") \
_Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"") \
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic pop")
// Useless _Pragma at the end,
// so developer less likely to forget to add backslash ("\") at the end
// when ignoring more warnings
#elif __GNUC__ >= 5
#define HBTL_PUSH_IGNORE_3RDPARTY_WARNING \
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic ignored \"-Wunused-but-set-variable\"") \
_Pragma("GCC diagnostic ignored \"-Wcast-qual\"") \
_Pragma("GCC diagnostic ignored \"-Wconversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wfloat-equal\"") \
_Pragma("GCC diagnostic ignored \"-Wold-style-cast\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-compare\"") \
_Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
_Pragma("GCC diagnostic ignored \"-Wstrict-overflow\"") \
_Pragma("GCC diagnostic ignored \"-Wzero-as-null-pointer-constant\"") \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"") \
_Pragma("GCC diagnostic ignored \"-Wnoexcept\"") \
_Pragma("GCC diagnostic ignored \"-Wself-assign\"")\
_Pragma("GCC diagnostic ignored \"-Wsometimes-uninitialized\"")\
_Pragma("GCC diagnostic push") \
_Pragma("GCC diagnostic pop")
// Useless _Pragma at the end,
// so developer less likely to forget to add backslash ("\") at the end
// when ignoring more warnings
#endif

#define HBTL_POP_IGNORE_3RDPARTY_WARNING _Pragma("GCC diagnostic pop")
// clang-format on

#if HBTL_CPP17_OR_GREATER
#define HBTL_NODISCARD [[nodiscard]]
#define HBTL_MAYBE_UNUSED [[maybe_unused]]
#define HBTL_CONSTEXPR_IF constexpr
#else
#define HBTL_NODISCARD
#define HBTL_MAYBE_UNUSED
#define HBTL_CONSTEXPR_IF
#endif

#if defined(__clang__) || __GNUC__ >= 11
#define HBTL_CONSTEXPR_IF_CONST_VAR HBTL_CONSTEXPR_IF
#else
// `constexpr if` in gcc9 is buggy,
// when used for equality of constant defined in lambda function
// `constexpr if` type equality check is still ok
#define HBTL_CONSTEXPR_IF_CONST_VAR
#endif

#if __GNUC__ == 11 && __GNUC_MINOR__ == 4
/// Some `constexpr if` related internal compiler error only in gcc11.4
/// not all `constexpr if` are affected
/// Use this macro to suppress `constexpr if` only in gcc11.4
#define HBTL_CONSTEXPR_IF_NO_GCC_114
#else
#define HBTL_CONSTEXPR_IF_NO_GCC_114 HBTL_CONSTEXPR_IF
#endif

#define HBTL_CONCAT_TOKEN(a, b) a##b
#define HBTL_CONCAT_3TOKEN(a, b, c) a##b##c

// NOTE: Inline namespace does not affect API, but it affects symbol mangling
// Used internally inside HBDK project to mangle HBTL symbols which is bundled inside HBRT
#ifdef HBTL_USE_HBRT_INLINE_NAMESPACE
#define HBTL_NAMESPACE_BEGIN                                                                                           \
  namespace hbtl {                                                                                                     \
  inline namespace hbrt4_v0
#else
#define HBTL_NAMESPACE_BEGIN namespace hbtl
#endif

#ifdef HBTL_USE_HBRT_INLINE_NAMESPACE
#define HBTL_NAMESPACE_END }
#else
#define HBTL_NAMESPACE_END
#endif
