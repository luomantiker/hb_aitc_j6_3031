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

#define UDE_CPP14_OR_GREATER (__cplusplus >= 201402L)
#define UDE_CPP17_OR_GREATER (__cplusplus >= 201703L)

#if !UDE_CPP14_OR_GREATER
#error "UDE requires at least C++14. C++11 is unsupported"
#endif

// set kernel define visible in shared library
#define UDE_EXPORTED __attribute__((visibility("default")))

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define UDE_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define UDE_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define UDE_LIKELY(EXPR) (EXPR)
#define UDE_UNLIKELY(EXPR) (EXPR)
#endif

#if UDE_CPP17_OR_GREATER
#define UDE_NODISCARD [[nodiscard]]
#define UDE_MAYBE_UNUSED [[maybe_unused]]
#define UDE_CONSTEXPR_IF constexpr
#else
#define UDE_NODISCARD
#define UDE_MAYBE_UNUSED
#define UDE_CONSTEXPR_IF
#endif
