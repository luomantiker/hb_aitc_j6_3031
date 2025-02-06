// HBTL macros that forces compiler generates codes for user-specified values

#pragma once
#ifndef HBTL_SUPPORT_VALUE_H_
#define HBTL_SUPPORT_VALUE_H_

#include "hbtl/Support/ErrorHandling.h"

HBTL_NAMESPACE_BEGIN {

  template <typename ValueType, ValueType val> struct ValueWrapper {
    static constexpr ValueType value = val;
  };

#if HBTL_CPP17_OR_GREATER
  template <auto V0, typename Func> bool tryRunByValue(decltype(V0) value, const Func &func) {
    if (value == V0) {
      (void)func(ValueWrapper<decltype(V0), V0>{});
      return true;
    } else {
      return false;
    }
  }

  template <auto V0, typename Func> void runByValue(decltype(V0) value, const Func &func) {
    bool result = tryRunByValue<V0>(value, func);
    if (!result) {
      hbtl_trap("Unexpected value");
    }
  }
#endif

  template <typename T, T V0, typename Func> bool tryRunByValue2(T value, const Func &func) {
    if (value == V0) {
      (void)func(ValueWrapper<T, V0>{});
      return true;
    } else {
      return false;
    }
  }

  /// `runByValue2` is similar to `runByValue`, but can be used in C++14 mode
  /// The type of constants must be declared in template function call
  template <typename T, T V0, typename Func> void runByValue2(T value, const Func &func) {
    bool result = tryRunByValue2<T, V0>(value, func);
    if (!result) {
      hbtl_trap("Unexpected value");
    }
  }

#if HBTL_CPP17_OR_GREATER

#define TRY_RUN_BY_VALUE_MACRO(...)                                                                                    \
  bool tryRunByValue(decltype(V0) value, const Func &func) {                                                           \
    if (tryRunByValue<V0>(value, func)) {                                                                              \
      return true;                                                                                                     \
    } else {                                                                                                           \
      return tryRunByValue<__VA_ARGS__>(value, func);                                                                  \
    }                                                                                                                  \
  }

#define RUN_BY_VALUE_MACRO(...)                                                                                        \
  void runByValue(decltype(V0) value, const Func &func) {                                                              \
    bool result = tryRunByValue<V0, __VA_ARGS__>(value, func);                                                         \
    if (!result) {                                                                                                     \
      hbtl_trap("Unexpected value");                                                                                   \
    }                                                                                                                  \
  }

#endif

#define TRY_RUN_BY_VALUE_MACRO2(...)                                                                                   \
  bool tryRunByValue2(T value, const Func &func) {                                                                     \
    if (tryRunByValue2<T, V0>(value, func)) {                                                                          \
      return true;                                                                                                     \
    } else {                                                                                                           \
      return tryRunByValue2<T, __VA_ARGS__>(value, func);                                                              \
    }                                                                                                                  \
  }

#define RUN_BY_VALUE_MACRO2(...)                                                                                       \
  void runByValue2(T value, const Func &func) {                                                                        \
    bool result = tryRunByValue2<T, V0, __VA_ARGS__>(value, func);                                                     \
    if (!result) {                                                                                                     \
      hbtl_trap("Unexpected value");                                                                                   \
    }                                                                                                                  \
  }

#if HBTL_CPP17_OR_GREATER

  template <auto V0, auto V1, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1)
  template <auto V0, auto V1, typename Func>
  RUN_BY_VALUE_MACRO(V1)

  template <auto V0, auto V1, auto V2, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2)
  template <auto V0, auto V1, auto V2, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2)

  template <auto V0, auto V1, auto V2, auto V3, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3)
  template <auto V0, auto V1, auto V2, auto V3, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, auto V14, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, auto V14, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14)

  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, auto V14, auto V15, typename Func>
  TRY_RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15)
  template <auto V0, auto V1, auto V2, auto V3, auto V4, auto V5, auto V6, auto V7, auto V8, auto V9, auto V10,
            auto V11, auto V12, auto V13, auto V14, auto V15, typename Func>
  RUN_BY_VALUE_MACRO(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15)

#undef TRY_RUN_BY_VALUE_MACRO
#undef RUN_BY_VALUE_MACRO

#endif

  template <typename T, T V0, T V1, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1)
  template <typename T, T V0, T V1, typename Func>
  RUN_BY_VALUE_MACRO2(V1)

  template <typename T, T V0, T V1, T V2, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2)
  template <typename T, T V0, T V1, T V2, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2)

  template <typename T, T V0, T V1, T V2, T V3, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3)
  template <typename T, T V0, T V1, T V2, T V3, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3)

  template <typename T, T V0, T V1, T V2, T V3, T V4, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4)
  template <typename T, T V0, T V1, T V2, T V3, T V4, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13,
            typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13,
            typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13, T V14,
            typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13, T V14,
            typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14)

  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13, T V14,
            T V15, typename Func>
  TRY_RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15)
  template <typename T, T V0, T V1, T V2, T V3, T V4, T V5, T V6, T V7, T V8, T V9, T V10, T V11, T V12, T V13, T V14,
            T V15, typename Func>
  RUN_BY_VALUE_MACRO2(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15)

#undef TRY_RUN_BY_VALUE_MACRO2
#undef RUN_BY_VALUE_MACRO2
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_VALUE_H_
