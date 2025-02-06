// Get Type prototype, for debugging purpose

#pragma once

#ifndef HBTL_SUPPORT_TYPEPROTOTYPE_H_
#define HBTL_SUPPORT_TYPEPROTOTYPE_H_

#include "hbtl/ADT/string_view.h"
#include "hbtl/Support/Compiler.h"
#include <array>
#include <iostream>
#include <type_traits>

HBTL_NAMESPACE_BEGIN {

  namespace detail {

  // Split const string into two string
  constexpr auto splitToTwoString(std::string_view str, std::string_view delim) {
    std::array<std::string_view, 2> result{};
    for (size_t i = 0; i < str.size(); i++) {
      if (str.substr(i, delim.size()) == delim) {
        result[0] = str.substr(0, i);
        result[1] = str.substr(i + delim.size(), str.size() - (i + delim.size()));
      }
    }

    return result;
  }

  /// Get type prototype as constexpr string
  /// but this function specially handles the void type
  template <typename T> constexpr std::string_view getTypePrototypeForDbgImpl() {
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
#endif
    if constexpr (std::is_same_v<T, void>) {
      return name;
    }
    // Implementation detail:
    // for `getTypePrototypeImpl<int>`
    // __PRETTY_FUNCTION__ may be similar to
    // constexpr std::string_view hbtl::detail::getTypePrototypeImpl() [with T = int; std::string_view =
    // std::basic_string_view<char>] We are removing the parts before and after `int`
    prefix = splitToTwoString(getTypePrototypeForDbgImpl<void>(), "void")[0];
    suffix = splitToTwoString(getTypePrototypeForDbgImpl<void>(), "void")[1];
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
  }
  } // namespace detail

  /// Get type prototype as constexpr string
  /// The returned string may be different depending on compiler type or version
  /// Should only use for debugging purpose
  /// Example usage and output:
  /// ```
  /// getTypePrototypeForDbg<void>  -> "void"
  /// getTypePrototypeForDbg<int> -> "int"
  /// void foo(int, int);
  /// getTypePrototypeForDbg<decltype(foo)> -> "void (*)(int, int)"
  /// ```
  template <typename T> constexpr std::string_view getTypePrototypeForDbg() {
    // std::decay_t is needed to make the output of
    // `getTypePrototypeForDbg<decltype(foo)>()` and `getTypePrototypeForDbg(foo)` are the same
    return detail::getTypePrototypeForDbgImpl<std::decay_t<T>>();
  }

  template <> constexpr std::string_view getTypePrototypeForDbg<void>() { return "void"; }

  template <typename T> constexpr std::string_view getTypePrototypeForDbg(T t) { return getTypePrototypeForDbg<T>(); }
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_TYPEPROTOTYPE_H_
