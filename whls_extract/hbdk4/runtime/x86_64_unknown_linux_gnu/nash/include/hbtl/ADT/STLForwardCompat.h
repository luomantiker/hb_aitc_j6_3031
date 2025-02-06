//===- STLForwardCompat.h - Library features from future STLs ------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains library features backported from future STL versions.
///
/// These should be replaced with their STL counterparts as the C++ version LLVM
/// is compiled with is updated.
///
//===----------------------------------------------------------------------===//

/// forked from llvm/ADT/STLForwardCompat.h
#pragma once

#include "hbtl/ADT/Optional.h"
#include <type_traits>

HBTL_NAMESPACE_BEGIN {

  //===----------------------------------------------------------------------===//
  //     Features from C++20
  //===----------------------------------------------------------------------===//

  template <typename T>
  struct remove_cvref // NOLINT(readability-identifier-naming)
  {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  template <typename T>
  using remove_cvref_t // NOLINT(readability-identifier-naming)
      = typename hbtl::remove_cvref<T>::type;

  //===----------------------------------------------------------------------===//
  //     Features from C++23
  //===----------------------------------------------------------------------===//

  // TODO: Remove this in favor of std::optional<T>::transform once we switch to
  // C++23.
  template <typename T, typename Function>
  auto transformOptional(const hbtl::optional<T> &O, const Function &F)->hbtl::optional<decltype(F(*O))> {
    if (O)
      return F(*O);
    return hbtl::nullopt;
  }

  // TODO: Remove this in favor of std::optional<T>::transform once we switch to
  // C++23.
  template <typename T, typename Function>
  auto transformOptional(hbtl::optional<T> && O, const Function &F)->hbtl::optional<decltype(F(*std::move(O)))> {
    if (O)
      return F(*std::move(O));
    return hbtl::nullopt;
  }

  // TODO: Remove this when use gcc7 or support c++17
  template <typename F, typename... Args>
  struct __is_invocable : std::is_constructible<std::function<void(Args...)>,
                                                std::reference_wrapper<typename std::remove_reference<F>::type>> {};

  template <typename R, typename F, typename... Args>
  struct __is_invocable_r : std::is_constructible<std::function<R(Args...)>,
                                                  std::reference_wrapper<typename std::remove_reference<F>::type>> {};
}
HBTL_NAMESPACE_END
