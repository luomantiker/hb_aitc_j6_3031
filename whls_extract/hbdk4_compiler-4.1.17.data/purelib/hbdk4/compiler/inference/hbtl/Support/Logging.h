// HBTL TraceType and Trace Macros

#pragma once

#include "hbtl/Support/Compiler.h"
#include <type_traits> // IWYU pragma: keep
HBTL_PUSH_IGNORE_3RDPARTY_WARNING
#include "spdlog/spdlog.h" // IWYU pragma: export
// ostr.h must be included after spdlog.h
#include "spdlog/fmt/ostr.h" // IWYU pragma: export
HBTL_POP_IGNORE_3RDPARTY_WARNING

HBTL_NAMESPACE_BEGIN {

#define HBTL_TRACE(...)                                                                                                \
  if HBTL_CONSTEXPR_IF (std::decay_t<decltype(scope)>::traceEn) {                                                      \
    if (scope.verbose) {                                                                                               \
      spdlog::info(__VA_ARGS__);                                                                                       \
    }                                                                                                                  \
  }

#define HBTL_REMARK(...)                                                                                               \
  if HBTL_CONSTEXPR_IF (std::decay_t<decltype(scope)>::traceEn) {                                                      \
    spdlog::info(__VA_ARGS__);                                                                                         \
  }

#define createFailure(...) hbtl::LogicalResult::failure(true, fmt::format(__VA_ARGS__))

#define RETURN_ERROR_IF_NOT(cond, ...)                                                                                 \
  if (cond) {                                                                                                          \
  } else {                                                                                                             \
    return createFailure(__VA_ARGS__);                                                                                 \
  }

#define LOGICAL_IMPLY(antecedent, consequent) (!(antecedent) || (consequent))

// use in hbtl lib file and support variadic args
#define HBTL_TRAP_INTERNAL(file, line, ...)                                                                            \
  spdlog::critical(__VA_ARGS__);                                                                                       \
  spdlog::critical("trapped at {0}, line {1}", file, line);                                                            \
  __builtin_trap();

#define HBTL_TRAP(...) HBTL_TRAP_INTERNAL(__FILE__, __LINE__, __VA_ARGS__)

/// trapped with death message when runtime assertion fail
#define TRAP_IF_NOT(exp, ...)                                                                                          \
  if (!(exp)) {                                                                                                        \
    HBTL_TRAP(__VA_ARGS__);                                                                                            \
  }
}
HBTL_NAMESPACE_END
