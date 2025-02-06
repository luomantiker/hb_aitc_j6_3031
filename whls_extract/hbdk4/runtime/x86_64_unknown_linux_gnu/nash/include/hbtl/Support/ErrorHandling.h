#pragma once
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/Context.h"
#include "hbtl/Support/LogicalResult.h"

// use in hbtl header file and only support single string info
#define hbtl_trap_internal(file, line, info)                                                                           \
  Context::get()->critical(("trapped at " + std::string(file) + ", line " + std::to_string(line)).c_str());            \
  Context::get()->critical(info);                                                                                      \
  __builtin_trap();

#define hbtl_trap(info) hbtl_trap_internal(__FILE__, __LINE__, info)

/// trapped with death message when runtime assertion fail
#define trap_if_not(exp, info)                                                                                         \
  if (!(exp)) {                                                                                                        \
    hbtl_trap(info);                                                                                                   \
  }
