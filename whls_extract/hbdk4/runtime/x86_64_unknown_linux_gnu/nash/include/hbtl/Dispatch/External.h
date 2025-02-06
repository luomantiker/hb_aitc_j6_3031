#pragma once

#include "hbtl/Support/Compiler.h"
#include "ude/public/Library.h"

namespace hbtl {
HBTL_EXPORTED const ude::UdeLibrary *getExternalHandle();
// External
extern const ude::UdeLibraryInit UDE_LIBRARY_static_init_External;
} // namespace hbtl
