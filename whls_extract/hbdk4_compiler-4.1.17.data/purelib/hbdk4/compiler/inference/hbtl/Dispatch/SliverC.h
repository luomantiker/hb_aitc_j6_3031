#pragma once

#include "hbtl/Support/Compiler.h"
#include "ude/public/Library.h"

namespace hbtl {
HBTL_EXPORTED const ude::UdeLibrary *getSliverCHandle();
// SILVERC
extern const ude::UdeLibraryInit UDE_LIBRARY_static_init_SILVERC;
} // namespace hbtl
