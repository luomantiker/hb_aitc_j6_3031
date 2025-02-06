/// forked from llvm/ADT/identity.h
#pragma once

#include "hbtl/Support/Compiler.h"

HBTL_NAMESPACE_BEGIN {

  // Similar to `std::identity` from C++20.
  template <class Ty> struct identity {
    using is_transparent = void;
    using argument_type = Ty;

    Ty &operator()(Ty &self) const { return self; }
    const Ty &operator()(const Ty &self) const { return self; }
  };
}
HBTL_NAMESPACE_END
