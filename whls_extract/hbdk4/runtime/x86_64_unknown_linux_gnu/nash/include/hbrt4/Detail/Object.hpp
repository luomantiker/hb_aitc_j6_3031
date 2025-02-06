// IWYU pragma: private, include "hbrt4/hbrt4.hpp"

#pragma once

#include "hbrt4-c/Detail/Object.h"

namespace hbrt4 {

// C++ helper to make it possible to put Hbrt4 objects inside `std::map`
template <typename Hbrt4ObjectType> struct CmpObjectByPtr {
  bool operator()(Hbrt4ObjectType a, Hbrt4ObjectType b) const { return hbrt4ObjectPtrLess(a, b); }
};
} // namespace hbrt4
