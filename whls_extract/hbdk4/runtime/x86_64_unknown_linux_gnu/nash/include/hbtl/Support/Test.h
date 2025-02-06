#pragma once

#include "hbtl/ADT/Random.h" // IWYU pragma: export
#include "hbtl/Tensor.h"

#ifdef HBDK_WHOLE_BUILD
#include "hbut/Testing/Support/gtest.h" // IWYU pragma: export
#endif                                  // HBDK_WHOLE_BUILD

#include "hbtl/Support/Compiler.h"
HBTL_PUSH_IGNORE_3RDPARTY_WARNING
#include "gtest/gtest.h" // IWYU pragma: export
HBTL_POP_IGNORE_3RDPARTY_WARNING

HBTL_NAMESPACE_BEGIN {

  template <typename T> inline void breakOnFirstMismatchedCoord(const hbtl::Tensor &lhs, const hbtl::Tensor &rhs) {
    EXPECT_EQ(lhs.getSizesCopy(), rhs.getSizesCopy()) << "shape mismatch";
    EXPECT_EQ(lhs.getType(), rhs.getType()) << "type mismatch";
    EXPECT_EQ(lhs.getType(), getElementType<T>()) << "type is not as expected " << getElementType<T>();

    for (const auto &e : enumerate(zip(lhs.getData<T>(), rhs.getData<T>()))) {
      // auto [a, b] = e.value();
      auto a = std::get<0>(e.value());
      auto b = std::get<1>(e.value());
      auto coord = lhs.getCoord(static_cast<int64_t>(e.index()));
      EXPECT_EQ(a, b) << " first mismatched coord " << hbtl::formatCoord(coord);
      if (a != b) {
        break;
      }
    }
  }

  template <typename T> std::string toString(T obj) {
    std::stringstream ss;
    ss << obj;
    return ss.str();
  }

#define CHECK_STATUS(s) ASSERT_TRUE(succeeded(s)) << s.getInfo();
}
HBTL_NAMESPACE_END
