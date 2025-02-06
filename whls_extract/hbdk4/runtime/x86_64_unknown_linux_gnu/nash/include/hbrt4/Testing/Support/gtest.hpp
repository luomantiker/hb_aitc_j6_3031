/// \file
/// Google Test Helper for HBRT4
#pragma once

#include "hbrt4-c/hbrt4-c.h"
#include "hbut/Testing/Support/gtest.h" // IWYU pragma: export

#define HBRT4_ASSERT_OK(x) ASSERT_EQ((x), HBRT4_STATUS_OK)
#define HBRT4_ASSERT_NULL_OBJ(x) ASSERT_EQ((x), HBRT4_STATUS_NULL_OBJECT)

#define HBRT4_ASSERT_IS_NULL_OBJ(x) ASSERT_EQ((x.priv_object.priv_opaque), 0)
#define HBRT4_ASSERT_EMPTY_ARR(x)                                                                                      \
  ASSERT_EQ((x.data), nullptr);                                                                                        \
  ASSERT_EQ((x.len), 0)
