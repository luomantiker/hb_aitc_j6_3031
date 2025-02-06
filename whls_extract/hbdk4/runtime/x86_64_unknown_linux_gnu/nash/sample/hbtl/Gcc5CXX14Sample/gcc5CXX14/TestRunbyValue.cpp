#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/Batch.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/Logging.h"
#include "hbtl/Support/LogicalResult.h"
#include "hbtl/Support/Test.h"
#include "hbtl/Support/Value.h"
#include "hbtl/Support/Verify.h"
#include "gtest/gtest.h"
#include <cstdint>

HBTL_NAMESPACE_BEGIN {

  LogicalResult Func_leakyRelu(Tensor & fout, const Tensor fin, float slope) {
    RETURN_ERROR_IF_NOT(unifyShape(fout, fin), "shape mismatch")
    RETURN_ERROR_IF_NOT(unifyElementType(fout, fin), "type mismatch")

    runByValue2<bool, true, false>(slope == 0, [&](const auto f) {
      constexpr bool isRelu = decltype(f)::value;
      runByElementType<int8_t, int16_t, int32_t, float>(fout.getType(), [&](const auto tv) {
        using T = typename decltype(tv)::type;
        runWithTrace([&](auto scope) {
          constexpr bool traceEn = decltype(scope)::traceEn;
          runByVariadicBatchOMP<traceEn>(-1, fout, [&](auto batch) {
            auto to = fout.select(batch).contiguous();
            auto vo = to.template getMutData<T>();

            auto ti = fin.select(batch).contiguous();
            auto vi = ti.template getData<T>();

            for (auto c = 0U; c < vo.size(); ++c) {
              scope.update(batch, c);
              if (isRelu) {
                vo[c] = std::max(static_cast<float>(0.0), static_cast<float>(vi[c]));
              } else {
                vo[c] = std::max(static_cast<float>(slope * vi[c]), static_cast<float>(vi[c]));
              }
              HBTL_TRACE("fout{} = fin{}*{}", vo[c], vi[c], slope);
            }
            fout.select(batch).copy(to);
          }); // end runByVariadicBatchOMP func
        });   // end runWithTrace func
      });     // end runByElementType func
    });       // end runByValue func
    return LogicalResult::success();
  }

  enum class Foo {
    num0,
    num1,
    num2,
    num3,
    num4,
    num5,
    num6,
    num7,
    num8,
    num9,
    num10,
    num11,
    num12,
    num13,
    num14,
    num15
  };

  static int testRunByValue2(Foo foo) {
    int value = -1;
    runByValue2<Foo, Foo::num0, Foo::num1, Foo::num2, Foo::num3, Foo::num4, Foo::num5, Foo::num6, Foo::num7, Foo::num8,
                Foo::num9, Foo::num10, Foo::num11, Foo::num12, Foo::num13, Foo::num14, Foo::num15>(
        foo, [&](const auto f) {
          switch (foo) {
          case Foo::num0:
            value = 0;
            break;
          case Foo::num1:
            value = 1;
            break;
          case Foo::num2:
            value = 2;
            break;
          case Foo::num3:
            value = 3;
            break;
          case Foo::num4:
            value = 4;
            break;
          case Foo::num5:
            value = 5;
            break;
          case Foo::num6:
            value = 6;
            break;
          case Foo::num7:
            value = 7;
            break;
          case Foo::num8:
            value = 8;
            break;
          case Foo::num9:
            value = 9;
            break;
          case Foo::num10:
            value = 10;
            break;
          case Foo::num11:
            value = 11;
            break;
          case Foo::num12:
            value = 12;
            break;
          case Foo::num13:
            value = 13;
            break;
          case Foo::num14:
            value = 14;
            break;
          case Foo::num15:
            value = 15;
            break;
          }
        }); // end runByValue func
    return value;
  }

  TEST(Tensor, RunByValue2) {
    ASSERT_EQ(testRunByValue2(Foo::num0), 0);
    ASSERT_EQ(testRunByValue2(Foo::num1), 1);
    ASSERT_EQ(testRunByValue2(Foo::num2), 2);
    ASSERT_EQ(testRunByValue2(Foo::num3), 3);
    ASSERT_EQ(testRunByValue2(Foo::num4), 4);
    ASSERT_EQ(testRunByValue2(Foo::num5), 5);
    ASSERT_EQ(testRunByValue2(Foo::num6), 6);
    ASSERT_EQ(testRunByValue2(Foo::num7), 7);
    ASSERT_EQ(testRunByValue2(Foo::num8), 8);
    ASSERT_EQ(testRunByValue2(Foo::num9), 9);
    ASSERT_EQ(testRunByValue2(Foo::num10), 10);
    ASSERT_EQ(testRunByValue2(Foo::num11), 11);
    ASSERT_EQ(testRunByValue2(Foo::num12), 12);
    ASSERT_EQ(testRunByValue2(Foo::num13), 13);
    ASSERT_EQ(testRunByValue2(Foo::num14), 14);
    ASSERT_EQ(testRunByValue2(Foo::num15), 15);
  }
}
HBTL_NAMESPACE_END
