// HBTL ADTExtras.h

#pragma once
#ifndef HBTL_SUPPORT_ADTEXTRAS_H_
#define HBTL_SUPPORT_ADTEXTRAS_H_

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/ADT/STLExtras.h"
#include "hbtl/Support/Compiler.h"

#include <vector>

HBTL_NAMESPACE_BEGIN {

  /****************************************************
   * Binary element-wise operation on two vectors (or a broad-casted scalar), generating a new vector
   ****************************************************/

#define BINARY_ELTWISE_OP(T, name, func)                                                                               \
  inline std::vector<T> name(ArrayRef<T> lhs, ArrayRef<T> rhs) {                                                       \
    auto minSize = std::min(lhs.size(), rhs.size());                                                                   \
    auto maxSize = std::max(lhs.size(), rhs.size());                                                                   \
    if (minSize != maxSize) {                                                                                          \
      assert(minSize == 1 && "binary broadcast operation's operands should have the same size or 1");                  \
    }                                                                                                                  \
    std::vector<T> result(maxSize);                                                                                    \
    for (size_t i = 0; i < maxSize; ++i) {                                                                             \
      result[i] = func(lhs[i % lhs.size()], rhs[i % rhs.size()]);                                                      \
    }                                                                                                                  \
    return result;                                                                                                     \
  }

#define BINARY_ELTWISE_OP_WITH_BROADCAST(T, name, func)                                                                \
  BINARY_ELTWISE_OP(T, name, func)                                                                                     \
  inline std::vector<T> name(ArrayRef<T> lhs, T rhs) {                                                                 \
    std::vector<T> result(lhs.size());                                                                                 \
    for (size_t i = 0; i < lhs.size(); ++i) {                                                                          \
      result[i] = static_cast<T>(func(lhs[i], rhs));                                                                   \
    }                                                                                                                  \
    return result;                                                                                                     \
  }

  // the following operators should be defined outside the `vector` namespace
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, operator+, std::plus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, operator-, std::minus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, operator*, std::multiplies<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, operator/, std::divides<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, operator%, std::modulus<>())

  BINARY_ELTWISE_OP_WITH_BROADCAST(double, operator+, std::plus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, operator-, std::minus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, operator*, std::multiplies<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, operator/, std::divides<>())

  namespace vector {

  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, add, std::plus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, sub, std::minus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, mul, std::multiplies<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, div, std::divides<>()) // round to zero

  BINARY_ELTWISE_OP_WITH_BROADCAST(double, add, std::plus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, sub, std::minus<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, mul, std::multiplies<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, div, std::divides<>()) // round to zero

  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, mod, std::modulus<>())

  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, max, std::max)
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, min, std::min)

  BINARY_ELTWISE_OP_WITH_BROADCAST(double, max, std::max)
  BINARY_ELTWISE_OP_WITH_BROADCAST(double, min, std::min)

  // Element-wise compare, return a vector. For example: eleEq({4, 5, 6}, {5, 5, 5}) -> {0, 1, 0}
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltEq, std::equal_to<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltNe, std::not_equal_to<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltGt, std::greater<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltGe, std::greater_equal<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltLt, std::less<>())
  BINARY_ELTWISE_OP_WITH_BROADCAST(int64_t, eltLe, std::less_equal<>())

#undef BINARY_ELTWISE_OP
#undef BINARY_ELTWISE_OP_WITH_BROADCAST

  /****************************************************
   * Unary element-wise operation using one vector, generating a new vector
   ****************************************************/

  inline std::vector<int64_t> copy(ArrayRef<int64_t> values) { return {values.begin(), values.end()}; }

#define UNARY_ELTWISE_OP(name, func)                                                                                   \
  inline std::vector<int64_t> name(ArrayRef<int64_t> values) {                                                         \
    std::vector<int64_t> results(values.size());                                                                       \
    for (uint64_t i = 0U; i < values.size(); ++i) {                                                                    \
      results[i] = func(values[i]);                                                                                    \
    }                                                                                                                  \
    return results;                                                                                                    \
  }

#undef UNARY_ELTWISE_OP

  /****************************************************
   * Reduce on two vectors (or a broad-casted scalar), generating a scalar bool
   ****************************************************/

#define BINARY_REDUCE_OP_INNER(T, name, keyword, func)                                                                 \
  inline bool keyword##name(ArrayRef<T> lhs, ArrayRef<T> rhs) {                                                        \
    assert((lhs.size() == rhs.size()) && "binary operation's operands should have the same size");                     \
    return keyword##_of(zip(lhs, rhs), [](const auto &p) { return func(std::get<0>(p), std::get<1>(p)); });            \
  }                                                                                                                    \
  inline bool keyword##name(ArrayRef<T> lhs, T rhs) {                                                                  \
    return keyword##_of(lhs, [rhs](T v) { return func(v, rhs); });                                                     \
  }

#define BINARY_REDUCE_OP(T, name, func)                                                                                \
  BINARY_REDUCE_OP_INNER(T, name, all, func)                                                                           \
  BINARY_REDUCE_OP_INNER(T, name, any, func)                                                                           \
  BINARY_REDUCE_OP_INNER(T, name, none, func)

  BINARY_REDUCE_OP(int64_t, Eq, std::equal_to<>())
  BINARY_REDUCE_OP(int64_t, Ne, std::not_equal_to<>())
  BINARY_REDUCE_OP(int64_t, Gt, std::greater<>())
  BINARY_REDUCE_OP(int64_t, Ge, std::greater_equal<>())
  BINARY_REDUCE_OP(int64_t, Lt, std::less<>())
  BINARY_REDUCE_OP(int64_t, Le, std::less_equal<>())

  BINARY_REDUCE_OP(double, Eq, std::equal_to<>())
  BINARY_REDUCE_OP(double, Ne, std::not_equal_to<>())
  BINARY_REDUCE_OP(double, Gt, std::greater<>())
  BINARY_REDUCE_OP(double, Ge, std::greater_equal<>())
  BINARY_REDUCE_OP(double, Lt, std::less<>())
  BINARY_REDUCE_OP(double, Le, std::less_equal<>())

#undef BINARY_REDUCE_OP_INNER
#undef BINARY_REDUCE_OP

  /****************************************************
   * Reduce on a vector, generating a scalar
   ****************************************************/

#define UNARY_REDUCE_OP(T, name, init, op)                                                                             \
  inline T name(ArrayRef<T> values) {                                                                                  \
    T result = init;                                                                                                   \
    for (auto v : values) {                                                                                            \
      result = op(v, result);                                                                                          \
    }                                                                                                                  \
    return result;                                                                                                     \
  }

  UNARY_REDUCE_OP(int64_t, reduceAdd, 0, std::plus<>())
  UNARY_REDUCE_OP(int64_t, reduceMul, 1, std::multiplies<>())
  UNARY_REDUCE_OP(int64_t, reduceMax, std::numeric_limits<int64_t>::min(), std::max)

  UNARY_REDUCE_OP(uint64_t, reduceAnd, std::numeric_limits<uint64_t>::max(), std::logical_and<>())
  UNARY_REDUCE_OP(uint64_t, reduceOr, 0U, std::logical_or<>())
  UNARY_REDUCE_OP(uint64_t, reduceMax, 0U, std::max)

  UNARY_REDUCE_OP(double, reduceAdd, 0, std::plus<>())
  UNARY_REDUCE_OP(double, reduceMul, 1, std::multiplies<>())
  UNARY_REDUCE_OP(double, reduceMax, std::numeric_limits<double>::min(), std::max)

#undef UNARY_REDUCE_OP
  } // namespace vector

  std::string toString(ArrayRef<int64_t> values, size_t max = 4UL);

  inline std::string toString(const std::vector<int64_t> &values, size_t max = 4UL) {
    return toString(ArrayRef<int64_t>(values), max);
  }

  template <typename T> bool inRangeOf(ArrayRef<T> range, const T &e) {
    return std::find(range.begin(), range.end(), e) != range.end();
  }
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_ADTEXTRAS_H_
