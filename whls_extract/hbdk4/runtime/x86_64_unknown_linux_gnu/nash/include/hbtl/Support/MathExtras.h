// HBTL MathExtras

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>

#include "hbtl/Support/Compiler.h"

HBTL_NAMESPACE_BEGIN {

  enum class RoundMode {
    FLOOR = 0,
    ROUND = 1,
    CEIL = 2,
    UNKNOWN = 3,
  };

  inline RoundMode symbolizeRoundMode(const std::string &round) {
    if (round == "ROUND") {
      return RoundMode::ROUND;
    } else if (round == "FLOOR") {
      return RoundMode::FLOOR;
    } else if (round == "CEIL") {
      return RoundMode::CEIL;
    } else {
      return RoundMode::UNKNOWN;
    }
  }

  constexpr inline const char *toString(RoundMode m) {
    switch (m) {
    case RoundMode::FLOOR:
      return "floor";
    case RoundMode::ROUND:
      return "round";
    case RoundMode::CEIL:
      return "ceil";
    default:
      return "unknown";
    }
  }

  template <typename T, typename S>
  constexpr inline T arithRightShift(T x, S rn, RoundMode roundMode = RoundMode::FLOOR) {
    static_assert(std::is_integral<T>::value, ""); // only accept integer type
    using uS = typename std::make_unsigned<S>::type;
    uS n = static_cast<uS>(rn);
    assert((n >= 0) && (n < sizeof(T) * 8U)); // bugprone when n exceeds range

    if (roundMode == RoundMode::FLOOR) {
      x >>= n;
    } else if (roundMode == RoundMode::ROUND) {
      auto carry = (n > 0) ? (x >> (n - 1) & 1) : 0;
      x >>= n;
      x += carry;
    } else if (roundMode == RoundMode::CEIL) {
      auto mask = (n > 0) ? (1ULL << n) - 1ULL : 0ULL;
      using BigT = typename std::conditional<std::is_signed<T>::value, int64_t, uint64_t>::type;
      BigT val = static_cast<BigT>(x);
      val += static_cast<T>(mask);
      x = static_cast<T>(val >> n);
    }
    return x;
  }

  template <typename T> constexpr inline T clamp(T v, T lo, T hi) {
    assert(lo <= hi);
    return (v < lo) ? lo : ((hi < v) ? hi : v);
  }

  template <typename T> constexpr inline T round(T v) { return std::round(v); }

  template <typename T> constexpr inline T remainder(T a, T b) {
    if HBTL_CONSTEXPR_IF (std::is_integral<T>::value) {
      auto remainder = static_cast<T>(a % b);
      if (remainder < 0) {
        remainder = static_cast<T>(remainder + b);
      }
      return remainder;
    } else {
      return static_cast<T>(a - static_cast<T>(floor(a / b)) * b);
    }
  }

  template <typename T> constexpr inline T min(T a, T b) { return std::min(a, b); }

  template <typename T> constexpr inline T max(T a, T b) { return std::max(a, b); }

  template <typename numericT, typename T> constexpr inline T clampNumericLimit(T v) {
    const auto min = static_cast<T>(std::numeric_limits<numericT>::min());
    static_assert(min == std::numeric_limits<numericT>::min(), "");
    const auto max = static_cast<T>(std::numeric_limits<numericT>::max());
    static_assert(max == std::numeric_limits<numericT>::max(), "");
    return clamp(v, min, max);
  }

  template <typename numericT, typename T> constexpr numericT clampCast(T v) {
    return static_cast<numericT>(clampNumericLimit<numericT>(v));
  }

  template <typename T> inline uint32_t getValidBitNum(T value) {
    static_assert(std::is_integral<T>::value, ""); // only accept integer
    if (std::is_signed<T>::value) {
      auto floatValue =
          static_cast<float>((value < 0) ? value + 1 : value); // e.g. given: -2, -1, 0, 1, should use 2bit
      int32_t exponent = 0;
      frexpf(floatValue, &exponent);
      assert(exponent >= 0);
      return static_cast<uint32_t>(exponent) + 1U;
    } else {
      auto floatValue = static_cast<float>(value);
      int32_t exponent = 0;
      frexpf(floatValue, &exponent);
      assert(exponent >= 0);
      return std::max(static_cast<uint32_t>(exponent), 1U);
    }
  }

  /// Extend bits from bitNum to bit size of T. Fill high order of bits with msb.
  template <typename T> inline T signBitExtension(uint32_t bits, uint32_t bitNum) {
    assert(sizeof(T) * 8U >= bitNum);
    static_assert(std::is_integral<T>::value, ""); // only accept integer
    uint32_t mask = 1U << (bitNum - 1U);
    return static_cast<T>((bits ^ mask) - mask);
  }

  template <typename T> constexpr inline uint32_t bitWidth() { return static_cast<uint32_t>(sizeof(T) * 8); }

  template <typename T> constexpr inline uint32_t signBit() { return static_cast<uint32_t>(sizeof(T) * 8 - 1); }

  /// Logical Right shift, the top bits are zero
  template <typename T> constexpr inline T logicalRightShift(T data, uint32_t n) {
    assert(n < bitWidth<T>());
    using unsigned_type = typename std::make_unsigned<T>::type;
    const auto result = static_cast<unsigned_type>(data) >> n;
    return static_cast<T>(result);
  }

  template <typename FieldType, typename DataType>
  constexpr inline FieldType getBitRange(uint32_t hi, uint32_t lo, DataType data) {
    static_assert(std::is_integral<DataType>::value, "bad data Type");
    static_assert(std::is_integral<FieldType>::value, "bad field type");
    static_assert(sizeof(DataType) >= sizeof(FieldType), "Sizeof data type should >= field type");
    assert((hi < bitWidth<DataType>()) && "Bad high bit");
    assert((lo <= hi) && "lo should <= hi");
    const auto temp = data << (signBit<DataType>() - hi);
    const auto shift = signBit<DataType>() - hi + lo;

    if HBTL_CONSTEXPR_IF (std::is_unsigned<FieldType>::value) {
      auto res = logicalRightShift(temp, shift);
      // we need a mask of (hi-lo+1) bits of 1
      const auto maskBitsNum = hi - lo + 1U;
      // generate a mask of many bits
      using UnsignedDataType = typename std::make_unsigned_t<DataType>;
      auto mask = static_cast<UnsignedDataType>((1UL << maskBitsNum) - 1UL);
      return static_cast<FieldType>(res & mask);
    } else {
      auto res = arithRightShift(temp, shift);
      return static_cast<FieldType>(res);
    }
  }

  /// Get the bits between [hi, lo] of s by field f
  /// Both hi and lo are inclusive
  template <uint32_t hi, uint32_t lo, typename FieldType, typename DataType>
  constexpr inline FieldType getBitRange(DataType data) {
    static_assert(std::is_integral<DataType>::value, "bad data Type");
    static_assert(std::is_integral<FieldType>::value, "bad field type");
    static_assert(sizeof(DataType) >= sizeof(FieldType), "Sizeof data type should >= field type");
    static_assert((lo >= 0), "Bad low bit");
    static_assert((hi < bitWidth<DataType>()), "Bad high bit");
    static_assert((lo <= hi), "lo should <= hi");
    return getBitRange<FieldType>(hi, lo, data);
  }

  /// Set the bits between [hi, lo] of s by field f
  /// Both hi and lo are inclusive
  /// "field" must have value representable by (hi - lo + 1) bits,
  /// which is affected by signedness of FieldType
  template <typename FieldType, typename DataType>
  constexpr inline DataType setBitRange(uint32_t hi, uint32_t lo, DataType data, FieldType field) {
    static_assert(std::is_integral<DataType>::value, "bad data Type");
    static_assert(std::is_integral<FieldType>::value, "bad field type");
    static_assert(sizeof(DataType) >= sizeof(FieldType), "Sizeof data type should >= field type");
    assert((hi < bitWidth<DataType>()) && "Bad high bit");
    assert((lo <= hi) && "lo should <= hi");

    // we need a mask of (hi-lo+1) bits of 1
    const auto maskBitsNum = hi - lo + 1U;

    // generate a mask of many bits first
    using UnsignedDataType = typename std::make_unsigned_t<DataType>;
    auto mask = std::numeric_limits<UnsignedDataType>::max();

    // if we actually need fewer bits, modify it
    if (maskBitsNum < bitWidth<DataType>()) {
      mask = static_cast<UnsignedDataType>(mask >> (bitWidth<DataType>() - maskBitsNum) << lo);
    }

    return static_cast<DataType>((static_cast<UnsignedDataType>(data) & (~mask)) |
                                 ((static_cast<UnsignedDataType>(field) << lo) & mask));
  }

  /// Set the bits between [hi, lo] of s by field f
  /// Both hi and lo are inclusive
  /// "field" must have value representable by (hi - lo + 1) bits,
  /// which is affected by signedness of FieldType
  template <uint32_t hi, uint32_t lo, typename FieldType, typename DataType>
  constexpr inline DataType setBitRange(DataType data, FieldType field) {
    static_assert(std::is_integral<DataType>::value, "bad data Type");
    static_assert(std::is_integral<FieldType>::value, "bad field type");
    static_assert(sizeof(DataType) >= sizeof(FieldType), "Sizeof data type should >= field type");
    static_assert((lo >= 0), "Bad low bit");
    static_assert((hi < bitWidth<DataType>()), "Bad high bit");
    static_assert((lo <= hi), "lo should <= hi");
    return setBitRange(hi, lo, data, field);
  }

  template <typename T, typename S> constexpr inline T leftShift(T data, S rn) {
    using uS = typename std::make_unsigned<S>::type;
    auto n = static_cast<uS>(rn);
    assert((n >= 0) && (n < bitWidth<T>()));
    using unsigned_type = typename std::make_unsigned<T>::type;
    const auto result = static_cast<unsigned_type>(data) << n;
    return static_cast<T>(result);
  }

  inline int64_t alignTo(int64_t Value, int64_t Align, int64_t Skew = 0) {
    assert(Align != 0 && "Align can't be 0.");
    Skew %= Align;
    return (Value + Align - 1 - Skew) / Align * Align + Skew;
  }

  /**
   * Return `numerator` / `denominator`, and make sure the remainder must be 0.
   */
  inline int64_t divideExact(int64_t numerator, int64_t denominator) {
    assert((denominator != 0) && "denominator cannot be 0");
    assert(((numerator % denominator) == 0) && "cannot be divided exactly, check the caller");
    return numerator / denominator;
  }

  /**
   * Return `numerator` / `denominator`, rounding to +inf.
   */
  inline int64_t divCeil(int64_t numerator, int64_t denominator) {
    assert((denominator != 0) && "denominator cannot be 0");
    auto result = numerator / denominator; // round to zero, always correct for negative result

    if ((numerator % denominator) != 0) {         // has remainder
      if ((numerator < 0) == (denominator < 0)) { // same sign -> float result is positive (e.g., 0.5)
        result += 1;
      }
    }
    return result;
  }

  inline int64_t alignDown(int64_t Value, int64_t Align, int64_t Skew = 0) {
    assert(Align != 0 && "Align can't be 0.");
    Skew %= Align;
    return (Value - Skew) / Align * Align + Skew;
  }

  template <typename T> inline T alignTo(T Value, T Align, T Skew = 0) {
    assert(Align != 0 && "Align can't be 0.");
    Skew %= Align;
    return (Value + Align - 1 - Skew) / Align * Align + Skew;
  }

  template <typename T> inline uint32_t getBitNum(T num) {
    static_assert(std::is_integral<T>::value, "only support integer");

    auto uNum = static_cast<uint64_t>(num); // Change to unsigned integer
    if HBTL_CONSTEXPR_IF (std::is_signed<T>::value) {
      if (num < 0) {
        uNum = ~(uNum); // Examples: -8 (0xfff8) -> 7 (0x0007),  -1 (0xffff) -> 0 (0x0000), will get same bit num
      }
    }

    uint64_t count = (std::is_signed<T>::value) ? 1UL : 0UL; // Sign bit is counted for signed type
    auto halfBitWidth = static_cast<uint64_t>(sizeof(T) * 4U);
    while (halfBitWidth > 0) {
      const auto higherHalf = (uNum >> halfBitWidth); // Example: T=uint32_t, higherHalf is higher 16 bits
      if (higherHalf != 0) {
        count += halfBitWidth;
        uNum = higherHalf; // Example: continue checking the higher 16 bits
      }
      halfBitWidth /= 2UL;
    }

    count += uNum; // halfBitWidth=0 now, so only 1 useful bit, so uNum must be 0 or 1
    return static_cast<uint32_t>(std::max(1UL, count)); // at least 1 bit
  }

  template <typename T> inline bool isInRange(int64_t value) {
    static_assert(std::is_integral<T>::value, ""); // only accept integer
    return std::numeric_limits<T>::min() <= value && value <= std::numeric_limits<T>::max();
  }

  /// Check if two float values are equal
  inline bool floatEq(float a, float b) { return std::abs(a - b) < std::numeric_limits<float>::epsilon(); }
  inline bool floatEq(double a, double b) { return std::abs(a - b) < std::numeric_limits<double>::epsilon(); }
  inline bool floatLt(float a, float b) { return !floatEq(a, b) && (a < b); }
  inline bool floatLt(double a, double b) { return !floatEq(a, b) && (a < b); }
  inline bool floatGt(float a, float b) { return !floatEq(a, b) && (a > b); }
  inline bool floatGt(double a, double b) { return !floatEq(a, b) && (a > b); }
}
HBTL_NAMESPACE_END
