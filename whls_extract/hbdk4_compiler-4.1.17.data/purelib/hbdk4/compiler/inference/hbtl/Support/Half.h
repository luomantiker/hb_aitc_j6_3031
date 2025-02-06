#pragma once

#include "hbtl/Support/Compiler.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <type_traits>

// TODO(hehaoqian): I want to use _Float16 types for X86
// but currently it has some problems in spdlog

HBTL_NAMESPACE_BEGIN {

#ifndef HBTL_USE_NATIVE_FLOAT16
  inline uint32_t fp32_to_bits(float val) {
    union {
      float as_value;
      uint32_t as_bits;
    } fp32 = {val};
    return fp32.as_bits;
  }

  inline float fp32_from_bits(uint32_t val) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {val};
    return fp32.as_value;
  }

  inline uint16_t floatToHalf(float val) {
    constexpr uint32_t scale_to_inf_bits = (uint32_t)239U << 23U;
    constexpr uint32_t scale_to_zero_bits = (uint32_t)17U << 23U;
    float scale_to_inf_val = 0;
    float scale_to_zero_val = 0;
    std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
    std::memcpy(&scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
    const float scale_to_inf = scale_to_inf_val;
    const float scale_to_zero = scale_to_zero_val;
    float base = (fabsf(val) * scale_to_inf) * scale_to_zero;
    const uint32_t w = fp32_to_bits(val);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000U;
    uint32_t bias = shl1_w & 0xFF000000U;
    if (bias < 0x71000000U) {
      bias = 0x71000000U;
    }
    base = fp32_from_bits((bias >> 1U) + 0x07800000U) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13U) & 0x00007C00U;
    const uint32_t mantissa_bits = bits & 0x00000FFFU;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<uint16_t>((sign >> 16U) | (shl1_w > 0xFF000000U ? 0x7E00U : nonsign));
  }

  inline float halfToFloat(uint16_t val) {
    const uint32_t w = (uint32_t)val << 16U;
    const uint32_t sign = w & 0x80000000U;
    const uint32_t two_w = w + w;
    constexpr uint32_t exp_offset = 0xE0U << 23U;
    constexpr uint32_t scale_bits = (uint32_t)15U << 23U;
    float exp_scale_val = 0;
    std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
    const float exp_scale = exp_scale_val;
    const float normalized_value = fp32_from_bits((two_w >> 4U) + exp_offset) * exp_scale;
    constexpr uint32_t magic_mask = 126U << 23U;
    constexpr float magic_bias = 0.5F;
    const float denormalized_value = fp32_from_bits((two_w >> 17U) | magic_mask) - magic_bias;
    constexpr uint32_t denormalized_cutoff = 1U << 27U;
    const uint32_t result =
        sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
  }

  class alignas(2) half final {
  public:
    half() = default;
    constexpr half(uint16_t bits) : data(bits) {} // NOLINT
    inline half(float value);                     // NOLINT
    inline half(double value);                    // NOLINT

    template <typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool> = true>
    inline half(Integer value); // NOLINT

    inline operator float() const; // NOLINT

  private:
    uint16_t data; // not allowed to access outside
  };

  // Constructor
  inline half::half(float value) : data(floatToHalf(value)) {}
  inline half::half(double value) : data(floatToHalf(static_cast<float>(value))) {}

  template <typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool>>
  inline half::half(Integer value) : data(floatToHalf(static_cast<float>(value))) {}

  // Conversion
  inline half::operator float() const { return halfToFloat(data); }

  // Arithmetic
  inline half operator+(const half &lhs, const half &rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }
  inline half operator-(const half &lhs, const half &rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }
  inline half operator*(const half &lhs, const half &rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }
  inline half operator/(const half &lhs, const half &rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }
  inline half operator%(const half &lhs, const half &rhs) {
    return std::fmod(static_cast<float>(lhs), static_cast<float>(rhs));
  }
  inline half operator-(const half &v) { return -static_cast<float>(v); }

  inline half &operator+=(half &lhs, const half &rhs) {
    lhs = lhs + rhs;
    return lhs;
  }
  inline half &operator-=(half &lhs, const half &rhs) {
    lhs = lhs - rhs;
    return lhs;
  }
  inline half &operator*=(half &lhs, const half &rhs) {
    lhs = lhs * rhs;
    return lhs;
  }
  inline half &operator/=(half &lhs, const half &rhs) {
    lhs = lhs / rhs;
    return lhs;
  }
  inline half &operator%=(half &lhs, const half &rhs) {
    lhs = lhs % rhs;
    return lhs;
  }

  // Arithmetic with float
  inline float operator+(half lhs, float rhs) { return static_cast<float>(lhs) + rhs; }
  inline float operator-(half lhs, float rhs) { return static_cast<float>(lhs) - rhs; }
  inline float operator*(half lhs, float rhs) { return static_cast<float>(lhs) * rhs; }
  inline float operator/(half lhs, float rhs) { return static_cast<float>(lhs) / rhs; }
  inline float operator%(half lhs, float rhs) { return std::fmod(static_cast<float>(lhs), rhs); }

  inline float operator+(float lhs, half rhs) { return lhs + static_cast<float>(rhs); }
  inline float operator-(float lhs, half rhs) { return lhs - static_cast<float>(rhs); }
  inline float operator*(float lhs, half rhs) { return lhs * static_cast<float>(rhs); }
  inline float operator/(float lhs, half rhs) { return lhs / static_cast<float>(rhs); }
  inline float operator%(float lhs, half rhs) { return std::fmod(lhs, static_cast<float>(rhs)); }

  inline float &operator+=(float &lhs, const half &rhs) { return lhs += static_cast<float>(rhs); }
  inline float &operator-=(float &lhs, const half &rhs) { return lhs -= static_cast<float>(rhs); }
  inline float &operator*=(float &lhs, const half &rhs) { return lhs *= static_cast<float>(rhs); }
  inline float &operator/=(float &lhs, const half &rhs) { return lhs /= static_cast<float>(rhs); }
  inline float &operator%=(float &lhs, const half &rhs) {
    lhs = lhs % rhs;
    return lhs;
  }

  // Arithmetic with double
  inline double operator+(half lhs, double rhs) { return static_cast<double>(lhs) + rhs; }
  inline double operator-(half lhs, double rhs) { return static_cast<double>(lhs) - rhs; }
  inline double operator*(half lhs, double rhs) { return static_cast<double>(lhs) * rhs; }
  inline double operator/(half lhs, double rhs) { return static_cast<double>(lhs) / rhs; }
  inline double operator%(half lhs, double rhs) { return std::fmod(static_cast<double>(lhs), rhs); }

  inline double operator+(double lhs, half rhs) { return lhs + static_cast<double>(rhs); }
  inline double operator-(double lhs, half rhs) { return lhs - static_cast<double>(rhs); }
  inline double operator*(double lhs, half rhs) { return lhs * static_cast<double>(rhs); }
  inline double operator/(double lhs, half rhs) { return lhs / static_cast<double>(rhs); }
  inline double operator%(double lhs, half rhs) { return std::fmod(lhs, static_cast<double>(rhs)); }

  // Arithmetic with int32
  inline half operator+(half lhs, int rhs) { return lhs + static_cast<float>(rhs); }
  inline half operator-(half lhs, int rhs) { return lhs - static_cast<float>(rhs); }
  inline half operator*(half lhs, int rhs) { return lhs * static_cast<float>(rhs); }
  inline half operator/(half lhs, int rhs) { return lhs / static_cast<float>(rhs); }
  inline half operator%(half lhs, int rhs) { return lhs % static_cast<float>(rhs); }

  inline half operator+(int lhs, half rhs) { return static_cast<float>(lhs) + rhs; }
  inline half operator-(int lhs, half rhs) { return static_cast<float>(lhs) - rhs; }
  inline half operator*(int lhs, half rhs) { return static_cast<float>(lhs) * rhs; }
  inline half operator/(int lhs, half rhs) { return static_cast<float>(lhs) / rhs; }
  inline half operator%(int lhs, half rhs) { return static_cast<float>(lhs) % rhs; }

  // Arithmetic with int64
  inline half operator+(half lhs, int64_t rhs) { return lhs + static_cast<float>(rhs); }
  inline half operator-(half lhs, int64_t rhs) { return lhs - static_cast<float>(rhs); }
  inline half operator*(half lhs, int64_t rhs) { return lhs * static_cast<float>(rhs); }
  inline half operator/(half lhs, int64_t rhs) { return lhs / static_cast<float>(rhs); }
  inline half operator%(half lhs, int64_t rhs) { return lhs % static_cast<float>(rhs); }

  inline half operator+(int64_t lhs, half rhs) { return static_cast<float>(lhs) + rhs; }
  inline half operator-(int64_t lhs, half rhs) { return static_cast<float>(lhs) - rhs; }
  inline half operator*(int64_t lhs, half rhs) { return static_cast<float>(lhs) * rhs; }
  inline half operator/(int64_t lhs, half rhs) { return static_cast<float>(lhs) / rhs; }
  inline half operator%(int64_t lhs, half rhs) { return static_cast<float>(lhs) % rhs; }

  inline bool operator<(const half &lhs, const half &rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }
  inline bool operator>(const half &lhs, const half &rhs) { return rhs < lhs; }
  inline bool operator<=(const half &lhs, const half &rhs) { return !(lhs > rhs); }
  inline bool operator>=(const half &lhs, const half &rhs) { return !(lhs < rhs); }
  inline bool operator==(const half &lhs, const half &rhs) {
    return static_cast<float>(lhs) == static_cast<float>(rhs);
  }
  inline bool operator!=(const half &lhs, const half &rhs) { return !(lhs == rhs); }

  inline bool operator<(const half &lhs, const int &rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }
  inline bool operator>(const half &lhs, const int &rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }
  inline bool operator<=(const half &lhs, const int &rhs) { return !(lhs > rhs); }
  inline bool operator>=(const half &lhs, const int &rhs) { return !(lhs < rhs); }
  inline bool operator==(const half &lhs, const int &rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }
  inline bool operator!=(const half &lhs, const int &rhs) { return !(lhs == rhs); }

  inline bool operator!(const half &v) { return v == 0; }

  std::ostream &operator<<(std::ostream &out, const half &value);
#endif // HBTL_USE_NATIVE_FLOAT16

  using half_t = half;

  inline uint16_t getRawData(const half_t &v) { return *reinterpret_cast<const uint16_t *>(&v); }
}
HBTL_NAMESPACE_END

#ifndef HBTL_USE_NATIVE_FLOAT16
namespace std {

template <> class numeric_limits<hbtl::half> {
public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;
  static constexpr hbtl::half min() { return (uint16_t)0x0400U; }
  static constexpr hbtl::half lowest() { return (uint16_t)0xFBFFU; }
  static constexpr hbtl::half max() { return (uint16_t)0x7BFFU; }
  static constexpr hbtl::half epsilon() { return (uint16_t)0x1400U; }
  static constexpr hbtl::half round_error() { return (uint16_t)0x3800U; }
  static constexpr hbtl::half infinity() { return (uint16_t)0x7C00U; }
  static constexpr hbtl::half quiet_NaN() { return (uint16_t)0x7E00U; }
  static constexpr hbtl::half signaling_NaN() { return (uint16_t)0x7D00U; }
  static constexpr hbtl::half denorm_min() { return (uint16_t)0x0001U; }
};

} // namespace std

#endif // HBTL_USE_NATIVE_FLOAT16
