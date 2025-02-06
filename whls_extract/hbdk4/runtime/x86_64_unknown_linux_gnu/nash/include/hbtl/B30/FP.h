#pragma once

#include "hbtl/Core/ElementType.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "FpCmodel/fpu.h"
#include "FpCmodel/libdpi.h"

HBTL_NAMESPACE_BEGIN {
  namespace b30 {
  namespace fp {
  enum FpType { FP16 = 0x50A, FP24 = 0x80F, FP32 = 0x817, BF16 = 0x807 };

#define GET_EXP_WIDTH(FpType) (static_cast<uint64_t>((static_cast<uint64_t>(FpType) >> 8U) & 0xFU))
#define GET_FRAC_WIDTH(FpType) ((static_cast<uint64_t>((static_cast<uint64_t>(FpType)) & 0xFFU)))

  enum FpRoundMode {
    EVEN, // nearest even
    ZERO, // zero
    PINF, // +inf
    NINF, // -inf
  };

  // union of two type, used to reinterpret between `T` and `R`
  template <typename T, typename R> union UT {
    T t;
    R r;
  };

  /// This is type class for float(fp16/fp32/bf16)
  /// Sig + Exp + fraction
  template <FpType T> class FloatT {
  public:
    FloatT() = default;
    using DataType = std::conditional_t<T == FP16, uint16_t, uint32_t>;

    explicit FloatT(int64_t v) { this->v = (static_cast<DataType>(v) & VALUE_MASK); }

    explicit FloatT(int64_t sign, int64_t exp, int64_t fraction) {
      v = setBitRange(FRAC_WIDTH - 1, 0, v, static_cast<DataType>(fraction));
      v = setBitRange(FRAC_WIDTH + EXP_WIDTH - 1, FRAC_WIDTH, v, static_cast<DataType>(exp));
      v = setBitRange(FRAC_WIDTH + EXP_WIDTH, FRAC_WIDTH + EXP_WIDTH, v, static_cast<DataType>(sign));
    }

    static FloatT FAdd(FloatT lhs, FloatT rhs, FpRoundMode mode = FpRoundMode::EVEN, bool ieee = false);
    static FloatT FSub(FloatT lhs, FloatT rhs, FpRoundMode mode = FpRoundMode::EVEN, bool ieee = false) {
      /// a - b = a + (-b)
      return FAdd(lhs, rhs.negative(), mode, ieee);
    }
    static FloatT FMul(FloatT lhs, FloatT rhs, FpRoundMode mode = FpRoundMode::EVEN, bool ieee = true);
    /// lhs < rhs. +0 is equal to -0 if `isSensitiveToZero` is true
    static bool FLess(FloatT lhs, FloatT rhs, bool isSensitiveToZero = true);
    static FloatT FMax(FloatT lhs, FloatT rhs) {
      // NOTE-FP-NAN: return nan if lhs or rhs is nan
      if (lhs.isNan() || rhs.isNan()) {
        return getFixedNan();
      }
      // max(+0/-0,-0/+0) = +0
      return FLess(lhs, rhs, false) ? rhs : lhs;
    }

    static FloatT FMin(FloatT lhs, FloatT rhs) {
      // NOTE-FP-NAN: return nan if lhs or rhs is nan
      if (lhs.isNan() || rhs.isNan()) {
        return getFixedNan();
      }
      // min(+0/-0,-0/+0) = -0
      return FLess(lhs, rhs, false) ? lhs : rhs;
    }

    /// common calculation
    bool operator<(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      return FLess(*this, ot, T != FP24);
    }
    bool operator>(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      return !(*this < ot) && (*this != ot); // NOLINT
    }
    bool operator==(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      if (isZero() && ot.isZero() && (T != FP24)) {
        return true; // +0 == -0
      }
      return v == ot.v;
    }
    bool operator!=(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      return !(*this == ot); // NOLINT
    }
    bool operator<=(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      return (*this < ot) || (*this == ot);
    }
    bool operator>=(const FloatT ot) const {
      // NOTE-FP-NAN: return false if input is nan
      if (isNan() || ot.isNan()) {
        return false;
      }
      return (*this > ot) || (*this == ot);
    }

    FloatT operator+(const FloatT ot) const { return FAdd(*this, ot); };
    FloatT operator-(const FloatT ot) const { return FSub(*this, ot); };
    FloatT operator*(const FloatT ot) const { return FMul(*this, ot); };

    // not a number
    [[nodiscard]] bool isNan() const { return (exponent() == EXP_MAX) && (fraction() != 0); }
    [[nodiscard]] bool isInf() const { return (exponent() == EXP_MAX) && (fraction() == 0); }
    // +inf
    [[nodiscard]] bool isPInf() const { return isPositive() && isInf(); }
    // -inf
    [[nodiscard]] bool isNInf() const { return !isPositive() && isInf(); }
    [[nodiscard]] bool isZero() const { return (exponent() == 0) && (fraction() == 0); }

    [[nodiscard]] bool isPositive() const { return sign() == 0; }
    [[nodiscard]] bool isNegative() const { return sign() == 1; }

    // subnormal: exp = 0, fraction != 0
    [[nodiscard]] bool isSubnormal() const { return (exponent() == 0) && (fraction() != 0); }

    static FloatT getFixedNan() { return FloatT<T>(FloatT<T>::FIXED_NAN_VALUE); }
    static FloatT getPInf() { return FloatT<T>(FloatT<T>::PINF_VALUE); }
    static FloatT getNInf() { return FloatT<T>(FloatT<T>::NINF_VALUE); }
    static FloatT getInf(bool sig) { return sig ? FloatT<T>(NINF_VALUE) : FloatT<T>(PINF_VALUE); }
    static FloatT getZero(bool sig) { return sig ? FloatT<T>(NZERO_VALUE) : FloatT<T>(PZERO_VALUE); }

    [[nodiscard]] FloatT abs() const {
      // NOTE-FP-NAN: return fixed nan(0x7C01) if input is nan
      if (isNan()) {
        return FloatT(FIXED_NAN_VALUE);
      }
      return isPositive() ? *(this) : negative();
    }

    /// return  -v
    [[nodiscard]] FloatT negative() const {

      auto sig = isPositive();

      auto res = setBitRange<bool>(EXP_WIDTH + FRAC_WIDTH, EXP_WIDTH + FRAC_WIDTH, v, sig);
      return FloatT<T>(res);
    }

    // Cast float from `T` to `OT`, roundMode is used for narrow cast (fp32 -> fp16)
    template <FpType OT> FloatT<OT> cast(FpRoundMode roundMode = EVEN) const {
      auto ftype_in = static_cast<bpufp::vpu_fp_type_e>(T);
      int fp_in = v;
      auto ftype_out = static_cast<bpufp::vpu_fp_type_e>(OT);
      int round_mode = static_cast<int>(roundMode);
      int out_data = 0;
      bpufp::fp2fp(ftype_in, fp_in, ftype_out, round_mode, &out_data);
      return FloatT<OT>(out_data);
    }

    [[nodiscard]] int64_t raw() const { return static_cast<int64_t>(v); }
    float toStdF32() {
      auto f32 = cast<FP32>();
      UT<uint32_t, float> t{0};
      t.t = static_cast<uint32_t>(f32.raw());
      return t.r;
    }

    template <typename U> U toInt(FpRoundMode roundMode = EVEN) {
      auto fp_type = static_cast<bpufp::vpu_fp_type_e>(T);
      int fdin = v;
      bpufp::vpu_int_type_e int_type;
      if constexpr (std::is_same_v<U, uint8_t>) {
        int_type = bpufp::UINT_8;
      } else if constexpr (std::is_same_v<U, int8_t>) {
        int_type = bpufp::SINT_8;
      } else if constexpr (std::is_same_v<U, uint16_t>) {
        int_type = bpufp::UINT_16;
      } else if constexpr (std::is_same_v<U, int16_t>) {
        int_type = bpufp::SINT_16;
      } else if constexpr (std::is_same_v<U, uint32_t>) {
        int_type = bpufp::UINT_32;
      } else if constexpr (std::is_same_v<U, int32_t>) {
        int_type = bpufp::SINT_32;
      } else {
        assert(false); // should not come here
      }
      int round_mode = static_cast<int>(roundMode);
      int out_data = 0;
      bpufp::fp2int(fp_type, fdin, int_type, round_mode, &out_data);
      return static_cast<U>(out_data);
    }

    template <typename U> static FloatT<T> fromInt(U iv, FpRoundMode roundMode = EVEN) {
      bpufp::vpu_int_type_e int_type;
      if constexpr (std::is_same_v<U, uint8_t>) {
        int_type = bpufp::UINT_8;
      } else if constexpr (std::is_same_v<U, int8_t>) {
        int_type = bpufp::SINT_8;
      } else if constexpr (std::is_same_v<U, uint16_t>) {
        int_type = bpufp::UINT_16;
      } else if constexpr (std::is_same_v<U, int16_t>) {
        int_type = bpufp::SINT_16;
      } else if constexpr (std::is_same_v<U, uint32_t>) {
        int_type = bpufp::UINT_32;
      } else if constexpr (std::is_same_v<U, int32_t>) {
        int_type = bpufp::SINT_32;
      } else {
        assert(false); // should not come here
      }
      auto fp_type = static_cast<bpufp::vpu_fp_type_e>(T);
      auto dint = static_cast<int>(iv);
      int round_mode = static_cast<int>(roundMode);
      int out_data = 0;
      bpufp::int2fp(int_type, dint, fp_type, round_mode, &out_data);
      return FloatT<T>(out_data);
    }

    // {S,E,M}
    // return E in float encoding
    [[nodiscard]] DataType exponent() const { return getBitRange<DataType>(FRAC_WIDTH + EXP_WIDTH - 1, FRAC_WIDTH, v); }
    // return e of math float,  e = E - bias
    [[nodiscard]] int64_t mExp() const {
      auto E = exponent();
      return (static_cast<int64_t>(E) - expBias());
    }

    [[nodiscard]] DataType fraction() const { return getBitRange<DataType>(FRAC_WIDTH - 1, 0, v); }
    [[nodiscard]] DataType sign() const {
      return getBitRange<DataType>(FRAC_WIDTH + EXP_WIDTH, FRAC_WIDTH + EXP_WIDTH, v);
    }
    // the bias between `E` and `e`
    static constexpr int64_t expBias() { return (1U << (EXP_WIDTH - 1)) - 1; };
    static constexpr FloatT max() { return min().abs(); }
    static constexpr FloatT min() { return FloatT(1, (1U << EXP_WIDTH) - 2, (1U << FRAC_WIDTH) - 1); }

  protected:
    DataType v{0}; // the raw data of fp

  private:
    static constexpr uint64_t EXP_WIDTH = GET_EXP_WIDTH(T);
    static constexpr uint64_t FRAC_WIDTH = GET_FRAC_WIDTH(T);
    static constexpr uint64_t ALL_WIDTH = 1 + EXP_WIDTH + FRAC_WIDTH;
    static constexpr uint64_t EXP_MAX = (1UL << EXP_WIDTH) - 1;
    static constexpr uint64_t EXP_MASK = EXP_MAX << FRAC_WIDTH;
    static constexpr uint64_t FRAC_MASK = (1UL << FRAC_WIDTH) - 1;
    static constexpr uint64_t SIG_MASK = 1U << (EXP_WIDTH + FRAC_WIDTH);
    static constexpr uint64_t VALUE_MASK = (1UL << ALL_WIDTH) - 1; // NOLINT
    static constexpr DataType FIXED_NAN_VALUE = EXP_MASK + 1;
    static constexpr DataType PINF_VALUE = EXP_MASK;            // +inf
    static constexpr DataType NINF_VALUE = SIG_MASK | EXP_MASK; // -inf
    static constexpr DataType PZERO_VALUE = 0;                  // +0
    static constexpr DataType NZERO_VALUE = SIG_MASK;           // -0
  };

  template <FpType T> FloatT<T> fromStdF32(float f);

  using F16 = FloatT<FpType::FP16>;
  using F32 = FloatT<FpType::FP32>;
  using F24 = FloatT<FpType::FP24>;

  } // namespace fp
  } // namespace b30

  inline b30::fp::F16 nativeHalfToFP16(const half_t val) { return b30::fp::F16(getRawData(val)); }
  inline half_t FP16ToNativeHalf(const b30::fp::F16 &val) { return *reinterpret_cast<const half_t *>(&val); }
}
HBTL_NAMESPACE_END
