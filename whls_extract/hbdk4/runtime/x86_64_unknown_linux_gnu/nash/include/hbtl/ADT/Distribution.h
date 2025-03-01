//===- Support/Distribution.h - Distribution classes ---------------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//
//
// Contain the code of std::xxx_distribution copied from libc++ of LLVM12
//
// This is to ensure our random number generation result does not affected
// by if we use libstdc++, or libc++, or version of the compiler
//
// Since the code is copied,
// only the minimum amount of modification
// is done to pass the compilation and lint.
// The variable name remains to begin with the underscore.
//===----------------------------------------------------------------------===//

#pragma once

#include "hbtl/Support/Compiler.h"

#include <cstdint>
#include <iterator>
#include <random>
#include <type_traits>

// Consider this header file as 3rdparty code, suppress most warnings
HBTL_PUSH_IGNORE_3RDPARTY_WARNING

HBTL_NAMESPACE_BEGIN {

  /// Count number of 0's from the most significant bit to the least
  ///   stopping at the first 1.
  ///
  /// Only unsigned integral types are allowed.
  ///
  /// Returns std::numeric_limits<T>::digits on an input of 0.
  template <typename T> HBTL_NODISCARD unsigned countl_zero(T Val) {
    static_assert(std::is_unsigned<T>::value, "Only unsigned integral types are allowed.");

    if (!Val)
      return std::numeric_limits<T>::digits;

    // Bisection method.
    unsigned ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }

  namespace dist_priv {
  template <unsigned long long _Xp, size_t _Rp> struct __log2_imp {
    static const size_t value =
        _Xp & ((unsigned long long)(1) << _Rp) ? _Rp : __log2_imp<_Xp, _Rp - 1>::value; // NOLINT
  };

  template <unsigned long long _Xp> struct __log2_imp<_Xp, 0> {
    static const size_t value = 0;
  };

  template <size_t _Rp> struct __log2_imp<0, _Rp> {
    static const size_t value = _Rp + 1;
  };

  template <class _UIntType, _UIntType _Xp> struct __log2 {
    static const size_t value = __log2_imp<_Xp, sizeof(_UIntType) * __CHAR_BIT__ - 1>::value;
  };

  template <class _Engine, class _UIntType> class __independent_bits_engine {
  public:
    // types
    typedef _UIntType result_type; // NOLINT

  private:
    typedef typename _Engine::result_type _Engine_result_type; // NOLINT
    // NOLINTNEXTLINE
    typedef typename std::conditional<sizeof(_Engine_result_type) <= sizeof(result_type), result_type,
                                      _Engine_result_type>::type _Working_result_type;

    _Engine &__e_;
    size_t __w_;
    size_t __w0_;
    size_t __n_;
    size_t __n0_;
    _Working_result_type __y0_;
    _Working_result_type __y1_;
    _Engine_result_type __mask0_;
    _Engine_result_type __mask1_;

    static constexpr const _Working_result_type _Rp = _Engine::max() - _Engine::min() + _Working_result_type(1);
    static constexpr const size_t __m = __log2<_Working_result_type, _Rp>::value;
    static constexpr const size_t _WDt = std::numeric_limits<_Working_result_type>::digits;
    static constexpr const size_t _EDt = std::numeric_limits<_Engine_result_type>::digits;

  public:
    // constructors and seeding functions
    __independent_bits_engine(_Engine &__e, size_t __w);

    // generating functions
    result_type operator()() { return __eval(std::integral_constant<bool, _Rp != 0>()); }

  private:
    result_type __eval(std::false_type); // NOLINT
    result_type __eval(std::true_type);  // NOLINT
  };

  template <class _Engine, class _UIntType>
  __independent_bits_engine<_Engine, _UIntType>::__independent_bits_engine(_Engine &__e, size_t __w)
      : __e_(__e), __w_(__w) {
    __n_ = __w_ / __m + (__w_ % __m != 0); // NOLINT
    __w0_ = __w_ / __n_;
    if (_Rp == 0) {
      __y0_ = _Rp;
    } else if (__w0_ < _WDt) {
      __y0_ = (_Rp >> __w0_) << __w0_;
    } else {
      __y0_ = 0;
    }
    if (_Rp - __y0_ > __y0_ / __n_) {
      ++__n_;
      __w0_ = __w_ / __n_;
      if (__w0_ < _WDt) {
        __y0_ = (_Rp >> __w0_) << __w0_;
      } else {
        __y0_ = 0;
      }
    }
    __n0_ = __n_ - __w_ % __n_;
    if (__w0_ < _WDt - 1) {
      __y1_ = (_Rp >> (__w0_ + 1)) << (__w0_ + 1);
    } else {
      __y1_ = 0;
    }
    __mask0_ = __w0_ > 0 ? _Engine_result_type(~0) >> (_EDt - __w0_) : _Engine_result_type(0);               // NOLINT
    __mask1_ = __w0_ < _EDt - 1 ? _Engine_result_type(~0) >> (_EDt - (__w0_ + 1)) : _Engine_result_type(~0); // NOLINT
  }

  template <class _Engine, class _UIntType>
  inline _UIntType __independent_bits_engine<_Engine, _UIntType>::__eval(std::false_type) { // NOLINT
    return static_cast<result_type>(__e_() & __mask0_);
  }

  template <class _Engine, class _UIntType>
  _UIntType __independent_bits_engine<_Engine, _UIntType>::__eval(std::true_type) { // NOLINT
    const size_t _WRt = std::numeric_limits<result_type>::digits;
    result_type _Sp = 0;
    for (size_t __k = 0; __k < __n0_; ++__k) {
      _Engine_result_type __u;
      do {
        __u = __e_() - _Engine::min();
      } while (__u >= __y0_);
      if (__w0_ < _WRt) {
        _Sp <<= __w0_;
      } else {
        _Sp = 0;
      }
      _Sp += __u & __mask0_;
    }
    for (size_t __k = __n0_; __k < __n_; ++__k) {
      _Engine_result_type __u;
      do {
        __u = __e_() - _Engine::min();
      } while (__u >= __y1_);
      if (__w0_ < _WRt - 1) {
        _Sp <<= __w0_ + 1;
      } else {
        _Sp = 0;
      }
      _Sp += __u & __mask1_;
    }
    return _Sp;
  }

  } // namespace dist_priv

  template <class _IntType = int> class uniform_int_distribution {
  public:
    // types
    typedef _IntType result_type; // NOLINT

    class param_type {
      result_type __a_;
      result_type __b_;

    public:
      typedef uniform_int_distribution distribution_type; // NOLINT

      explicit param_type(result_type __a = 0, result_type __b = std::numeric_limits<result_type>::max())
          : __a_(__a), __b_(__b) {}

      HBTL_NODISCARD result_type a() const { return __a_; }
      HBTL_NODISCARD result_type b() const { return __b_; }

      friend bool operator==(const param_type &__x, const param_type &__y) {
        return __x.__a_ == __y.__a_ && __x.__b_ == __y.__b_;
      }
      friend bool operator!=(const param_type &__x, const param_type &__y) { return !(__x == __y); }
    };

  private:
    param_type __p_;

  public:
    // constructors and reset functions
    uniform_int_distribution() : uniform_int_distribution(0) {}
    explicit uniform_int_distribution(result_type __a, result_type __b = std::numeric_limits<result_type>::max())
        : __p_(param_type(__a, __b)) {}

    explicit uniform_int_distribution(const param_type &__p) : __p_(__p) {}
    void reset() {}

    // generating functions
    template <class _URNG> result_type operator()(_URNG &__g) { return (*this)(__g, __p_); }
    template <class _URNG> result_type operator()(_URNG &__g, const param_type &__p);

    // property functions
    HBTL_NODISCARD result_type a() const { return __p_.a(); }
    HBTL_NODISCARD result_type b() const { return __p_.b(); }

    HBTL_NODISCARD param_type param() const { return __p_; }
    void param(const param_type &__p) { __p_ = __p; }

    HBTL_NODISCARD result_type min() const { return a(); }
    HBTL_NODISCARD result_type max() const { return b(); }

    friend bool operator==(const uniform_int_distribution &__x, const uniform_int_distribution &__y) {
      return __x.__p_ == __y.__p_;
    }
    friend bool operator!=(const uniform_int_distribution &__x, const uniform_int_distribution &__y) {
      return !(__x == __y);
    }
  };

  template <class _IntType>
  template <class _URNG>
  typename uniform_int_distribution<_IntType>::result_type uniform_int_distribution<_IntType>::operator()(
      _URNG &__g, const param_type &__p) {
    // NOLINTNEXTLINE
    typedef typename std::conditional<sizeof(result_type) <= sizeof(uint32_t), uint32_t, uint64_t>::type _UIntType;
    const _UIntType _Rp = _UIntType(__p.b()) - _UIntType(__p.a()) + _UIntType(1);
    if (_Rp == 1) {
      return __p.a();
    }
    const size_t _Dt = std::numeric_limits<_UIntType>::digits;
    typedef dist_priv::__independent_bits_engine<_URNG, _UIntType> _Eng; // NOLINT
    if (_Rp == 0) {
      return static_cast<result_type>(_Eng(__g, _Dt)());
    }
    size_t __w = _Dt - countl_zero(_Rp) - 1;
    if ((_Rp & (std::numeric_limits<_UIntType>::max() >> (_Dt - __w))) != 0) {
      ++__w;
    }
    _Eng __e(__g, __w);
    _UIntType __u;
    do {
      __u = __e();
    } while (__u >= _Rp);
    return static_cast<result_type>(__u + __p.a());
  }

  template <class _RandomAccessIterator, class _UniformRandomNumberGenerator>
  void shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last, _UniformRandomNumberGenerator && __g) {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type difference_type; // NOLINT
    typedef uniform_int_distribution<ptrdiff_t> _Dp;                                               // NOLINT
    typedef typename _Dp::param_type _Pp;                                                          // NOLINT
    difference_type __d = __last - __first;
    if (__d > 1) {
      _Dp __uid;
      for (--__last, (void)--__d; __first < __last; ++__first, (void)--__d) {
        difference_type __i = __uid(__g, _Pp(0, __d));
        if (__i != difference_type(0)) {
          std::swap(*__first, *(__first + __i));
        }
      }
    }
  }

  template <class _RealType = double> class uniform_real_distribution {
  public:
    // types
    typedef _RealType result_type; // NOLINT

    class param_type {
      result_type __a_;
      result_type __b_;

    public:
      typedef uniform_real_distribution distribution_type; // NOLINT

      explicit param_type(result_type __a = 0, result_type __b = 1) : __a_(__a), __b_(__b) {}

      HBTL_NODISCARD result_type a() const { return __a_; }
      HBTL_NODISCARD result_type b() const { return __b_; }

      friend bool operator==(const param_type &__x, const param_type &__y) {
        return __x.__a_ == __y.__a_ && __x.__b_ == __y.__b_;
      }
      friend bool operator!=(const param_type &__x, const param_type &__y) { return !(__x == __y); }
    };

  private:
    param_type __p_;

  public:
    // constructors and reset functions
    uniform_real_distribution() : uniform_real_distribution(0) {}
    explicit uniform_real_distribution(result_type __a, result_type __b = 1) : __p_(param_type(__a, __b)) {}
    explicit uniform_real_distribution(const param_type &__p) : __p_(__p) {}
    void reset() {}

    // generating functions
    template <class _URNG> HBTL_NODISCARD result_type operator()(_URNG &__g) { return (*this)(__g, __p_); }
    template <class _URNG> HBTL_NODISCARD result_type operator()(_URNG &__g, const param_type &__p);

    // property functions
    HBTL_NODISCARD result_type a() const { return __p_.a(); }
    HBTL_NODISCARD result_type b() const { return __p_.b(); }

    HBTL_NODISCARD param_type param() const { return __p_; }
    void param(const param_type &__p) { __p_ = __p; }

    HBTL_NODISCARD result_type min() const { return a(); }
    HBTL_NODISCARD result_type max() const { return b(); }

    friend bool operator==(const uniform_real_distribution &__x, const uniform_real_distribution &__y) {
      return __x.__p_ == __y.__p_;
    }
    friend bool operator!=(const uniform_real_distribution &__x, const uniform_real_distribution &__y) {
      return !(__x == __y);
    }
  };

  template <class _RealType>
  template <class _URNG>
  inline typename uniform_real_distribution<_RealType>::result_type uniform_real_distribution<_RealType>::operator()(
      _URNG &__g, const param_type &__p) {
    return (__p.b() - __p.a()) * std::generate_canonical<_RealType, std::numeric_limits<_RealType>::digits>(__g) +
           __p.a();
  }

  template <class _RealType = double> class normal_distribution {
  public:
    // types
    typedef _RealType result_type; // NOLINT

    class param_type {
      result_type __mean_;
      result_type __stddev_;

    public:
      typedef normal_distribution distribution_type; // NOLINT

      explicit param_type(result_type __mean = 0, result_type __stddev = 1) : __mean_(__mean), __stddev_(__stddev) {}

      HBTL_NODISCARD result_type mean() const { return __mean_; }

      HBTL_NODISCARD result_type stddev() const { return __stddev_; }

      friend bool operator==(const param_type &__x, const param_type &__y) {
        return __x.__mean_ == __y.__mean_ && __x.__stddev_ == __y.__stddev_;
      }
      friend bool operator!=(const param_type &__x, const param_type &__y) { return !(__x == __y); }
    };

  private:
    param_type __p_;
    result_type _V_ = 0;
    bool _V_hot_ = false;

  public:
    // constructors and reset functions

    normal_distribution() : normal_distribution(0) {}

    explicit normal_distribution(result_type __mean, result_type __stddev = 1) : __p_(param_type(__mean, __stddev)) {}

    explicit normal_distribution(const param_type &__p) : __p_(__p) {}

    void reset() { _V_hot_ = false; }

    // generating functions
    template <class _URNG>

    result_type operator()(_URNG &__g) {
      return (*this)(__g, __p_);
    }
    template <class _URNG> result_type operator()(_URNG &__g, const param_type &__p);

    // property functions

    HBTL_NODISCARD result_type mean() const { return __p_.mean(); }

    HBTL_NODISCARD result_type stddev() const { return __p_.stddev(); }

    HBTL_NODISCARD param_type param() const { return __p_; }

    void param(const param_type &__p) { __p_ = __p; }

    HBTL_NODISCARD result_type min() const { return -std::numeric_limits<result_type>::infinity(); }

    HBTL_NODISCARD result_type max() const { return std::numeric_limits<result_type>::infinity(); }

    friend bool operator==(const normal_distribution &__x, const normal_distribution &__y) {
      return __x.__p_ == __y.__p_ && __x._V_hot_ == __y._V_hot_ && (!__x._V_hot_ || __x._V_ == __y._V_);
    }
    friend bool operator!=(const normal_distribution &__x, const normal_distribution &__y) { return !(__x == __y); }
  };

  template <class _RealType>
  template <class _URNG>
  _RealType normal_distribution<_RealType>::operator()(_URNG &__g, const param_type &__p) {
    result_type _Up;
    if (_V_hot_) {
      _V_hot_ = false;
      _Up = _V_;
    } else {
      uniform_real_distribution<result_type> _Uni(-1, 1);
      result_type __u;
      result_type __v;
      result_type __s;
      do {
        __u = _Uni(__g);
        __v = _Uni(__g);
        __s = __u * __u + __v * __v;
      } while (__s > 1 || __s == 0);
      result_type _Fp = std::sqrt(-2 * std::log(__s) / __s);
      _V_ = __v * _Fp;
      _V_hot_ = true;
      _Up = __u * _Fp;
    }
    return _Up * __p.stddev() + __p.mean();
  }
}
HBTL_NAMESPACE_END

HBTL_POP_IGNORE_3RDPARTY_WARNING
