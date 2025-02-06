#pragma once

#include "Common.h"
#include "Compiler.h"
#include "Protocols.h"
#include "Types.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <type_traits>

namespace ude {

class Variable {

private:
  struct Base {
    virtual ~Base() = default;
  };

  template <typename T> struct Derived : public Base {
    explicit Derived(T &&value) : value(std::forward<T>(value)) {} // move the value
    explicit Derived(T &value) : value(value) {}
    T value;
  };

public:
  explicit Variable(bool value) {
    u.as_bool = value;
    type = Type::bool8;
  }

  explicit Variable(int8_t value) {
    u.as_int8 = value;
    type = Type::si8;
  }

  explicit Variable(int16_t value) {
    u.as_int16 = value;
    type = Type::si16;
  }

  explicit Variable(int32_t value) {
    u.as_int32 = value;
    type = Type::si32;
  }

  explicit Variable(int64_t value) {
    u.as_int64 = value;
    type = Type::si64;
  }

  explicit Variable(uint8_t value) {
    u.as_uint8 = value;
    type = Type::ui8;
  }

  explicit Variable(uint16_t value) {
    u.as_uint16 = value;
    type = Type::ui16;
  }

  explicit Variable(uint32_t value) {
    u.as_uint32 = value;
    type = Type::ui32;
  }

  explicit Variable(uint64_t value) {
    u.as_uint64 = value;
    type = Type::ui64;
  }

  explicit Variable(double value) {
    u.as_double = value;
    type = Type::f64;
  }

  explicit Variable(float value) {
    u.as_float = value;
    type = Type::f32;
  }

  // UDE prefer user use protocol type as a bridge to wrap variable
  explicit Variable(BufferRef buf) {
    u.data = new Derived<BufferRef>(buf);
    type = Type::buffer;
    onHeap = true;
  }

  Variable(VectorRef vec, ude::Type type) {
    u.data = new Derived<VectorRef>(vec);
    this->type = type;
    onHeap = true;
  }

  explicit Variable(TensorRef tensor) {
    u.data = new Derived<TensorRef>(tensor);
    type = Type::tensor;
    onHeap = true;
  }

  explicit Variable(TupleRef tuple) {
    u.data = new Derived<TupleRef>(tuple);
    type = Type::tuple;
    onHeap = true;
  }

  explicit Variable(StringRef str) {
    u.data = new Derived<StringRef>(str);
    type = Type::str;
    onHeap = true;
  }

  // implicit constructor refers to clone constructor
  template <typename T,
            std::enable_if_t<all_type_of<is_different<T, Variable>, std::is_pointer<T>>::value, bool> = true>
  explicit Variable(T value) : type(ude::getType<Pointer>()), isProtocolv(false) {
    u.data = reinterpret_cast<Base *>(value); // Give a pointer, just keep it.
  }

  // Give a no protocol type, we just record it with operator new.
  template <typename T, std::enable_if_t<all_type_of<is_different<T, Variable>, is_not_pointer<T>>::value, bool> = true>
  explicit Variable(T &&value, ude::Type type) : type(type), onHeap(true), isProtocolv(false) {
    u.data = new Derived<std::remove_reference_t<std::remove_cv_t<T>>>(std::forward<T>(value));
  }

  ~Variable() {
    if (onHeap) {
      delete u.data;
    }
  }

  // Why non-copyable? Because of some types may be keep by operator new.
  // Copy maybe behave heavier.
  Variable(const Variable &) = delete; // non-copyable
  Variable(Variable &&rhs) noexcept {
    u = rhs.u;
    type = rhs.type;
    isProtocolv = rhs.isProtocolv;
    onHeap = rhs.onHeap;
    rhs.u.data = nullptr;
  }                                       // moveable
  Variable &operator=(Variable) = delete; // non-copyable
  Variable &operator=(Variable &&rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    u = rhs.u;
    type = rhs.type;
    isProtocolv = rhs.isProtocolv;
    onHeap = rhs.onHeap;
    rhs.u.data = nullptr;
    return *this;
  } // move assignment

  template <typename T> T &getRef() {
    if (!getIsPtr()) {
      if constexpr (is_arithmetic_v<intrinsic_t<T>>) {
        if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, bool>) {
          return u.as_bool;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int64_t>) {
          return u.as_int64;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int32_t>) {
          return u.as_int32;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int16_t>) {
          return u.as_int16;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int8_t>) {
          return u.as_int8;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint64_t>) {
          return u.as_uint64;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint32_t>) {
          return u.as_uint32;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint16_t>) {
          return u.as_uint16;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint8_t>) {
          return u.as_uint8;
        } else if UDE_CONSTEXPR_IF (is_same_v<float, intrinsic_t<T>>) {
          return u.as_float;
        } else if UDE_CONSTEXPR_IF (is_same_v<double, intrinsic_t<T>>) {
          return u.as_double;
        } else {
          assert(false && "Unknow basic type!");
        }
      } else {
        if (onHeap) {
          return reinterpret_cast<Derived<std::decay_t<T>> *>(u.data)->value;
        } else {
          assert(false && "This type should not come here!");
        }
      }
    }
    return *reinterpret_cast<std::decay_t<T> *>(u.data);
  }

  template <typename T> const T &getRef() const {
    if (!getIsPtr()) {
      if constexpr (is_arithmetic_v<intrinsic_t<T>>) {
        if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, bool>) {
          return u.as_bool;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int64_t>) {
          return u.as_int64;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int32_t>) {
          return u.as_int32;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int16_t>) {
          return u.as_int16;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, int8_t>) {
          return u.as_int8;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint64_t>) {
          return u.as_uint64;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint32_t>) {
          return u.as_uint32;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint16_t>) {
          return u.as_uint16;
        } else if UDE_CONSTEXPR_IF (is_same_v<intrinsic_t<T>, uint8_t>) {
          return u.as_uint8;
        } else if UDE_CONSTEXPR_IF (is_same_v<float, intrinsic_t<T>>) {
          return u.as_float;
        } else if UDE_CONSTEXPR_IF (is_same_v<double, intrinsic_t<T>>) {
          return u.as_double;
        } else {
          assert(false && "Unknow basic type!");
        }
      } else {
        if (onHeap) {
          return reinterpret_cast<Derived<std::decay_t<T>> *>(u.data)->value;
        } else {
          assert(false && "This type should not come here!");
        }
      }
    }
    return *reinterpret_cast<std::decay_t<T> *>(u.data);
  }

  UDE_NODISCARD bool getIsPtr() const { return type == ude::getType<Pointer>(); }

  UDE_NODISCARD Type getType() const { return type; }

  UDE_NODISCARD bool isProtocol() const { return isProtocolv; }

private:
  Type type = Type::invalid;
  bool onHeap = false;
  bool isProtocolv = true;
  union Payload {
    Payload() : as_int64(0) {}
    Base *data = nullptr;
    int64_t as_int64;
    int32_t as_int32;
    int16_t as_int16;
    int8_t as_int8;
    uint64_t as_uint64;
    uint32_t as_uint32;
    uint16_t as_uint16;
    uint8_t as_uint8;
    bool as_bool;
    double as_double;
    float as_float;
  } u;
};

} // namespace ude
