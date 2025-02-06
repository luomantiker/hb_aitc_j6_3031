// ElementType and runByElementType

#pragma once
#include <cstdint>
#include <cstring>

#include "hbtl/Support/ErrorHandling.h"
#include "hbtl/Support/Half.h"

HBTL_NAMESPACE_BEGIN {

#define BYTE_SIZE_SHIFT 16U
#define SIGN_SHIFT 8U
#define SIGN_MASK ((1U << 2U) - 1U)
#define INDEX_MASK 255U

  enum ElementType : size_t {

#define ELEMENT_TYPE(NAME, BYTES, SIGN, INDEX) NAME = (BYTES << BYTE_SIZE_SHIFT) + (SIGN << SIGN_SHIFT) + INDEX

    ELEMENT_TYPE(si4x2, 1U, 1U, 1U), // b25bpu
    ELEMENT_TYPE(si8, 1U, 1U, 2U),   // bpu/arm/riscv/vpu
    ELEMENT_TYPE(si16, 2U, 1U, 3U),  // bpu/arm/riscv/vpu
    ELEMENT_TYPE(si32, 4U, 1U, 4U),  // bpu/arm/riscv/vpu
    ELEMENT_TYPE(si64, 8U, 1U, 5U),  // arm/riscv

    ELEMENT_TYPE(ui8, 1U, 2U, 1U),   // arm/riscv/vpu
    ELEMENT_TYPE(ui16, 2U, 2U, 2U),  // arm/riscv/vpu
    ELEMENT_TYPE(ui32, 4U, 2U, 3U),  // arm/riscv/vpu
    ELEMENT_TYPE(ui64, 8U, 2U, 4U),  // arm/riscv
    ELEMENT_TYPE(bool8, 1U, 2U, 5U), // bpu/arm

    ELEMENT_TYPE(cf8, 1U, 0U, 1U), // maybe in future
    ELEMENT_TYPE(f16, 2U, 0U, 2U), // arm/riscv
    ELEMENT_TYPE(f32, 4U, 0U, 3U), // arm/riscv
    ELEMENT_TYPE(f64, 8U, 0U, 4U), // arm/riscv

    ELEMENT_TYPE(vbf16, 2U, 0U, 5U), // vpu
    ELEMENT_TYPE(vf32, 4U, 0U, 6U),  // vpu

    ELEMENT_TYPE(tf32, 4U, 0U, 7U), // maybe in future

    ELEMENT_TYPE(opaque8, 1U, 3U, 200U),
    ELEMENT_TYPE(opaque16, 2U, 3U, 200U),
    ELEMENT_TYPE(opaque32, 4U, 3U, 200U),
    ELEMENT_TYPE(opaque64, 8U, 3U, 200U),
    ELEMENT_TYPE(opaque128, 16U, 3U, 200U),
    ELEMENT_TYPE(opaque256, 32U, 3U, 200U),
    ELEMENT_TYPE(opaque512, 64U, 3U, 200U),
    ELEMENT_TYPE(opaque1024, 128U, 3U, 200U),
    ELEMENT_TYPE(opaque2048, 256U, 3U, 200U),

    ELEMENT_TYPE(invalid, 65535U, 255U, 255U),
#undef ELEMENT_TYPE
  };

  constexpr inline int64_t getByteSize(ElementType type) { return static_cast<int64_t>(type >> BYTE_SIZE_SHIFT); }
  constexpr inline int64_t getBitWidth(ElementType type) { return static_cast<int64_t>(type >> BYTE_SIZE_SHIFT) * 8; }

  /// get signedness of given type. return 0 means floating-point, 1 means signed integer, 2 means unsigned integer, 3
  /// means opaque type
  constexpr inline int64_t getSignedness(ElementType type) {
    return static_cast<int64_t>((type >> SIGN_SHIFT) & SIGN_MASK);
  }
  constexpr inline bool isIntegral(ElementType type) {
    return hbtl::getSignedness(type) == 1 || hbtl::getSignedness(type) == 2;
  }
  constexpr inline bool isFloat(ElementType type) { return hbtl::getSignedness(type) == 0; }
  constexpr inline bool isOpaque(ElementType type) { return hbtl::getSignedness(type) == 3; }

  constexpr inline int64_t getIndex(ElementType type) { return static_cast<int64_t>(type & INDEX_MASK); }
  constexpr inline bool isF16(ElementType type) { return hbtl::getSignedness(type) == 0 && hbtl::getIndex(type) == 2; }
  constexpr inline bool isBool(ElementType type) { return hbtl::getSignedness(type) == 2 && hbtl::getIndex(type) == 5; }

#undef BYTE_SIZE_SHIFT
#undef SIGN_SHIFT
#undef SIGN_MASK

  constexpr inline const char *toString(ElementType type) {
    switch (type) {

#define ELEMENT_TYPE(NAME)                                                                                             \
  case (NAME):                                                                                                         \
    return #NAME;

      ELEMENT_TYPE(si4x2)
      ELEMENT_TYPE(si8)
      ELEMENT_TYPE(si16)
      ELEMENT_TYPE(si32)
      ELEMENT_TYPE(si64)

      ELEMENT_TYPE(ui8)
      ELEMENT_TYPE(ui16)
      ELEMENT_TYPE(ui32)
      ELEMENT_TYPE(ui64)
      ELEMENT_TYPE(bool8)

      ELEMENT_TYPE(cf8)
      ELEMENT_TYPE(f16)
      ELEMENT_TYPE(f32)
      ELEMENT_TYPE(f64)

      ELEMENT_TYPE(vbf16)
      ELEMENT_TYPE(vf32)

      ELEMENT_TYPE(tf32)

      ELEMENT_TYPE(opaque8)
      ELEMENT_TYPE(opaque16)
      ELEMENT_TYPE(opaque32)
      ELEMENT_TYPE(opaque64)
      ELEMENT_TYPE(opaque128)
      ELEMENT_TYPE(opaque256)
      ELEMENT_TYPE(opaque512)
      ELEMENT_TYPE(opaque1024)
      ELEMENT_TYPE(opaque2048)

      ELEMENT_TYPE(invalid)
#undef ELEMENT_TYPE
    }
    return "unreachable";
  }

  inline std::ostream &operator<<(std::ostream &os, const ElementType &type) {
    return os << std::string(toString(type));
  }

  inline ElementType fromString(std::string & typeStr) {
    if (typeStr == "si4x2") {
      return ElementType::si4x2;
    } else if (typeStr == "si8") {
      return ElementType::si8;
    } else if (typeStr == "si16") {
      return ElementType::si16;
    } else if (typeStr == "si32") {
      return ElementType::si32;
    } else if (typeStr == "si64") {
      return ElementType::si64;
    } else if (typeStr == "ui8") {
      return ElementType::ui8;
    } else if (typeStr == "ui16") {
      return ElementType::ui16;
    } else if (typeStr == "ui32") {
      return ElementType::ui32;
    } else if (typeStr == "ui64") {
      return ElementType::ui64;
    } else if (typeStr == "bool8") {
      return ElementType::bool8;
    } else if (typeStr == "cf8") {
      return ElementType::cf8;
    } else if (typeStr == "f16") {
      return ElementType::f16;
    } else if (typeStr == "f32") {
      return ElementType::f32;
    } else if (typeStr == "f64") {
      return ElementType::f64;
    } else if (typeStr == "vbf16") {
      return ElementType::vbf16;
    } else if (typeStr == "vf32") {
      return ElementType::vf32;
    } else if (typeStr == "tf32") {
      return ElementType::tf32;
    }

    return ElementType::invalid;
  }

  template <size_t bytes> struct OpaqueType {
    std::array<unsigned char, bytes> data;
  };

  template <size_t bytes> bool operator==(const OpaqueType<bytes> &lhs, const OpaqueType<bytes> &rhs) {
    return lhs.data == rhs.data;
  }
  template <size_t bytes> bool operator!=(const OpaqueType<bytes> &lhs, const OpaqueType<bytes> &rhs) {
    return lhs.data != rhs.data;
  }

  using opaque8_t = OpaqueType<1U>;
  using opaque16_t = OpaqueType<2U>;
  using opaque32_t = OpaqueType<4U>;
  using opaque64_t = OpaqueType<8U>;
  using opaque128_t = OpaqueType<16U>;
  using opaque256_t = OpaqueType<32U>;
  using opaque512_t = OpaqueType<64U>;
  using opaque1024_t = OpaqueType<128U>;
  using opaque2048_t = OpaqueType<256U>;

  constexpr inline ElementType getOpaqueType(int64_t byteSize) {
    switch (byteSize) {
    case 1U:
      return ElementType::opaque8;
    case 2U:
      return ElementType::opaque16;
    case 4U:
      return ElementType::opaque32;
    case 8U:
      return ElementType::opaque64;
    case 16U:
      return ElementType::opaque128;
    case 32U:
      return ElementType::opaque256;
    case 64U:
      return ElementType::opaque512;
    case 128U:
      return ElementType::opaque1024;
    case 256U:
      return ElementType::opaque2048;
    default:
      return ElementType::invalid;
    }
  }

  constexpr inline ElementType getOpaqueType(ElementType type) { return getOpaqueType(getByteSize(type)); }

  using invalid_t = std::nullptr_t; // support invalid type for invalid tensor

  template <typename T> constexpr inline ElementType getElementType() {
    using U = typename std::remove_cv<T>::type;

#define ELEMENT_TYPE(TYPE, ELEMENT_TYPE)                                                                               \
  if (std::is_same<U, TYPE>::value) {                                                                                  \
    return ELEMENT_TYPE;                                                                                               \
  }
    ELEMENT_TYPE(int8_t, ElementType::si8)
    ELEMENT_TYPE(int16_t, ElementType::si16)
    ELEMENT_TYPE(int32_t, ElementType::si32)
    ELEMENT_TYPE(int64_t, ElementType::si64)

    ELEMENT_TYPE(uint8_t, ElementType::ui8)
    ELEMENT_TYPE(uint16_t, ElementType::ui16)
    ELEMENT_TYPE(uint32_t, ElementType::ui32)
    ELEMENT_TYPE(uint64_t, ElementType::ui64)
    ELEMENT_TYPE(bool, ElementType::bool8)

    ELEMENT_TYPE(half_t, ElementType::f16)
    ELEMENT_TYPE(float, ElementType::f32)
    ELEMENT_TYPE(double, ElementType::f64)

    ELEMENT_TYPE(char, ElementType::opaque8)
    ELEMENT_TYPE(opaque8_t, ElementType::opaque8)
    ELEMENT_TYPE(opaque16_t, ElementType::opaque16)
    ELEMENT_TYPE(opaque32_t, ElementType::opaque32)
    ELEMENT_TYPE(opaque64_t, ElementType::opaque64)
    ELEMENT_TYPE(opaque128_t, ElementType::opaque128)
    ELEMENT_TYPE(opaque256_t, ElementType::opaque256)
    ELEMENT_TYPE(opaque512_t, ElementType::opaque512)
    ELEMENT_TYPE(opaque1024_t, ElementType::opaque1024)
    ELEMENT_TYPE(opaque2048_t, ElementType::opaque2048)

    ELEMENT_TYPE(invalid_t, ElementType::invalid)

#undef ELEMENT_TYPE
    // FIXME when f16, bf16, tf32, cf8, si4x2 are implemented.
    return ElementType::invalid;
  }

  template <typename Func> void runByOpaqueType(int64_t byteSize, const Func &func) {

#define TRY_RUN_BY_OPAQUE_TYPE(opaque)                                                                                 \
  if (byteSize == static_cast<int64_t>(sizeof(opaque))) {                                                              \
    opaque tv = {};                                                                                                    \
    (void)func(tv);                                                                                                    \
    return;                                                                                                            \
  }

    TRY_RUN_BY_OPAQUE_TYPE(opaque8_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque16_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque32_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque64_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque128_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque256_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque512_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque1024_t)
    TRY_RUN_BY_OPAQUE_TYPE(opaque2048_t)

#undef TRY_RUN_BY_OPAQUE_TYPE
    hbtl_trap(("no opaque type for byte size {}" + std::to_string(byteSize)).c_str());
  }

  /// forked from hbut/Support/type_traits.h
  /// Determine whether any same types exist among all types
  template <typename T0, typename T1, typename... Args>
  struct any_same_type
      : public std::integral_constant<bool, any_same_type<T0, T1>::value || any_same_type<T0, Args...>::value ||
                                                any_same_type<T1, Args...>::value> {};

  /// Determine whether any same types exist among all types
  template <typename T0, typename T1>
  struct any_same_type<T0, T1> : public std::integral_constant<bool, std::is_same<T0, T1>::value> {};

  /// Determine whether any same types exist among all types
  template <typename T0, typename T1, typename... Args>
  constexpr static bool any_same_type_v = any_same_type<T0, T1, Args...>::value;

  template <typename Type> struct TypeWrapper {
    using type = Type;

    static constexpr bool valid = !std::is_same<Type, invalid_t>::value;

    template <typename AltType> using type_or_default = typename std::conditional_t<valid, Type, AltType>;
  };

  template <typename T, typename Func> bool tryRunByElementType(ElementType type, const Func &func) {
    static_assert(!hbtl::any_same_type_v<T, Func>, "Duplicate type is not allowed");
    if (type == getElementType<T>()) {
      (void)func(TypeWrapper<T>());
      return true;
    }
    return false;
  }

  template <typename T0, typename Func> void runByElementType(ElementType type, const Func &func) {
    static_assert(!hbtl::any_same_type_v<T0, Func>, "Duplicate type is not allowed");
    bool result = tryRunByElementType<T0>(type, func);
    if (!result) {
      hbtl_trap(("unknown element type: " + std::string(toString(type))).c_str());
    }
  }

#define TRY_RUN_BY_ELEMENT_TYPE_MACRO(...)                                                                             \
  bool tryRunByElementType(ElementType type, const Func &func) {                                                       \
    static_assert(!hbtl::any_same_type_v<T0, __VA_ARGS__>, "Duplicate type is not allowed");                           \
    if (tryRunByElementType<T0>(type, func)) {                                                                         \
      return true;                                                                                                     \
    } else {                                                                                                           \
      return tryRunByElementType<__VA_ARGS__>(type, func);                                                             \
    }                                                                                                                  \
  }

#define RUN_BY_ELEMENT_TYPE_MACRO(...)                                                                                 \
  void runByElementType(ElementType type, const Func &func) {                                                          \
    static_assert(!hbtl::any_same_type_v<T0, __VA_ARGS__>, "Duplicate type is not allowed");                           \
    bool result = tryRunByElementType<T0, __VA_ARGS__>(type, func);                                                    \
    if (!result) {                                                                                                     \
      hbtl_trap(("unknown element type: " + std::string(toString(type))).c_str());                                     \
    }                                                                                                                  \
  }

  template <typename T0, typename T1, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1)
  template <typename T0, typename T1, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1)

  template <typename T0, typename T1, typename T2, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2)
  template <typename T0, typename T1, typename T2, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2)

  template <typename T0, typename T1, typename T2, typename T3, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3)
  template <typename T0, typename T1, typename T2, typename T3, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename T11, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename T11, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)

  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename T11, typename T12, typename Func>
  TRY_RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)
  template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
            typename T8, typename T9, typename T10, typename T11, typename T12, typename Func>
  RUN_BY_ELEMENT_TYPE_MACRO(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)

#undef TRY_RUN_BY_ELEMENT_TYPE_MACRO
#undef RUN_BY_ELEMENT_TYPE_MACRO
}
HBTL_NAMESPACE_END
