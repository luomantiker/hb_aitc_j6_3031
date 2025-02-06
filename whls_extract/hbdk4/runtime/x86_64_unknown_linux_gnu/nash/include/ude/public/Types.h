#pragma once

#include "Common.h"
#include "Compiler.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ude {

struct BufferRef;
struct TensorRef;
struct VectorRef;
struct TupleRef;
struct Pointer;
struct Heap;
struct StringRef;

#define SEMANTIC_SHIFT 24U
#define SEMANTIC_MASK ((1U << 8U) - 1U)

#define BYTESIZE_SHIFT 16U
#define BYTESIZE_MASK ((1U << 8U) - 1U)

#define SIGN_SHIFT 8U
#define SIGN_MASK ((1U << 8U) - 1U)

enum Type : size_t {

#define ELEMENT_TYPE(NAME, SEMANTIC, BYTES, SIGN, INDEX)                                                               \
  NAME = (SEMANTIC << SEMANTIC_SHIFT) + (BYTES << BYTESIZE_SHIFT) + (SIGN << SIGN_SHIFT) + INDEX

  ELEMENT_TYPE(si4x2, 0U, 1U, 1U, 1U),
  ELEMENT_TYPE(si8, 0U, 1U, 1U, 2U),
  ELEMENT_TYPE(si16, 0U, 2U, 1U, 3U),
  ELEMENT_TYPE(si32, 0U, 4U, 1U, 4U),
  ELEMENT_TYPE(si64, 0U, 8U, 1U, 5U),

  ELEMENT_TYPE(ui8, 0U, 1U, 2U, 1U),
  ELEMENT_TYPE(ui16, 0U, 2U, 2U, 2U),
  ELEMENT_TYPE(ui32, 0U, 4U, 2U, 3U),
  ELEMENT_TYPE(ui64, 0U, 8U, 2U, 4U),

  ELEMENT_TYPE(bool8, 0U, 1U, 2U, 5U),

  ELEMENT_TYPE(f16, 0U, 2U, 0U, 2U),
  ELEMENT_TYPE(f32, 0U, 4U, 0U, 3U),
  ELEMENT_TYPE(f64, 0U, 8U, 0U, 4U),

  ELEMENT_TYPE(Int32Vec, 3U, 4U, 1U, 4U),
  ELEMENT_TYPE(Int64Vec, 3U, 8U, 1U, 5U),
  ELEMENT_TYPE(F32Vec, 3U, 4U, 0U, 3U),
  ELEMENT_TYPE(F64Vec, 3U, 8U, 0U, 4U),
  ELEMENT_TYPE(TensorVec, 3U, 0U, 4U, 0U),

  ELEMENT_TYPE(tensor, 1U, 0U, 4U, 0U),
  ELEMENT_TYPE(vector, 1U, 0U, 4U, 1U),
  ELEMENT_TYPE(buffer, 1U, 0U, 4U, 2U),
  ELEMENT_TYPE(tuple, 1U, 0U, 4U, 3U),
  ELEMENT_TYPE(str, 1U, 0U, 4U, 4U),

  ELEMENT_TYPE(ptr, 2U, 1U, 1U, 1U),

  ELEMENT_TYPE(invalid, 255U, 65535U, 255U, 255U),
#undef ELEMENT_TYPE
};

using invalid_t = std::nullptr_t; // support invalid type for invalid tensor

inline const char *toString(ude::Type type) {
  switch (type) {
#define CaseOfKind(kind, strName)                                                                                      \
  case Type::kind:                                                                                                     \
    return strName
    CaseOfKind(si4x2, "si4x2");
    CaseOfKind(si8, "int8_t");
    CaseOfKind(si16, "int16_t");
    CaseOfKind(si32, "int32_t");
    CaseOfKind(si64, "int64_t");
    CaseOfKind(ui8, "uint8_t");
    CaseOfKind(ui16, "uint16_t");
    CaseOfKind(ui32, "uint32_t");
    CaseOfKind(ui64, "uint64_t");
    CaseOfKind(bool8, "bool");
    CaseOfKind(f16, "half");
    CaseOfKind(f32, "float");
    CaseOfKind(f64, "double");
    CaseOfKind(tensor, "Tensor");
    CaseOfKind(vector, "Vector");
    CaseOfKind(str, "Str");
  case Type::Int64Vec:
    return "int64_t[]";
  case Type::F32Vec:
    return "float[]";
  case Type::F64Vec:
    return "double[]";
  case Type::TensorVec:
    return "Tensor[]";
  default:
    return "Unknown";
  }
}

template <typename T> constexpr inline Type getType() {
  using U = intrinsic_t<T>;

#define GET_TYPE(TYPE, ENUM)                                                                                           \
  if UDE_CONSTEXPR_IF (is_same_v<U, TYPE>) {                                                                           \
    return ENUM;                                                                                                       \
  }
  GET_TYPE(int8_t, Type::si8)
  GET_TYPE(int16_t, Type::si16)
  GET_TYPE(int32_t, Type::si32)
  GET_TYPE(int64_t, Type::si64)

  GET_TYPE(uint8_t, Type::ui8)
  GET_TYPE(uint16_t, Type::ui16)
  GET_TYPE(uint32_t, Type::ui32)
  GET_TYPE(uint64_t, Type::ui64)
  GET_TYPE(bool, Type::bool8)
  GET_TYPE(std::vector<int64_t>, Type::Int64Vec)
  GET_TYPE(std::vector<float>, Type::F32Vec)
  GET_TYPE(std::vector<float>, Type::F64Vec)

  GET_TYPE(float, Type::f32)
  GET_TYPE(double, Type::f64)

  GET_TYPE(BufferRef, Type::buffer)
  GET_TYPE(TensorRef, Type::tensor)
  GET_TYPE(VectorRef, Type::vector)
  GET_TYPE(TupleRef, Type::tuple)
  GET_TYPE(StringRef, Type::str)

  GET_TYPE(Pointer, Type::ptr)

#undef GET_TYPE
  return Type::invalid;
}

enum Semantic : size_t {
  Elemental = 0,
  Protocol = 1,
  Ptr = 2,
  Vector = 3,
  Unknown = 4,
};

enum Signedness : size_t {
  Float = 0U,
  Signed = 1U,
  Unsigned = 2U,
  Opaque = 3U,
  Invalid = 4U,
};

constexpr inline size_t getSemantic(Type type) {
  return static_cast<Semantic>((type >> SEMANTIC_SHIFT) & SEMANTIC_MASK);
}

constexpr inline bool isPtr(Type type) { return ude::getSemantic(type) == Semantic::Ptr; }

constexpr inline bool isProtocols(Type type) { return ude::getSemantic(type) == Semantic::Protocol; }

constexpr inline bool isVector(Type type) { return ude::getSemantic(type) == Semantic::Vector; }

constexpr inline bool isElemental(Type type) { return ude::getSemantic(type) == Semantic::Elemental; }

constexpr inline size_t getByteSize(Type type) {
  assert(isElemental(type));
  return (type >> BYTESIZE_SHIFT) & BYTESIZE_MASK;
}
constexpr inline size_t getBitWidth(Type type) { return ude::getByteSize(type) * 8U; }

constexpr inline Signedness getSignedness(Type type) {
  assert(isElemental(type));
  return static_cast<Signedness>((type >> SIGN_SHIFT) & SIGN_MASK);
}

constexpr inline bool isIntegral(Type type) {
  return ude::getSignedness(type) == Signedness::Signed || ude::getSignedness(type) == Signedness::Unsigned;
}
constexpr inline bool isFloat(Type type) { return ude::getSignedness(type) == Signedness::Float; }
constexpr inline bool isOpaque(Type type) { return ude::getSignedness(type) == Signedness::Opaque; }

#undef SEMANTIC_MASK
#undef SEMANTIC_SHIFT

#undef BYTESIZE_SHIFT
#undef BYTESIZE_MASK

#undef SIGN_SHIFT
#undef SIGN_MASK

} // namespace ude
