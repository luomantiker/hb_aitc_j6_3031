#pragma once

#include "ude/public/Caster.h"
#include "ude/public/Protocols.h"
#include "ude/public/Status.h"
#include "ude/public/Types.h"
#include "ude/public/Variable.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ude {

template <class VecType, class ValueType, Type id> struct VecConvert {
  Status load(Variable *var, size_t idx = 0) {
    if (var == nullptr || (var->getType() != Type::ptr && var->getType() != id)) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(id) + ", but got a type " + toString(var->getType());
      return Status::failure(true, errMsg.c_str());
    }

    if (var->isProtocol()) {
      auto vecRef = var->getRef<VectorRef>();

      auto vecSize = vecRef.size / sizeof(ValueType);
      value.resize(vecSize);
      memcpy(value.data(), vecRef.ptr, vecRef.size);
    } else {
      value = var->getRef<VecType>();
    }
    return Status::success();
  }

  Status sync(Variable *var) {
    if (var == nullptr || (var->getType() != Type::ptr && var->getType() != id)) {
      return Status::failure(true, "sync failed");
    }
    if (var->isProtocol()) {
      auto vecRef = var->getRef<VectorRef>();
      memcpy(vecRef.ptr, value.data(), vecRef.size);
    } else {
      var->getRef<VecType>() = value;
    }
    return Status::success();
  }

  UDE_TYPE_CASTER(VecType, id)
};

// C++17 can work
// C++14 Don't use std::vector which not specify in here.
template <class Vec, class Alloc>
struct TypeCaster<std::vector<Vec, Alloc>> : VecConvert<std::vector<Vec, Alloc>, Vec, getType<Vec>()> {};

template <class Alloc>
struct TypeCaster<std::vector<int32_t, Alloc>> : VecConvert<std::vector<int32_t, Alloc>, int32_t, Type::Int32Vec> {};

template <class Alloc>
struct TypeCaster<std::vector<int64_t, Alloc>> : VecConvert<std::vector<int64_t, Alloc>, int64_t, Type::Int64Vec> {};

template <class Alloc>
struct TypeCaster<std::vector<float, Alloc>> : VecConvert<std::vector<float, Alloc>, float, Type::F32Vec> {};

template <class Alloc>
struct TypeCaster<std::vector<double, Alloc>> : VecConvert<std::vector<double, Alloc>, double, Type::F64Vec> {};

template <> class TypeCaster<std::string> {
public:
  Status load(Variable *src, size_t idx = 0) {
    // now std::string is wrapped by copy at heap
    if (src == nullptr || Type::str != src->getType()) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(Type::str) + ", but got a type " +
                           toString(src->getType());
      return Status::failure(true, errMsg.c_str());
    }
    if (!src->isProtocol()) {
      value = src->getRef<std::string>();
    } else {
      auto strRef = src->getRef<StringRef>();
      value = std::string(strRef.data, strRef.len);
    }
    return Status::success();
  }

  // we believe that the string never change.
  Status sync(Variable *src) { return success(); }
  UDE_TYPE_CASTER(std::string, Type::str)
};

template <typename T, typename Alloc> VectorRef wrapStlVec(const std::vector<T, Alloc> &vec) {
  return {vec.data(), vec.size() * sizeof(T), getType<T>()};
}

inline StringRef wrapStdString(const std::string &str) { return StringRef(str); }

} // namespace ude
