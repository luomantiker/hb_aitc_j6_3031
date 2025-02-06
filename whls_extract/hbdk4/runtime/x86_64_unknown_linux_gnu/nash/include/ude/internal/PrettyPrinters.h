#pragma once

#include "ude/internal/Plan.h"
#include "ude/internal/Schema.h"
#include "ude/public/ArrayRef.h"
#include "ude/public/Kernel.h"
#include "ude/public/Protocols.h"
#include "ude/public/Variable.h"
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace ude {

template <typename T> std::ostream &operator<<(std::ostream &os, ArrayRef<T> Vec) {
  constexpr size_t maxDisplayElements = 10U;
  size_t i = 0;
  os << "[";
  for (const auto &e : Vec) {
    if (i++ > 0) {
      os << ", ";
    }
    os << e;
    if (i > maxDisplayElements) {
      break;
    }
  }
  if (i != Vec.size()) {
    os << ", ...";
  }
  os << "]";

  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ude::TensorRef &tf) {
  os << "Tensor<" << toString(tf.dtype);
  const auto &shape = tf.shape;
  if (!shape.empty()) {
    os << ",";
    for (auto i = 0U; i < shape.size() - 1; ++i) {
      os << shape[i] << "x";
    }
    os << shape.back();
  }
  const auto &stride = tf.stride;
  if (!stride.empty()) {
    os << ",";
    for (auto i = 0U; i < stride.size() - 1; ++i) {
      os << stride[i] << "x";
    }
    os << stride.back();
  }
  os << ", address: " << tf.ptr;
  os << ">";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ude::VectorRef &vf) {
  os << "Vector";
  if (vf.ptr != nullptr) {
    os << "{" << ude::toString(vf.dtype) << "}";
    if (vf.dtype == ude::Type::f32) {
      auto *begin = static_cast<float *>(vf.ptr);
      auto len = vf.size / sizeof(float);
      auto vec = ArrayRef<float>(begin, len);
      os << vec;
    } else if (vf.dtype == ude::Type::f64) {
      auto *begin = static_cast<double *>(vf.ptr);
      auto len = vf.size / sizeof(double);
      auto vec = ArrayRef<double>(begin, len);
      os << vec;
    } else if (vf.dtype == ude::Type::si32) {
      auto *begin = static_cast<int32_t *>(vf.ptr);
      auto len = vf.size / sizeof(int32_t);
      auto vec = ArrayRef<int32_t>(begin, len);
      os << vec;
    } else if (vf.dtype == ude::Type::si64) {
      auto *begin = static_cast<int64_t *>(vf.ptr);
      auto len = vf.size / sizeof(int64_t);
      auto vec = ArrayRef<int64_t>(begin, len);
      os << vec;
    } else {
      os << "OpaqueVec";
    }
  }

  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ude::Kernel &kernel) {
  os << "kernel's name is " << kernel.name << ", schema is " << ude::Schema(&kernel).getSchema() << ", DispatchKey is "
     << static_cast<size_t>(kernel.key);

  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ude::Variable *var) {
  if (var != nullptr) {
    auto iType = var->getType();
#define POPULATE(type, cppType)                                                                                        \
  if (Type::type == iType) {                                                                                           \
    os << var->getRef<cppType>();                                                                                      \
    return os;                                                                                                         \
  }
    POPULATE(bool8, bool)
    POPULATE(f32, float)
    POPULATE(f64, double)
    POPULATE(si8, int8_t)
    POPULATE(si16, int16_t)
    POPULATE(si32, int32_t)
    POPULATE(si64, int64_t)
    POPULATE(ui8, uint8_t)
    POPULATE(ui16, uint16_t)
    POPULATE(ui32, uint32_t)
    POPULATE(ui64, uint64_t)
    POPULATE(tensor, TensorRef)
    POPULATE(F32Vec, VectorRef)
    POPULATE(F64Vec, VectorRef)
    POPULATE(Int32Vec, VectorRef)
    POPULATE(Int64Vec, VectorRef)

    if (iType == ude::Type::TensorVec) {
      const auto &tfVec = var->getRef<std::vector<ude::Variable>>();
      os << "[";
      for (const auto &tf : tfVec) {
        const auto &iTf = tf.getRef<TensorRef>();
        os << iTf;
      }
      os << "]";
    }
  }
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const ude::Task &task) {
  os << "task is " << *task.touchKernel();
  os << ", args:(";
  auto var = task.touchVariable();
  size_t i = 0U;
  for (auto *ins : var) {
    if (i != 0U) {
      os << ", ";
    }
    os << ins;
    ++i;
  }
  os << ")";
  return os;
}

} // namespace ude
