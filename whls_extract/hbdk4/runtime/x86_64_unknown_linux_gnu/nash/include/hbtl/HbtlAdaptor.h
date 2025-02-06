#pragma once

#include "hbtl/Core/ElementType.h"
#include "hbtl/Core/Storage.h"
#include "hbtl/Core/Tensor.h"
#include "hbtl/StlAdaptor.h"
#include "hbtl/Tensor.h"
#include "ude/public/Caster.h"
#include "ude/public/Protocols.h"
#include "ude/public/Status.h"
#include "ude/public/Types.h"
#include "ude/public/Variable.h"
#include <cstddef>
#include <cstdint>

namespace ude {

template <> class TypeCaster<hbtl::Tensor> {
private:
  hbtl::ElementType typeFromUdeToHbtl(ude::Type type) {
    return static_cast<hbtl::ElementType>(static_cast<size_t>(type));
  }

public:
  Status load(Variable *src, size_t idx = 0) {
    if (src == nullptr || (src->getType() != Type::tensor && src->getType() != Type::ptr)) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(Type::str) + ", but got a type " +
                           (src == nullptr ? "invalid" : toString(src->getType()));
      return Status::failure(true, errMsg.c_str());
    }

    if (src->isProtocol()) {
      const auto &tensorRef = src->getRef<TensorRef>();

      if (tensorRef.ptr != nullptr) {
        value = hbtl::Tensor::createExternal(tensorRef.shape, tensorRef.stride,
                                             static_cast<hbtl::ElementType>(tensorRef.dtype),
                                             hbtl::ArrayRef<char>{static_cast<char *>(tensorRef.ptr), tensorRef.size});
      } else {
        value.setType(static_cast<hbtl::ElementType>(tensorRef.dtype));
        value.getRefRank() = tensorRef.rank;
        value.setShape(tensorRef.shape);
      }
    } else {
      value = src->getRef<hbtl::Tensor>();
    }
    return Status::success();
  }

  Status sync(Variable *to) {
    if (to == nullptr || (to->getType() != Type::tensor && to->getType() != Type::ptr)) {
      return failure();
    }
    if (to->getType() == Type::ptr) {
      to->getRef<hbtl::Tensor>() = value;
      return success();
    }
    auto &tensorRef = to->getRef<TensorRef>();
    tensorRef.rank = value.getRank();
    tensorRef.dtype = static_cast<ude::Type>(value.getType());
    tensorRef.shape = value.getSizes();
    tensorRef.stride = value.getStrides();
    return success();
  }

  UDE_TYPE_CASTER(hbtl::Tensor, Type::tensor)
};

template <> struct TypeCaster<std::vector<hbtl::Tensor>> {

  Status load(Variable *var, size_t idx = 0) {
    // now we use heap to wrap vector<TensorRef>
    if (var == nullptr || (var->getType() != Type::TensorVec)) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(Type::str) + ", but got a type " +
                           toString(var->getType());
      return Status::failure(true, errMsg.c_str());
    }
    TypeCaster<hbtl::Tensor> caster;
    auto &wrappedVar = var->getRef<std::vector<Variable>>();

    for (auto &inst : wrappedVar) {
      auto r = caster.load(&inst);
      if (r.failed()) {
        std::string errMsg = "you should use vector<Variable> to wrap vector<Tensor>.";
      }
      value.emplace_back(caster);
    }

    return Status::success();
  }

  Status sync(Variable *to) { return success(); }
  UDE_TYPE_CASTER(std::vector<hbtl::Tensor>, Type::TensorVec)
};

template <> struct TypeCaster<hbtl::ElementType> {
  Status load(Variable *var, size_t idx = 0) {
    // use unsigned int64 to hold HbtlElementType
    if (var == nullptr || (var->getType() != Type::ui64)) {
      std::string errMsg = "The " + std::to_string(idx) + "-th parameter fails to convert. " +
                           "Kernel expect type is " + toString(Type::str) + ", but got a type " +
                           toString(var->getType());
      return Status::failure(true, errMsg.c_str());
    }

    value = static_cast<hbtl::ElementType>(var->getRef<uint64_t>());
    return Status::success();
  }

  Status sync(Variable *to) { return success(); }
  UDE_TYPE_CASTER(hbtl::ElementType, Type::ui64)
};

inline ude::Type cvtType(hbtl::ElementType type) { return static_cast<ude::Type>(type); }

inline ude::TensorRef wrapTensor(const hbtl::Tensor &tensor, bool readonly = false) {
  auto tensorInfo = tensor.getBytes();
  const auto *storage = tensor.getStorage();
  bool isCuda = storage != nullptr && storage->isCuda();
  return {tensorInfo.data(),
          tensorInfo.size(),
          cvtType(tensor.getType()),
          tensor.getRank(),
          {tensor.getSizes().data(), tensor.getSizes().size()},
          {tensor.getStrides().data(), tensor.getStrides().size()},
          readonly,
          isCuda ? "cuda" : "cpu",
          isCuda ? storage->cudaDeviceId() : 0};
}

} // namespace ude
