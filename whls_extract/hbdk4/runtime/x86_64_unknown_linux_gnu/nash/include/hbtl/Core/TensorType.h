// HBTL TensorType

#pragma once
#ifndef HBTL_SUPPORT_TENSORTYPE_H_
#define HBTL_SUPPORT_TENSORTYPE_H_

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Core/ElementType.h"
#include "hbtl/Support/ADTExtras.h" // IWYU pragma: keep
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h" // IWYU pragma: keep
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <utility> // IWYU pragma: keep
#include <vector>

HBTL_NAMESPACE_BEGIN {
  constexpr size_t typeAxisLimit = 11U;
  class HBTL_EXPORTED TensorType {
  public:
    /// allow empty tensorType
    TensorType() = default;
    /// return false when construct from nothing
    explicit operator bool() const { return type != ElementType::invalid; }

    HBTL_NODISCARD bool valid() const { return this->operator bool(); }

    TensorType(const TensorType &) = default;
    TensorType(TensorType &) = default;

    TensorType &operator=(const TensorType &) = default;

    /// default constructor
    TensorType(ArrayRef<int64_t> sizes, ElementType type) : type(type) {
      assert(sizes.size() <= typeAxisLimit && "tensorType's max rank is 11");
      rank = sizes.size();
      for (auto i = 0U; i < rank; ++i) {
        this->sizes[i] = sizes[i];
      }
    }

    bool operator==(const TensorType &other) const {
      if (other.getSizes() != this->getSizes()) {
        return false;
      }
      if (other.getType() != this->getType()) {
        return false;
      }
      return true;
    }

    bool operator!=(const TensorType &other) const { return !((*this) == other); }

    HBTL_NODISCARD ElementType getType() const { return type; }

    // get footprint members
    HBTL_NODISCARD int64_t getRank() const { return static_cast<int64_t>(rank); }
    HBTL_NODISCARD int64_t getSize(int64_t dim) const { return sizes[normDim(dim)]; }
    HBTL_NODISCARD ArrayRef<int64_t> getSizes() const { return {sizes.begin(), rank}; }
    HBTL_NODISCARD MutableArrayRef<int64_t> getMutSizes() { return {sizes.begin(), rank}; }
    HBTL_NODISCARD std::vector<int64_t> getSizesCopy() const { return getSizes().vec(); }

    HBTL_NODISCARD inline int64_t normDim(int64_t dim) const {
      dim += ((dim < 0) ? getRank() : 0);
      assert(dim >= 0 && (getRank() == 0 || dim < getRank()) && "invalid dim");
      return dim;
    }

    HBTL_NODISCARD inline std::vector<int64_t> normDim(ArrayRef<int64_t> dims) const {
      auto ret = std::vector<int64_t>(dims.size());
      for (auto i = 0U; i < dims.size(); ++i) {
        ret[i] = normDim(dims[i]);
      }
      return ret;
    }

  private:
    size_t rank = 0;
    std::array<int64_t, typeAxisLimit> sizes = {};
    ElementType type = ElementType::invalid; /// element type of data
  };

  HBTL_EXPORTED std::ostream &operator<<(std::ostream &os, const TensorType &tensor);
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_TENSORTYPE_H_
