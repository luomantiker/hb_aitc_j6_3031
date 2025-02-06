// HBTL Shape checking

#pragma once
#include "hbtl/Core/ElementType.h"
#ifndef HBTL_SUPPORT_VERIFY_H_
#define HBTL_SUPPORT_VERIFY_H_

#include "hbtl/ADT/Optional.h"
#include "hbtl/Support/ADTExtras.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  bool mustValid(const T &t, const TensorType &...rs) {
    if (t.getType() == ElementType::invalid) {
      return false;
    }
    return mustValid(rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>>
  bool mustValid(const T &t) {
    return static_cast<bool>(t.getType() != ElementType::invalid);
  }

  /// check tensors that have specified
  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyRank(int64_t rank, const T &t, const TensorType &...rs) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (rank != -1) {
        if (rank != t.getRank()) {
          return hbtl::nullopt;
        }
      } else {
        rank = t.getRank();
      }
    }
    return unifyRank(rank, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyRank(int64_t rank, const T &t) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (rank != -1) {
        if (rank != t.getRank()) {
          return hbtl::nullopt;
        }
      } else {
        rank = t.getRank();
      }
    }
    return rank;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyRank(const T &t, const TensorType &...rs) {
    return unifyRank(-1, t, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyAxis(int64_t dim, int64_t size, const T &t, const TensorType &...rs) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (size != -1) {
        if (t.getSize(dim) != size) {
          return hbtl::nullopt;
        }
      } else {
        size = t.getSize(dim);
      }
    }
    return unifyAxis(dim, size, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyAxis(int64_t dim, int64_t size, const T &t) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (size != -1) {
        if (t.getSize(dim) != size) {
          return hbtl::nullopt;
        }
      } else {
        size = t.getSize(dim);
      }
    }
    return size;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<int64_t> unifyAxis(int64_t dim, const T &t, const TensorType &...rs) {
    return unifyAxis(dim, -1, t, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  hbtl::optional<ElementType> unifyElementType(ElementType type, const T &t, const TensorType &...rs) {
    if (t.getType() != ElementType::invalid) {
      if (type != ElementType::invalid) {
        if (t.getType() != type) {
          return hbtl::nullopt;
        }
      } else {
        type = t.getType();
      }
    }
    return unifyElementType(type, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>>
  hbtl::optional<ElementType> unifyElementType(ElementType type, const T &t) {
    if (t.getType() != ElementType::invalid) {
      if (type != ElementType::invalid) {
        if (t.getType() != type) {
          return hbtl::nullopt;
        }
      } else {
        type = t.getType();
      }
    }
    return type;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<ElementType> unifyElementType(const T &t, const TensorType &...rs) {
    return unifyElementType(ElementType::invalid, t, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<std::vector<int64_t>> unifyShape(std::vector<int64_t> shape, const T &t,
                                                                    const TensorType &...rs) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (!shape.empty()) {
        if (shape != t.getSizes()) {
          return hbtl::nullopt;
        }
      } else {
        shape = t.getSizes();
      }
    }
    return unifyShape(shape, rs...);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>>
  HBTL_MAYBE_UNUSED hbtl::optional<std::vector<int64_t>> unifyShape(std::vector<int64_t> shape, const T &t) {
    if (t.getType() != ElementType::invalid && t.getRank() > 0) {
      if (!shape.empty()) {
        if (shape != t.getSizes()) {
          return hbtl::nullopt;
        }
      } else {
        shape = t.getSizes();
      }
    }
    return shape;
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same<T, Tensor>::value || std::is_same<T, TensorType>::value>,
            typename... TensorType>
  HBTL_MAYBE_UNUSED hbtl::optional<std::vector<int64_t>> unifyShape(const T &t, const TensorType &...rs) {
    return unifyShape(std::vector<int64_t>{}, t, rs...);
  }
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_VERIFY_H_
