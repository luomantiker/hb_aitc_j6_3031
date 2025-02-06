// HBTL SmallTensorRef

#pragma once
#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Support/Compiler.h"
#include <cstddef>
#include <utility>

HBTL_NAMESPACE_BEGIN {

  template <typename T, size_t rank, bool isC = true> class TensorRef {

  public:
    TensorRef() = default;
    TensorRef(TensorRef &) = default;
    TensorRef(TensorRef &&) noexcept = default;
    TensorRef &operator=(TensorRef &&) = default;
    TensorRef(ArrayRef<T> data, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides)
        : data(data), sizes(sizes), strides(strides) {
      assert(sizes.size() == rank && "TensorRef rank and sizes mismatch");
      if HBTL_CONSTEXPR_IF (isC) {
        assert((sizes.rbegin()[0] == 1 || strides.rbegin()[0] == 0 || strides.rbegin()[0] == sizeof(T)) &&
               "TensorRef need the origin tensor contiguous in last dim");
      }
    }

    T operator[](ArrayRef<int64_t> coord) const {
      assert(coord.size() == rank);
      return data[getPos(coord)];
    }

    HBTL_NODISCARD size_t getPos(ArrayRef<int64_t> coord) const {
      assert(coord.size() == rank);
      size_t pos = 0;

      if HBTL_CONSTEXPR_IF (isC) {
        for (auto i = 0U; i < rank - 1; ++i) {
          pos += coord[i] * strides[i];
        }
        pos /= sizeof(T);
        pos += coord[rank - 1];
      } else {
        for (auto i = 0U; i < rank; ++i) {
          pos += coord[i] * strides[i];
        }
        pos /= sizeof(T);
      }

      return pos;
    }

    HBTL_NODISCARD inline int64_t getSize(int64_t dim) const { return sizes[dim]; }
    HBTL_NODISCARD inline ArrayRef<int64_t> getSizes() const { return sizes; }
    HBTL_NODISCARD inline constexpr int64_t getRank() const { return rank; }

    template <size_t subRank> TensorRef<T, subRank, isC> select(ArrayRef<int64_t> coord) const {
      assert(coord.size() + subRank == rank);
      int64_t elemOffset = 0;
      constexpr auto reduceRank = rank - subRank;
      for (auto i = 0U; i < reduceRank; ++i) {
        elemOffset += coord[i] * strides[i];
      }
      elemOffset /= sizeof(T);
      return TensorRef<T, subRank, isC>(data.drop_front(elemOffset), sizes.drop_front(coord.size()),
                                        strides.drop_front(coord.size()));
    }

    const T *getData() { return data.data(); }

    HBTL_NODISCARD size_t getLimit() const { return data.size(); }

  private:
    ArrayRef<T> data;
    ArrayRef<int64_t> sizes;
    ArrayRef<int64_t> strides;
  };

  template <typename T, size_t rank, bool isC = true> class MutableTensorRef : public TensorRef<T, rank, isC> {
  public:
    MutableTensorRef(MutableArrayRef<T> data, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides)
        : TensorRef<T, rank, isC>(data, sizes, strides) {}
    explicit MutableTensorRef(TensorRef<T, rank, isC> tensorRef) : TensorRef<T, rank, isC>(tensorRef) {}

    HBTL_NODISCARD int64_t getSize(int64_t dim) const { return TensorRef<T, rank, isC>::getSize(dim); }
    HBTL_NODISCARD ArrayRef<int64_t> getSizes() const { return TensorRef<T, rank, isC>::getSizes(); }
    HBTL_NODISCARD constexpr int64_t getRank() const { return TensorRef<T, rank, isC>::getRank(); }

    T *getData() { return const_cast<T *>(TensorRef<T, rank, isC>::getData()); }

    T operator[](ArrayRef<int64_t> coord) const { return TensorRef<T, rank, isC>::operator[](coord); }

    T &operator[](ArrayRef<int64_t> coord) { return getData()[TensorRef<T, rank, isC>::getPos(coord)]; }

    HBTL_NODISCARD size_t getLimit() const { return TensorRef<T, rank, isC>::getLimit(); }

    template <size_t subRank> MutableTensorRef<T, subRank, isC> select(ArrayRef<int64_t> coord) {
      return MutableTensorRef<T, subRank, isC>{TensorRef<T, rank, isC>::template select<subRank>(coord)};
    }
  };
}
HBTL_NAMESPACE_END
