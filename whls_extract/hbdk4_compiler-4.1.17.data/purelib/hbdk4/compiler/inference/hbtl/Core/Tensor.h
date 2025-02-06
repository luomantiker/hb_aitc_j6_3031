// HBTL Tensor

#pragma once

#ifndef HBTL_SUPPORT_TENSOR_H_
#define HBTL_SUPPORT_TENSOR_H_

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Support/Compiler.h"
#include <array>
#include <cstdint>

#include <utility>

#include "hbtl/Core/ElementType.h"
#include "hbtl/Core/Storage.h"
#include "hbtl/Core/TensorRef.h"
#include "hbtl/Core/TensorType.h"
#include "hbtl/Support/ADTExtras.h"
#include "hbtl/Support/MathExtras.h"
HBTL_NAMESPACE_BEGIN {

  HBTL_EXPORTED std::vector<int64_t> getStrides(ArrayRef<int64_t> sizes, int64_t byteSize);
  HBTL_EXPORTED std::pair<std::vector<int64_t>, std::vector<int64_t>> optimize(
      ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, int64_t byteSize);

  constexpr size_t axisLimit = 11U;
  class HBTL_EXPORTED Tensor {
  public:
    constexpr static size_t MAGIC_HEAD = 0XFADE;

    /// allow empty tensor
    Tensor() = default;
    /// return false when construct from nothing
    explicit operator bool() const {
      if (storage == nullptr) {
        return false;
      }
      if (type == ElementType::invalid) {
        return false;
      }
      return std::all_of(sizes.begin(), sizes.begin() + rank, [](auto s) { return s > 0; });
    }

    HBTL_NODISCARD bool valid() const { return this->operator bool(); }

    Tensor(const Tensor &) = default;
    Tensor(Tensor &) = default;

    Tensor &operator=(const Tensor &) = default;

    /// default constructor
    Tensor(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ElementType type, int64_t offset,
           std::shared_ptr<Storage> storage)
        : type(type), offset(offset) {
      assert(sizes.size() <= axisLimit && "tensor's max rank is 11");
      assert(sizes.size() == strides.size() && "sizes and strides should have same size");
      rank = sizes.size();
      for (auto i = 0U; i < rank; ++i) {
        this->sizes[i] = sizes[i];
        this->strides[i] = strides[i];
      }
      setStorage(std::move(storage));
    }

    bool operator==(const Tensor &other) const {
      if (other.getSizes() != this->getSizes()) {
        return false;
      }
      if (other.getStrides() != this->getStrides()) {
        return false;
      }
      if (other.getByteOffset() != this->getByteOffset()) {
        return false;
      }
      if (other.getStorage() != this->getStorage()) {
        return false;
      }
      return true;
    }

    bool operator!=(const Tensor &other) const { return !((*this) == other); }

    /// create from external data with enum type
    static Tensor createExternal(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ElementType type,
                                 ArrayRef<char> data, int64_t offset = 0, DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(data, deviceType);
      return {sizes, strides, type, offset, std::move(storage)};
    }
    /// create from external mutable data with enum type
    static Tensor createExternal(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ElementType type,
                                 MutableArrayRef<char> data, int64_t offset = 0,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(data, deviceType);
      return {sizes, strides, type, offset, std::move(storage)};
    }
    /// create from external data with template type
    template <typename T>
    static Tensor createExternal(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<T> data,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(
          ArrayRef<char>(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T)), deviceType);
      return {sizes, strides, getElementType<T>(), 0, std::move(storage)};
    }
    /// create from external mutable data with template type
    template <typename T>
    static Tensor createExternal(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, MutableArrayRef<T> data,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(
          MutableArrayRef<char>(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T)), deviceType);

      return {sizes, strides, getElementType<T>(), 0, std::move(storage)};
    }

    /// create from external data with enum type
    static Tensor createExternal(ArrayRef<int64_t> sizes, ElementType type, ArrayRef<char> data,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(data, deviceType);
      return {sizes, hbtl::getStrides(sizes, getByteSize(type)), type, 0, std::move(storage)};
    }
    /// create from external mutable data with enum type
    static Tensor createExternal(ArrayRef<int64_t> sizes, ElementType type, MutableArrayRef<char> data,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(data, deviceType);
      return {sizes, hbtl::getStrides(sizes, getByteSize(type)), type, 0, std::move(storage)};
    }
    /// create from external data with template type
    template <typename T>
    static Tensor createExternal(ArrayRef<int64_t> sizes, ArrayRef<T> data, DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(
          ArrayRef<char>(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T)), deviceType);
      const auto type = getElementType<T>();
      return {sizes, hbtl::getStrides(sizes, getByteSize(type)), type, 0, std::move(storage)};
    }
    /// create from external mutable data with template type
    template <typename T>
    static Tensor createExternal(ArrayRef<int64_t> sizes, MutableArrayRef<T> data,
                                 DeviceType deviceType = DeviceType::cpu) {
      auto storage = Storage::createExternal(
          MutableArrayRef<char>(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T)), deviceType);
      const auto type = getElementType<T>();
      return {sizes, hbtl::getStrides(sizes, getByteSize(type)), type, 0, std::move(storage)};
    }

    /// create one dimensional tensor from external data
    template <typename T> static Tensor createExternal(ArrayRef<T> data, DeviceType deviceType = DeviceType::cpu) {
      return createExternal<T>(data.size(), data, deviceType);
    }
    /// create one dimensional tensor from external data
    template <typename T>
    static Tensor createExternal(MutableArrayRef<T> data, DeviceType deviceType = DeviceType::cpu) {
      return createExternal<T>(data.size(), data, deviceType);
    }

    /// create an invalid tensor
    static Tensor createInvalid() { return {}; }
    /// create a scalar tensor
    static Tensor createScalar(ElementType type) { return createUninit({}, {}, type); }
    template <typename T> static Tensor createScalar() { return createUninit<T>({}); }
    /// create from footprint and type
    static Tensor createUninit(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ElementType type,
                               DeviceType device = DeviceType::cpu, size_t deviceId = 0, void *ptr = nullptr) {
      auto size = getByteSize(type);
      for (auto i = 0U; i < sizes.size(); ++i) {
        size += (sizes[i] - 1) * strides[i];
      }
      if (ptr != nullptr) {
        return {sizes, strides, type, 0,
                std::make_shared<Storage>(ptr, static_cast<size_t>(size), device, deviceId, false, true)};
      }
      return {sizes, strides, type, 0, Storage::createUninit(static_cast<size_t>(size))};
    }
    /// create contiguous tensor from shape and type
    static Tensor createUninit(ArrayRef<int64_t> shape, ElementType type, DeviceType device = DeviceType::cpu,
                               size_t deviceId = 0, void *ptr = nullptr) {
      return createUninit(shape, hbtl::getStrides(shape, getByteSize(type)), type, device, deviceId, ptr);
    }
    /// create contiguous tensor from shape with template type
    template <typename T> static Tensor createUninit(ArrayRef<int64_t> shape) {
      auto type = getElementType<T>();
      return createUninit(shape, hbtl::getStrides(shape, getByteSize(type)), type);
    }
    /// create contiguous tensor from tensor type
    static Tensor createUninit(const TensorType &type) { return createUninit(type.getSizes(), type.getType()); }
    /// create without storage, only for infer type. Any operation on the storage is invalid, do not use
    static Tensor createWithoutStorage(ArrayRef<int64_t> shape, ElementType type) {
      auto shapeWoDynInfo = shape.vec();
      if ((shapeWoDynInfo.size() != 1) || (shapeWoDynInfo[0] != -1)) {
        for (auto i = 0UL; i < shapeWoDynInfo.size(); ++i) {
          if (shapeWoDynInfo[i] < 0) {
            shapeWoDynInfo[i] *= -1;
          }
        }
      }
      return {shape, hbtl::getStrides(shapeWoDynInfo, getByteSize(type)), type, 0, nullptr};
    }

    /// create contiguous tensor from shape and specified initialize value
    template <typename T> static Tensor createSplatted(ArrayRef<int64_t> shape, T val) {
      auto tensor = createUninit<T>(shape);
      for (auto &v : tensor.template getMutData<T>()) {
        v = val;
      }
      return tensor;
    }

    /// create contiguous tensor from shape and custom data generator
    template <typename T, typename Generator> static Tensor createGenerated(ArrayRef<int64_t> shape, Generator gen) {
      auto tensor = createUninit<T>(shape);
      for (auto &v : tensor.template getMutData<T>()) {
        v = gen();
      }
      return tensor;
    }

    // compute tensor's bytesize
    HBTL_NODISCARD int64_t getByteCapacity(int64_t startDim = 0) const {
      int64_t limit = getByteSize(type);
      for (auto i = startDim; i < getRank(); ++i) {
        limit += (sizes[i] - 1) * strides[i];
      }
      return limit;
    }

    HBTL_NODISCARD int64_t getByteOffset() const { return offset; }
    HBTL_NODISCARD int64_t getElementOffset() const {
      assert(getByteOffset() % std::max(getByteSize(type), 1L) == 0); // unaligned access may cause catastrophe
      return getByteOffset() / std::max(getByteSize(type), 1L);
    }
    HBTL_NODISCARD int64_t getByteLimit() const { return (getByteCapacity() + offset); }
    HBTL_NODISCARD int64_t getElementLimit() const { return getByteLimit() / std::max(getByteSize(type), 1L); }
    HBTL_NODISCARD int64_t getElementSize() const { return getByteCapacity() / std::max(getByteSize(type), 1L); }

    HBTL_NODISCARD ElementType getType() const { return type; }
    HBTL_NODISCARD TensorType getTensorType() const { return {this->getSizes(), this->getType()}; }

    // get footprint members
    HBTL_NODISCARD int64_t getRank() const { return static_cast<int64_t>(rank); }
    HBTL_NODISCARD int64_t getSize(int64_t dim) const { return sizes[normDim(dim)]; }
    HBTL_NODISCARD int64_t getStride(int64_t dim) const { return strides[normDim(dim)]; }
    HBTL_NODISCARD std::shared_ptr<Storage> getStoragePtr() const { return storage; }
    HBTL_NODISCARD const Storage *getStorage() const { return storage.get(); }
    HBTL_NODISCARD Storage *getStorage() { return storage.get(); }
    HBTL_NODISCARD ArrayRef<int64_t> getSizes() const { return {sizes.begin(), rank}; }
    HBTL_NODISCARD ArrayRef<int64_t> getStrides() const { return {strides.begin(), rank}; }
    HBTL_NODISCARD MutableArrayRef<int64_t> getMutSizes() { return {sizes.begin(), rank}; }
    HBTL_NODISCARD MutableArrayRef<int64_t> getMutStrides() { return {strides.begin(), rank}; }
    HBTL_NODISCARD std::vector<int64_t> getSizesCopy() const { return getSizes().vec(); }
    HBTL_NODISCARD std::vector<int64_t> getStridesCopy() const { return getStrides().vec(); }

    void setStorage(std::shared_ptr<Storage> storage) {
      if (storage != nullptr) {
        assert(offset >= 0 && offset <= static_cast<int64_t>(storage->byteSize()) && "invalid offset");
        assert(getByteLimit() <= static_cast<int64_t>(storage->byteSize()) &&
               "invalid storage"); // any valid tensor should
      }
      this->storage = std::move(storage);
    }

    void setByteOffset(int64_t offset) { this->offset = offset; }

    // get hash value using BLAKE3
    HBTL_NODISCARD Tensor raw() const {
      auto shape = this->getSizesCopy();
      if (shape.empty()) { // rank == 0
        shape.push_back(1);
      }
      shape.rbegin()[0] *= getByteSize(this->getType());
      return this->contiguous().view<char>(shape);
    }

    HBTL_NODISCARD OwningArrayRef<char> serialize() const;
    static Tensor deserialize(ArrayRef<char> data, bool copy = false);

    /// get coordinate from element position
    HBTL_NODISCARD std::vector<int64_t> getCoord(int64_t pos) const {
      std::vector<int64_t> coord;
      auto byteSize = std::max(getByteSize(type), 1L);
      if (pos < getElementSize()) {
        coord.reserve(rank);
        for (auto i = 0U; i < rank; ++i) {
          if (strides[i] == 0) {
            coord.push_back(0);
            continue;
          }
          const auto stride = strides[i] / byteSize;
          auto size = pos / stride;
          if (size >= sizes[i]) {
            coord = {};
            break;
          }
          coord.push_back(size);
          pos %= stride;
        }
      }
      assert(!coord.empty() && "pos not in range");
      return coord;
    }

    /// get position from element coordinate
    HBTL_NODISCARD int64_t getPos(ArrayRef<int64_t> coord) const {
      const auto pos = getBytePos(coord);
      assert(pos >= 0 && "coord exceed range");
      return pos / std::max(getByteSize(type), 1L);
    }

    /// get byte offset of given coord
    HBTL_NODISCARD int64_t getBytePos(ArrayRef<int64_t> coord) const {
      assert(coord.size() == rank && "rank mismatch");
      int64_t pos = 0;
      for (auto i = 0U; i < rank; ++i) {
        if (coord[i] >= sizes[i]) {
          return -1;
        }
        pos += strides[i] * coord[i];
      }
      return pos;
    }
    /// get a copy tensor value. need calculate multiply times. slow.
    template <typename T> T value(ArrayRef<int64_t> coord) const {
      auto pos = getPos(coord);
      assert(pos >= 0 && "invalid pos");
      return getRawData<T>()[pos];
    }
    /// get a reference of tensor value. need calculate multiply times. slow.
    template <typename T> T &value(ArrayRef<int64_t> coord) {
      auto pos = getPos(coord);
      assert(pos >= 0 && "invalid pos");
      return getMutRawData<T>()[pos];
    }

    /// get an ArrayRef of all data. it quires to the tensor be contiguous. safe to iterate incrementally.
    template <typename T> ArrayRef<T> getData() const & {
      assert(isContiguous() && "only valid for contiguous data");
      return this->getRawData<T>();
    }

    /// TODO: make this function valid
    // template <typename T> ArrayRef<T> getData() && = delete;

    /// get a writable ArrayRef of all data. it quires to the tensor be contiguous. safe to iterate incrementally.
    template <typename T> MutableArrayRef<T> getMutData() & {
      assert(isContiguous() && "only valid for contiguous data");
      return this->getMutRawData<T>();
    }

    /// get a contiguous TensorRef of the bigTensor

    template <typename T, size_t rank> HBTL_NODISCARD TensorRef<T, rank> select(ArrayRef<int64_t> coord) const {
      assert(static_cast<int64_t>(rank + coord.size()) == getRank() &&
             "sum of coord size and rank should be equal to Rank");
      assert(getElementType<T>() == type && "the target type and origin tensor type mismatch");

      int64_t elemOffset = 0;
      for (auto i = 0U; i < coord.size(); ++i) {
        elemOffset += coord[i] * getStride(i);
      }
      const auto byteSize = std::max(getByteSize(type), 1L);
      elemOffset /= byteSize;
      int64_t cap = getByteCapacity(static_cast<int64_t>(coord.size())) / byteSize;
      return TensorRef<T, rank>(getRawData<T>().slice(elemOffset, cap), getSizes().drop_front(coord.size()),
                                getStrides().drop_front(coord.size()));
    }

    template <typename T, size_t rank> HBTL_NODISCARD MutableTensorRef<T, rank> select(ArrayRef<int64_t> coord) {
      assert(static_cast<int64_t>(rank + coord.size()) == getRank() &&
             "sum of coord size and rank should be equal to Rank");
      assert(getElementType<T>() == type && "the target type and origin tensor type mismatch");

      // getData will compute the big tensor origin offset
      int64_t elemOffset = 0;
      for (auto i = 0U; i < coord.size(); ++i) {
        elemOffset += coord[i] * getStride(i);
      }
      const auto byteSize = std::max(getByteSize(type), 1L);
      elemOffset /= byteSize;
      int64_t cap = getByteCapacity(static_cast<int64_t>(coord.size())) / byteSize;
      return MutableTensorRef<T, rank>(getMutRawData<T>().slice(elemOffset, cap), getSizes().drop_front(coord.size()),
                                       getStrides().drop_front(coord.size()));
    }

    // tensorRef for tensor which may be not contiguous in last dim
    template <typename T, size_t rank, bool isC>
    HBTL_NODISCARD TensorRef<T, rank, false> select(ArrayRef<int64_t> coord) const {
      assert(static_cast<int64_t>(rank + coord.size()) == getRank() &&
             "sum of coord size and rank should be equal to Rank");
      assert(getElementType<T>() == type && "the target type and origin tensor type mismatch");

      int64_t elemOffset = 0;
      for (auto i = 0U; i < coord.size(); ++i) {
        elemOffset += coord[i] * getStride(i);
      }
      const auto byteSize = std::max(getByteSize(type), 1L);
      elemOffset /= byteSize;
      int64_t cap = getByteCapacity(static_cast<int64_t>(coord.size())) / byteSize;
      return TensorRef<T, rank, false>(getRawData<T>().slice(elemOffset, cap), getSizes().drop_front(coord.size()),
                                       getStrides().drop_front(coord.size()));
    }

    template <typename T, size_t rank, bool isC>
    HBTL_NODISCARD MutableTensorRef<T, rank, false> select(ArrayRef<int64_t> coord) {
      assert(static_cast<int64_t>(rank + coord.size()) == getRank() &&
             "sum of coord size and rank should be equal to Rank");
      assert(getElementType<T>() == type && "the target type and origin tensor type mismatch");

      // getData will compute the big tensor origin offset
      int64_t elemOffset = 0;
      for (auto i = 0U; i < coord.size(); ++i) {
        elemOffset += coord[i] * getStride(i);
      }
      const auto byteSize = std::max(getByteSize(type), 1L);
      elemOffset /= byteSize;
      int64_t cap = getByteCapacity(static_cast<int64_t>(coord.size())) / byteSize;
      return MutableTensorRef<T, rank, false>(getMutRawData<T>().slice(elemOffset, cap),
                                              getSizes().drop_front(coord.size()),
                                              getStrides().drop_front(coord.size()));
    }

    /// get a child tensor that uses same storage form parent tensor. Coord can be partial (< real rank).
    HBTL_NODISCARD Tensor select(ArrayRef<int64_t> coord) const {
      assert(isInside(coord) && "select coord not inside tensor");
      int64_t byteOffset = getByteOffset();
      for (auto i = 0U; i < coord.size(); ++i) {
        byteOffset += coord[i] * getStride(i);
      }
      return {getSizes().drop_front(coord.size()), getStrides().drop_front(coord.size()), getType(), byteOffset,
              storage};
    }
    /// get a child tensor that uses same storage from parent tensor.
    HBTL_NODISCARD Tensor select(int64_t index) const {
      assert(index < this->getSize(0) && index >= 0 && "select index exceeds tensor shape");
      auto byteOffset = getByteOffset() + index * strides[0];
      return {getSizes().drop_front(), getStrides().drop_front(), getType(), byteOffset, storage};
    }

    /// get a child tensor that uses same storage from parent tensor.
    HBTL_NODISCARD Tensor select(int64_t dim, int64_t index) const {
      dim = normDim(dim);
      index += (index < 0) ? this->getSize(dim) : 0;
      assert((index < this->getSize(dim)) && (index >= 0) && "select index exceeds tensor shape");

      auto byteOffset = this->getByteOffset() + index * this->getStride(dim);

      auto newShape = this->getSizesCopy();
      auto newStrides = this->getStridesCopy();
      newShape.erase(newShape.begin() + dim);
      newStrides.erase(newStrides.begin() + dim);
      return {newShape, newStrides, type, byteOffset, storage};
    }

    /// copy a tensor of same shape to current tensor. skip if two tensors are essentially identical.
    void copy(const Tensor &other) {
      assert(getSizes() == other.getSizes() && "cannot copy tensor with different shape");
      if (HBTL_UNLIKELY(other != *this)) {
        fill(other);
      }
    }

    /// fill a small tensor to current big tensor.
    void fill(const Tensor &small, ArrayRef<int64_t> begin = {}, ArrayRef<int64_t> step = {});

    /// crop sub tensor. no data copy. only modify tensor states.
    HBTL_NODISCARD Tensor crop(ArrayRef<int64_t> shape, ArrayRef<int64_t> begin = {},
                               ArrayRef<int64_t> step = {}) const;

    /// slice sub tensor. same as crop but only on one dimension
    HBTL_NODISCARD Tensor slice(int64_t dim, int64_t begin, int64_t end) const {
      dim = normDim(dim);

      assert((0 <= end) && (end <= this->getSize(dim)) && "slice end exceeds tensor shape");
      assert((0 <= begin) && (begin <= end) && "invalid slice begin");

      auto byteOffset = this->getByteOffset() + begin * this->getStride(dim);

      auto newShape = this->getSizesCopy();
      newShape[dim] = end - begin;
      return {newShape, getStrides(), type, byteOffset, storage};
    }

    /// view tensor with other footprint and type
    HBTL_NODISCARD Tensor view(ElementType newType) const {
      assert(getBitWidth(newType) == getBitWidth(type) && "tensor can view as new type of same type width");
      return {getSizes(), getStrides(), newType, offset, storage};
    }
    /// view tensor with other footprint and type
    template <typename T> HBTL_NODISCARD Tensor view() const { return view(getElementType<T>()); }

    /// view tensor with new type and new shape. current tensor must be contiguous
    HBTL_NODISCARD Tensor view(ElementType newType, ArrayRef<int64_t> newShape) const {
      assert(isContiguous() && "only contiguous tensor can view as new type and new shape");
      assert(getByteSize(this->getType()) * vector::reduceMul(this->getSizes()) ==
                 getByteSize(newType) * vector::reduceMul(newShape) &&
             "byte size of new type and new shape should be identical");
      return {newShape, hbtl::getStrides(newShape, getByteSize(newType)), newType, offset, storage};
    }

    /// view tensor with new shape. current tensor must be contiguous
    template <typename T> HBTL_NODISCARD Tensor view(ArrayRef<int64_t> newShape) const {
      return view(getElementType<T>(), newShape);
    }

    /// view tensor with new footprint
    HBTL_NODISCARD Tensor view(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides) const {
      return {sizes, strides, type, offset, storage};
    }

    /// view tensor with new shape
    HBTL_NODISCARD Tensor view(ArrayRef<int64_t> newShape) const { return view(type, newShape); }

    /// cast tensor element to other type.
    HBTL_NODISCARD Tensor cast(ElementType newType) const;
    template <typename T> HBTL_NODISCARD Tensor cast() const { return cast(getElementType<T>()); }

    /// reshape tensor to other shape.
    HBTL_NODISCARD Tensor reshape(ArrayRef<int64_t> shape) const { return this->contiguous().view(shape); }

    /// flatten axes in the range.
    HBTL_NODISCARD Tensor flatten(int64_t begin, int64_t end = -1) const {
      begin = normDim(begin);
      end = normDim(end);
      if (begin == end) {
        return *this;
      }

      std::vector<int64_t> newShape;
      for (auto i = 0; i < this->getRank(); ++i) {
        auto axis = sizes[i];
        if (i > begin && i <= end) {
          newShape.rbegin()[0] *= axis;
        } else {
          newShape.push_back(axis);
        }
      }
      return this->reshape(newShape);
    }

    /// check if tensor can be broadcast to target shape
    HBTL_NODISCARD bool isBroadcastable(ArrayRef<int64_t> targetShape, int64_t until = -1) const {
      auto targetRank = static_cast<int64_t>(targetShape.size());
      if (this->getRank() > targetRank) {
        return false;
      }

      auto rankDiff = targetRank - this->getRank();

      until = (until < 0) ? until + this->getRank() : until;
      for (auto i = 0; i <= until; ++i) {
        if (this->getSize(i) != targetShape[static_cast<size_t>(i + rankDiff)]) {
          if (this->getSize(i) != 1) {
            return false;
          }
        }
      }
      return true;
    }

    // broadcast to targe shape
    HBTL_NODISCARD Tensor broadcast(ArrayRef<int64_t> targetShape, int64_t until = -1) const {
      assert(isBroadcastable(targetShape, until));

      auto targetRank = static_cast<int64_t>(targetShape.size());
      auto rankDiff = targetRank - this->getRank();

      std::vector<int64_t> newSize;
      std::vector<int64_t> newStride;
      newSize.reserve(targetShape.size());
      newStride.reserve(targetShape.size());
      for (auto i = 0; i < rankDiff; ++i) {
        newSize.emplace_back(targetShape[static_cast<size_t>(i)]);
        newStride.emplace_back(0);
      }

      until = (until < 0) ? until + this->getRank() : until;
      for (auto i = 0; i <= until; ++i) {
        auto targetSize = targetShape[static_cast<size_t>(i + rankDiff)];
        if (sizes[i] != targetSize) {
          newSize.emplace_back(targetSize);
          newStride.emplace_back(0);
        } else {
          newSize.emplace_back(sizes[i]);
          newStride.emplace_back(strides[i]);
        }
      }
      for (auto i = until + 1; i < this->getRank(); ++i) {
        newSize.emplace_back(sizes[i]);
        newStride.emplace_back(strides[i]);
      }

      return {newSize, newStride, type, offset, storage};
    }

    // insert axis before dim
    HBTL_NODISCARD Tensor unsqueeze(int64_t dim) const {
      // if shape is [2,3], unsqueeze dim -1 means view as [2,3,1], unsqueeze dim 2 means view as [2,3,1]
      auto normDim = (dim < 0) ? dim + this->getRank() + 1 : dim;
      if (normDim == this->getRank()) { // append axis to the last.
        std::vector<int64_t> newSize = {getSizes().begin(), getSizes().end()};
        std::vector<int64_t> newStride = {getStrides().begin(), getStrides().end()};
        newSize.emplace_back(1);
        newStride.emplace_back(0);
        return {newSize, newStride, type, offset, storage};
      }

      std::vector<int64_t> newSize;
      std::vector<int64_t> newStride;
      newSize.reserve(static_cast<size_t>(this->getRank()) + 1U);
      newStride.reserve(static_cast<size_t>(this->getRank()) + 1U);
      for (auto i = 0U; i < getRank(); ++i) {
        if (static_cast<int64_t>(i) == normDim) {
          newSize.emplace_back(1);
          newStride.emplace_back(0);
        }
        newSize.emplace_back(sizes[i]);
        newStride.emplace_back(strides[i]);
      }

      return {newSize, newStride, type, offset, storage};
    }

    // delete axis before dim
    HBTL_NODISCARD Tensor squeeze(int64_t dim) const {
      // if shape is [2,3], unsqueeze dim -1 means view as [2,3,1], unsqueeze dim 2 means view as [2,3,1]
      auto normDim = (dim < 0) ? dim + this->getRank() + 1 : dim;
      assert(sizes[normDim] == 1 && "cannot squeeze axis size other than 1");

      std::vector<int64_t> newSize;
      std::vector<int64_t> newStride;
      newSize.reserve(getRank() - 1);
      newStride.reserve(getRank() - 1);
      for (auto i = 0U; i < getRank(); ++i) {
        if (i == normDim) {
          continue;
        }
        newSize.push_back(sizes[i]);
        newStride.push_back(strides[i]);
      }
      return {newSize, newStride, type, offset, storage};
    }

    /// expand tensor to a bigger one filled with value.
    template <typename T> HBTL_NODISCARD Tensor expand(ArrayRef<int64_t> shape, T val) const {
      return createSplatted<T>(shape, val).fill(*this);
    }

    /// expand tensor to a bigger one filled with 0.
    HBTL_NODISCARD Tensor expandZeros(ArrayRef<int64_t> largeTensorShape) const;

    /// broadcast tensor along a specific dimension
    HBTL_NODISCARD Tensor broadcast(int64_t dim, int64_t size) const {
      dim = normDim(dim);
      assert(sizes[dim] == 1 && "only axis with size of 1 is broadcastable");

      std::vector<int64_t> newSize = getSizesCopy();
      std::vector<int64_t> newStride = getStridesCopy();

      *(newSize.begin() + dim) = size;
      *(newStride.begin() + dim) = 0;
      return {newSize, newStride, type, offset, storage};
    }

    /// transpose tensor to a new dim order, permute must be ranging from 0 to tensor rank-1
    HBTL_NODISCARD Tensor transpose(ArrayRef<int64_t> permute) const {
      assert(this->getRank() == static_cast<int64_t>(permute.size()) && "permute invalid");
      // make negative axis positive
      auto newPermute = normDim(permute);
      // check if permute is unique

      std::vector<int64_t> newSize;
      std::vector<int64_t> newStride;
      for (auto p : permute) {
        p = (p >= 0) ? p : (permute.size() + p);

        newSize.push_back(sizes[p]);
        newStride.push_back(strides[p]);
      }
      return {newSize, newStride, type, offset, storage};
    }

    /// clone to a new tensor and new storage.
    HBTL_NODISCARD Tensor clone() const {
      if (HBTL_LIKELY(*this)) {
        auto newT = createUninit(getSizes(), getType());
        newT.fill(*this);
        return newT;
      }
      return *this;
    }

    /// check contiguous since axis. accept negative value. if not, copy data to new storage.
    HBTL_NODISCARD Tensor contiguous(int64_t since = 0) const {
      if (HBTL_LIKELY(*this)) {
        since = normDim(since);
        auto isC = isContiguous(since);
        if (!isC) {
          return clone();
        }
      }
      return *this;
    }

    /// check whether two tensors are essentially the same
    static bool equal(const Tensor &lhs, const Tensor &rhs);

    template <typename T> void set(T val) {
      for (auto &v : getMutData<T>()) {
        v = val;
      }
    }

    /// get read-only raw array ref. may need iterate with stride. internal use only.
    template <typename T> ArrayRef<T> getRawData() const {
      if (HBTL_LIKELY(*this)) {
        assert((type == ElementType::invalid || sizeof(T) == getByteSize(type)) && "type mismatch");
        auto data = storage->getData<T>();
        return data.slice(static_cast<size_t>(getElementOffset()), static_cast<size_t>(getElementSize()));
      }
      return {};
    }

    /// get writable raw array ref. may need iterate with stride. internal use only.
    template <typename T> MutableArrayRef<T> getMutRawData() {
      if (HBTL_LIKELY(*this)) {
        assert((type == ElementType::invalid || ::hbtl::getElementType<T>() == type) && "type mismatch");
        auto data = storage->getMutData<T>();
        return data.slice(static_cast<size_t>(getElementOffset()), static_cast<size_t>(getElementSize()));
      }
      return {};
    }

    inline ArrayRef<unsigned char> getBytes() const {
      if (HBTL_LIKELY(*this)) {
        auto data = storage->getData<unsigned char>();
        return data.slice(static_cast<size_t>(getByteOffset()), static_cast<size_t>(getByteCapacity()));
      }
      return {};
    }

    inline MutableArrayRef<unsigned char> getMutBytes() {
      if (HBTL_LIKELY(*this)) {
        auto data = storage->getMutData<unsigned char>();
        return data.slice(static_cast<size_t>(getByteOffset()), static_cast<size_t>(getByteCapacity()));
      }
      return {};
    }

    /// reinterpret tensor to new type. e.g. tensor<10x10xsi8> can be reinterpret to tensor<10x5xsi16>
    HBTL_NODISCARD Tensor reinterpret(ElementType newType) const;

    /// reinterpret tensor to new type. e.g. tensor<10x10xsi8> can be reinterpret to tensor<10x5xsi16>
    template <typename T> HBTL_NODISCARD Tensor reinterpret() const { return reinterpret(getElementType<T>()); }

    // sorted dims according stride for judging overlap
    HBTL_NODISCARD std::vector<int64_t> getOrderedIndices() const {
      std::vector<int64_t> sortedIndices(rank, 0);
      for (auto i = 1U; i < rank; ++i) {
        sortedIndices[i] = i;
      }
      ::hbtl::stable_sort(sortedIndices, [this](const auto &left, const auto &right) {
        return std::make_tuple(strides[left], sizes[left]) < std::make_tuple(strides[right], sizes[right]);
      });
      return sortedIndices;
    }

    /// check if strides are overlapping, broadcast(zero stride) dims are excluded
    HBTL_NODISCARD bool isOverlapped() const;

    HBTL_NODISCARD bool isContiguous(int64_t since = 0) const;

    HBTL_NODISCARD inline int64_t normDim(int64_t dim) const {
      dim += ((dim < 0) ? getRank() : 0);
      assert(dim >= 0 && (getRank() == 0 || dim < getRank()) && "invalid dim");
      return dim;
    }

    HBTL_NODISCARD inline bool isInside(ArrayRef<int64_t> coord) const {
      if (coord.size() > rank) {
        return false;
      }
      for (auto i = 0U; i < coord.size(); ++i) {
        if (coord[i] < 0 || coord[i] >= sizes[i]) {
          return false;
        }
      }
      return true;
    }

    HBTL_NODISCARD inline std::vector<int64_t> normDim(ArrayRef<int64_t> dims) const {
      auto ret = std::vector<int64_t>(dims.size());
      for (auto i = 0U; i < dims.size(); ++i) {
        ret[i] = normDim(dims[i]);
      }
      return ret;
    }

    // Used for inferShape
    void setType(hbtl::ElementType newType) { type = newType; }

    void setShape(const std::vector<int64_t> &newShape) {
      assert(newShape.size() <= axisLimit && "hbtl Tensor's new shape can't greater than axisLimit");
      memcpy(sizes.data(), newShape.data(), sizeof(int64_t) * newShape.size());
      rank = newShape.size();
    }

    void setStride(const std::vector<int64_t> &newStride) {
      assert(newStride.size() <= axisLimit && "hbtl Tensor's new stride can't greater than axisLimit");
      memcpy(strides.data(), newStride.data(), sizeof(int64_t) * newStride.size());
      rank = newStride.size();
    }

    ElementType &getRefType() { return type; }

    size_t &getRefRank() { return rank; }

  private:
    size_t rank = 0;
    std::array<int64_t, axisLimit> sizes = {};
    std::array<int64_t, axisLimit> strides = {};
    ElementType type = ElementType::invalid; /// element type of data
    int64_t offset = 0;                      /// byte offset of storage
    std::shared_ptr<Storage> storage = nullptr;
  };

  HBTL_EXPORTED std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_TENSOR_H_
