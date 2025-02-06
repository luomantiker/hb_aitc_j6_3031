#pragma once

#include "ArrayRef.h"
#include "Types.h"
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ude {

struct BufferRef {
  BufferRef() = default;
  BufferRef(void *ptr, size_t size, bool readonly = false, const char *device = nullptr)
      : ptr(ptr), size(size), readonly(readonly), device(device) {}
  void *ptr = nullptr; // Pointer to the underlying storage
  size_t size = 0;     // Byte length of the underlying storage

  bool readonly = false; // Flag to indicate if the underlying storage may be written to

  const char *device = nullptr; // Indicate device of the underlying storage
};

struct TensorRef {
  TensorRef() = default;
  TensorRef(const void *ptr, size_t size, Type type, int64_t rank, ude::ArrayRef<int64_t> shape_,
            ude::ArrayRef<int64_t> stride_, bool readonly = false, const char *device = nullptr, size_t deviceId = 0)
      : dtype(type), rank(rank), shape(shape_), stride(stride_), ptr(const_cast<void *>(ptr)), size(size),
        readonly(readonly), device(device), deviceId(deviceId) {}

  Type dtype = Type::invalid;  // Data type of the underlying storage
  int64_t rank = 0;            // Number of dimensions
  std::vector<int64_t> shape;  // Shape of the tensor (1 entry per dimension)
  std::vector<int64_t> stride; // Number of bytes between adjacent entries (for each per dimension)
  void *ptr = nullptr;
  size_t size = 0;
  bool readonly = false;
  const char *device = nullptr;
  size_t deviceId = 0;
};

struct VectorRef {
  VectorRef(const void *ptr, size_t size, Type type, bool readonly = false, const char *device = nullptr)
      : dtype(type), ptr(const_cast<void *>(ptr)), size(size), readonly(readonly), device(device) {}
  Type dtype = Type::invalid; // Data type of the underlying storage
  void *ptr = nullptr;
  size_t size = 0;
  bool readonly = false;
  const char *device = nullptr;
};

struct TupleRef : public BufferRef {
  TupleRef() = default;
  TupleRef(void *ptr, size_t size, Type type, size_t tupleSize, bool readonly = false, const char *device = nullptr)
      : BufferRef(ptr, size, readonly, device), dtype(type), size(tupleSize) {}

  Type dtype = Type::invalid;
  size_t size = 0;
};

struct StringRef {
  explicit StringRef(const std::string &s) : data(s.data()), len(s.size()) {}
  StringRef(const char *data_, size_t len_) : data(data_), len(len_) {}
  const char *data;
  size_t len;
};

// handle for pointer
struct Pointer {};

// Type trait for protocol detection
template <typename T>
struct is_protocol
    : std::integral_constant<bool, any_type_of<std::is_base_of<TensorRef, T>, std::is_base_of<VectorRef, T>,
                                               std::is_base_of<BufferRef, T>, std::is_base_of<TupleRef, T>>::value> {};

} // namespace ude
