// HBTL Storage

#pragma once

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Support/Compiler.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>

HBTL_NAMESPACE_BEGIN {
  enum DeviceType : size_t {
    cpu = 0,
    cuda = 1,
  };

  class HBTL_EXPORTED Storage {

  public:
    Storage(void *ptr, size_t len, DeviceType device, size_t deviceId, bool writeable, bool owned)
        : ptr(ptr), len(len), device(device), deviceId(deviceId), writeable(writeable), isOwned(owned) {}

    Storage(Storage &) = delete;                // uncopyable
    Storage(const Storage &) = delete;          // uncopyable
    Storage &operator=(const Storage) = delete; // uncopyable

    ~Storage();

    /// create uninitialized storage. ptr is owned by storage.
    static std::shared_ptr<Storage> createUninit(size_t len, DeviceType device = DeviceType::cpu, size_t deviceId = 0);

    /// create from existing data.
    static std::shared_ptr<Storage> createExternal(ArrayRef<char> data, DeviceType device = DeviceType::cpu,
                                                   size_t deviceId = 0);

    /// create from existing mutable data.
    static std::shared_ptr<Storage> createExternal(MutableArrayRef<char> data, DeviceType device = DeviceType::cpu,
                                                   size_t deviceId = 0);

    template <typename T> HBTL_NODISCARD inline size_t size() const {
      assert(aligned<T>() && "alignment violation");
      return len / sizeof(T);
    }

    HBTL_NODISCARD size_t byteSize() const { return len; }

    HBTL_NODISCARD inline bool owned() const { return isOwned; }

    template <typename T> HBTL_NODISCARD bool aligned() const { return (ptr != nullptr) && (len % sizeof(T) == 0U); }

    HBTL_NODISCARD inline bool readonly() const { return !writeable; }

    HBTL_NODISCARD inline bool isCuda() const { return device == DeviceType::cuda; }

    HBTL_NODISCARD inline size_t cudaDeviceId() const { return deviceId; }

    template <typename T> inline const T *data() const {
      assert(aligned<T>() && "alignment violation");
      return reinterpret_cast<const T *>(ptr);
    }

    template <typename T> inline T *data() {
      assert(aligned<T>() && "alignment violation");
      return reinterpret_cast<T *>(ptr);
    }

    template <typename T> ArrayRef<T> getData() const { return ArrayRef<T>(data<T>(), size<T>()); }

    template <typename T> MutableArrayRef<T> getMutData() { return {data<T>(), size<T>()}; }

  private:
    void *ptr = nullptr;
    size_t len = 0U;
    DeviceType device = DeviceType::cpu;
    size_t deviceId = 0U;
    bool writeable; /// indicate permissions for getMutData
    bool isOwned;
  };
}
HBTL_NAMESPACE_END
