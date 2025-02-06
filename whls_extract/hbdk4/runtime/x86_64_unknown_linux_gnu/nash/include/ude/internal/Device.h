#pragma once

#include <memory>

namespace ude {

class Stream {
public:
  using Base = Stream;
  Stream() = default;
  Stream(Stream &) = delete;
  Stream(const Stream &) = delete;
  virtual ~Stream() = default;

  [[nodiscard]] virtual size_t getId() const { return 0; }
  virtual void destroy() {}
  virtual void synchronize() {}
};

class Device {
public:
  using Base = Device;
  Device() = default;
  Device(Device &) = delete;
  Device(const Device &) = delete;
  virtual ~Device() = default;

  [[nodiscard]] virtual size_t getId() const { return 0; }
  virtual void setId(size_t id) {}
  virtual void *alloc(size_t len) { return nullptr; }
  virtual void free(void *p) {}
  virtual void memCpyHostToDevice(void *dst, const void *src, size_t count) {}
  virtual void memCpyDeviceToHost(void *dst, const void *src, size_t count) {}
  virtual void memCpyDeviceToDevice(void *dst, const void *src, size_t count) {}
  virtual void checkMemory(const void *ptr) {}
  virtual void synchronize() {}
  [[nodiscard]] virtual size_t getDeviceCount() const { return 0; }
};

struct DeviceFactory {
  DeviceFactory() = default;
  virtual ~DeviceFactory() = default;
  virtual std::unique_ptr<Device> createDevice() { return nullptr; };
  virtual std::unique_ptr<Stream> createStream() { return nullptr; };
};

} // namespace ude
