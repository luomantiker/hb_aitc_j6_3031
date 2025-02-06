#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "ude/internal/Device.h"
#include "ude/public/ArrayRef.h"
#include "ude/public/Protocols.h"
#include "ude/public/Status.h"
#include "ude/public/Variable.h"

namespace mlir::hbdk::pub::xxq {

#define XQ_CAPI_EXPORTED __attribute__((visibility("default")))

using UserFunction =
    std::function<ude::Status(void *, ude::ArrayRef<ude::Variable *>, ude::ArrayRef<ude::Variable *>, void *)>;
using UserDataDeleter = std::function<void(void *)>;

namespace detail {
class ContextImpl;
class ModuleImpl;
class SessionImpl;
class FunctionImpl;
class ArgumentImpl;
class TypeImpl;
class QuantInfoImpl;
class EngineImpl;
} // namespace detail

class Module;
class Session;
class Function;
class Argument;
class Type;
class QuantInfo;

// MLIR Context
class XQ_CAPI_EXPORTED Context {
public:
  Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  ~Context();

  [[nodiscard]] ude::Status loadCudaLibrary(const char *libPath);
  [[nodiscard]] ude::Status loadCustomLibrary(const char *libPath);
  [[nodiscard]] ude::DeviceFactory *getDeviceFactory() const;
  [[nodiscard]] detail::ContextImpl *getImpl() const;

private:
  std::unique_ptr<detail::ContextImpl> impl;
};

// MLIR Module
class XQ_CAPI_EXPORTED Module {
public:
  Module() = delete;
  explicit Module(const char *filename);
  explicit Module(ude::ArrayRef<char> bytecode);
  explicit Module(ude::StringRef rawString);
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  ~Module();

  [[nodiscard]] ude::Status verify() const;
  [[nodiscard]] ude::Status load(Context *ctx);
  [[nodiscard]] std::vector<Function> getFunctions() const;
  [[nodiscard]] detail::ModuleImpl *getImpl() const;

private:
  std::unique_ptr<detail::ModuleImpl> impl;
};

// HBIR Session
class XQ_CAPI_EXPORTED Session {
public:
  Session();
  explicit Session(Module *module);
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  ~Session();

  void setEnableTrack(bool value);
  [[nodiscard]] bool getEnableTrack() const;
  void setVerbose(bool value);
  [[nodiscard]] bool getVerbose() const;
  void setBackend(ude::StringRef backend);
  [[nodiscard]] std::string getBackend() const;
  void setDevice(ude::StringRef type, size_t id = 0);
  [[nodiscard]] std::string getDevice() const;
  [[nodiscard]] ude::Status loadCudaLibrary(const char *libPath);

  ude::Status feed(ude::StringRef funcName, ude::MutableArrayRef<ude::Variable *> outs,
                   ude::ArrayRef<ude::Variable *> ins, bool record_db = false);
  ude::Status feed(void *op, ude::MutableArrayRef<ude::Variable *> outs, ude::ArrayRef<ude::Variable *> ins,
                   bool record_db = false);

  void registerUserPostHook(UserFunction userFunc, void *userData, UserDataDeleter userDel);

private:
  std::unique_ptr<detail::SessionImpl> impl;
};

// MLIR Function
class XQ_CAPI_EXPORTED Function {
  friend Module;

public:
  Function() = delete;
  Function(const Function &) = default;
  Function &operator=(const Function &) = default;
  ~Function() = default;

  [[nodiscard]] ude::StringRef getName() const;
  [[nodiscard]] ude::StringRef getDesc() const;
  [[nodiscard]] std::vector<Argument> getInputs() const;
  [[nodiscard]] std::vector<Argument> getOutputs() const;

protected:
  explicit Function(detail::FunctionImpl *i);

private:
  detail::FunctionImpl *impl;
};

// MLIR Value
class XQ_CAPI_EXPORTED Argument {
  friend Function;

public:
  Argument() = delete;
  Argument(const Argument &) = default;
  Argument &operator=(const Argument &) = default;
  ~Argument() = default;

  [[nodiscard]] ude::StringRef getName() const;
  [[nodiscard]] ude::StringRef getDesc() const;
  [[nodiscard]] Type getType() const;
  [[nodiscard]] QuantInfo getQuantInfo(size_t idx = 0) const;

protected:
  explicit Argument(detail::ArgumentImpl *i);

private:
  detail::ArgumentImpl *impl;
};

// MLIR Type
class XQ_CAPI_EXPORTED Type {
  friend Argument;

public:
  Type() = delete;
  Type(const Type &) = default;
  Type &operator=(const Type &) = default;
  ~Type() = default;

  [[nodiscard]] ude::Type getTypeTag() const;
  [[nodiscard]] ude::ArrayRef<int64_t> getShape() const;
  [[nodiscard]] ude::ArrayRef<int64_t> getMaxShape() const;
  [[nodiscard]] int64_t getByteSize() const;
  [[nodiscard]] ude::Type getElementType() const;
  [[nodiscard]] std::vector<Type> getNestedType() const;

protected:
  explicit Type(detail::TypeImpl *i);

private:
  detail::TypeImpl *impl;
};

// Quant Info
class XQ_CAPI_EXPORTED QuantInfo {
  friend class Argument;

public:
  QuantInfo() = delete;
  QuantInfo(const QuantInfo &) = default;
  QuantInfo &operator=(const QuantInfo &) = default;
  ~QuantInfo() = default;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] ude::ArrayRef<float> getScales() const;
  [[nodiscard]] ude::ArrayRef<int32_t> getZeros() const;
  [[nodiscard]] bool hasAxis() const;
  [[nodiscard]] int32_t getAxis() const;

protected:
  explicit QuantInfo(detail::QuantInfoImpl *i);

private:
  detail::QuantInfoImpl *impl;
};

} // namespace mlir::hbdk::pub::xxq
