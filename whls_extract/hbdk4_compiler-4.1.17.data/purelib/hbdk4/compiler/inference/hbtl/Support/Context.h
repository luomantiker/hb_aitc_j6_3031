/// manage hbtl configurations
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/Support/Compiler.h"

HBTL_NAMESPACE_BEGIN {

  namespace detail {
  class ContextImpl; // IWYU pragma: keep
  }                  // namespace detail

  enum class LogLevel : size_t { INFO, WARN, CRITICAL };

  /// a singleton context for hbtl
  class HBTL_EXPORTED Context {
  public:
    ~Context() = default;

    /// get the process independent hbtl context
    static Context *get();

    /// when trace point is set, kernels are run with traceEn true.
    void setTracePoint(ArrayRef<int64_t> point = {});
    ArrayRef<int64_t> getTracePoint();

    /// when set to false, verification bypassed at each invocation
    void setEnableVerify(bool enable);
    bool getEnableVerify();

    int64_t getNumThreads();
    void setNumThreads(int64_t num);
    bool getVerbose();

    /// get dynamic assigned typeid of given type string.
    size_t getTypeId(const char *typeStr);
    /// get type string of a typeid
    const char *getTypeStr(size_t typeId);

    void log(LogLevel level, const char *msg);
    void info(const char *msg);
    void critical(const char *msg);
    void warn(const char *msg);

    /// register custom logger callback. arguments are: log level, message and user data
    void regLogger(std::function<void(LogLevel, const char *, void *)>, void *);

  private:
    Context(); // singleton
    std::unique_ptr<detail::ContextImpl> impl;
  };
}
HBTL_NAMESPACE_END
