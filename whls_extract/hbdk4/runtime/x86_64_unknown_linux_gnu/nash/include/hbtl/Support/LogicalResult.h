/// forked from mlir/Support/LogicalResult.h
#pragma once

#include "hbtl/ADT/Optional.h"
#include "hbtl/ADT/string_view.h"
#include "hbtl/Support/Compiler.h"
#include <cassert>
#include <string>
#include <utility>

HBTL_NAMESPACE_BEGIN {

  class HBTL_NODISCARD LogicalResult {
  public:
    /// If isSuccess is true a `success` result is generated, otherwise a
    /// 'failure' result is generated.
    static LogicalResult success(bool isSuccess = true, hbtl::string_view info = {}) { return {isSuccess, info}; }

    /// If isFailure is true a `failure` result is generated, otherwise a
    /// 'success' result is generated.
    static LogicalResult failure(bool isFailure = true, hbtl::string_view info = {}) { return {!isFailure, info}; }

    /// Returns true if the provided LogicalResult corresponds to a success value.
    HBTL_NODISCARD bool LogicalResultToBool() const { return isSuccess; }

    /// Returns true if the provided LogicalResult corresponds to a failure value.
    HBTL_NODISCARD bool failed() const { return !isSuccess; }

    /// Returns true if the provided LogicalResult corresponds to a success value.
    HBTL_NODISCARD bool succeeded() const { return isSuccess; }

    /// Get message
    HBTL_NODISCARD hbtl::string_view getInfo() const { return info; }

    HBTL_NODISCARD const std::string &getMsg() const { return info; }

    // NOLINTNEXTLINE
    operator bool() const { return isSuccess; }

  private:
    LogicalResult(bool isSuccess, hbtl::string_view info) : isSuccess(isSuccess), info(info) {}

    bool isSuccess;
    std::string info;
  };

  /// Utility function to generate a LogicalResult. If isSuccess is true a
  /// `success` result is generated, otherwise a 'failure' result is generated.
  inline LogicalResult success(bool isSuccess = true, hbtl::string_view info = {}) {
    return LogicalResult::success(isSuccess, info);
  }

  /// Utility function to generate a LogicalResult. If isFailure is true a
  /// `failure` result is generated, otherwise a 'success' result is generated.
  inline LogicalResult failure(bool isFailure = true, hbtl::string_view info = {}) {
    return LogicalResult::failure(isFailure, info);
  }

  /// Utility function that returns true if the provided LogicalResult corresponds
  /// to a success value.
  inline bool LogicalResultToBool(LogicalResult result) { return result.LogicalResultToBool(); }

  /// Utility function that returns true if the provided LogicalResult corresponds
  /// to a failure value.
  inline bool failed(LogicalResult result) { return result.failed(); }

  /// Utility function that returns true if the provided LogicalResult corresponds
  /// to a success value.
  inline bool succeeded(LogicalResult result) { return result.succeeded(); }

  /// This class provides support for representing a failure result, or a valid
  /// value of type `T`. This allows for integrating with LogicalResult, while
  /// also providing a value on the success path.
  template <typename T> class HBTL_NODISCARD FailureOr : public hbtl::optional<T> {
  public:
    /// Allow constructing from a LogicalResult. The result *must* be a failure.
    /// Success results should use a proper instance of type `T`.
    FailureOr(LogicalResult result) : info(result.getInfo()) {
      assert(failed(result) && "success should be constructed with an instance of 'T'");
    }
    FailureOr() : FailureOr(failure()) {}
    FailureOr(T &&y) : hbtl::optional<T>(std::forward<T>(y)) {}
    FailureOr(const T &y) : hbtl::optional<T>(y) {}
    template <typename U, std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
    FailureOr(const FailureOr<U> &other)
        : hbtl::optional<T>(failed(other) ? hbtl::optional<T>() : hbtl::optional<T>(*other)) {}

    operator LogicalResult() const { return failure(!this->has_value(), this->getInfo()); }

    /// Get message
    HBTL_NODISCARD hbtl::string_view getInfo() const { return info; }

  private:
    /// Hide the bool conversion as it easily creates confusion.
    using hbtl::optional<T>::operator bool;
    using hbtl::optional<T>::has_value;

    std::string info;
  };

  /// Wrap a value on the success path in a FailureOr of the same value type.
  template <typename T, typename = std::enable_if_t<!std::is_convertible<T, bool>::value>> inline auto success(T && t) {
    return FailureOr<std::decay_t<T>>(std::forward<T>(t));
  }

  /// This class represents success/failure for parsing-like operations that find
  /// it important to chain together failable operations with `||`.  This is an
  /// extended version of `LogicalResult` that allows for explicit conversion to
  /// bool.
  ///
  /// This class should not be used for general error handling cases - we prefer
  /// to keep the logic explicit with the `LogicalResultToBool`/`failed` predicates.
  /// However, traditional monadic-style parsing logic can sometimes get
  /// swallowed up in boilerplate without this, so we provide this for narrow
  /// cases where it is important.
  ///
  class HBTL_NODISCARD ParseResult : public LogicalResult {
  public:
    ParseResult(LogicalResult result = success()) : LogicalResult(std::move(result)) {}

    /// Failure is true in a boolean context.
    explicit operator bool() const { return failed(); }
  };
}
HBTL_NAMESPACE_END
