#pragma once

#include "Compiler.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace ude {

class UDE_NODISCARD Status {
public:
  /// If isSuccess is true a `success` result is generated, otherwise a
  /// 'failure' result is generated.
  static Status success(bool isSuccess = true, const char *msg = nullptr) { return {isSuccess, msg}; }

  /// If isFailure is true a `failure` result is generated, otherwise a
  /// 'success' result is generated.
  static Status failure(bool isFailure = true, const char *msg = {}) { return {!isFailure, msg}; }

  /// Returns true if the provided Status corresponds to a failure value.
  UDE_NODISCARD bool failed() const { return !isSuccess; }

  /// Returns true if the provided Status corresponds to a success value.
  UDE_NODISCARD bool succeeded() const { return isSuccess; }

  /// Get message
  UDE_NODISCARD std::string getMsg() const { return (msg == nullptr) ? "" : std::string{msg}; }

  /// Convert to bool
  // NOLINTNEXTLINE
  operator bool() const { return isSuccess; }

  Status(const Status &other) { copyFrom(other); }

  Status &operator=(const Status &other) {
    if (&other == this) {
      return *this;
    }
    free(msg);
    copyFrom(other);
    return *this;
  }

  Status(Status &&other) noexcept {
    msg = other.msg;
    isSuccess = other.isSuccess;
    other.msg = nullptr;
  }

  Status &operator=(Status &&other) noexcept {
    if (this != &other) {
      free(msg);
      msg = other.msg;
      isSuccess = other.isSuccess;
      other.msg = nullptr;
    }
    return *this;
  }

  ~Status() { free(msg); }

private:
  void copyFrom(const Status &other) {
    if (other.msg != nullptr) {
      auto len = strlen(other.msg) + 1;
      msg = static_cast<char *>(malloc(len));
      memcpy(msg, other.msg, len);
    }
    isSuccess = other.isSuccess;
  }

  Status(bool isSuccess, const char *msg) : isSuccess(isSuccess) {
    if (msg != nullptr) {
      this->msg = static_cast<char *>(malloc(strlen(msg) + 1));
      memcpy(this->msg, msg, strlen(msg) + 1);
    }
  }

  bool isSuccess{};
  char *msg = nullptr;
};

/// Utility function to generate a Status. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline Status success(bool isSuccess = true, const char *msg = {}) { return Status::success(isSuccess, msg); }

/// Utility function to generate a Status. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
inline Status failure(bool isFailure = true, const char *msg = {}) { return Status::failure(isFailure, msg); }

/// Utility function that returns true if the provided Status corresponds
/// to a failure value.
inline bool failed(const Status &result) { return result.failed(); }

/// Utility function that returns true if the provided Status corresponds
/// to a success value.
inline bool succeeded(const Status &result) { return result.succeeded(); }

} // namespace ude
