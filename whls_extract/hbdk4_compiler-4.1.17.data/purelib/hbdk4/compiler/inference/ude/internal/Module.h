#pragma once

#include "ude/public/Common.h"
#include "ude/public/Compiler.h"
#include "ude/public/Library.h"
#include "ude/public/Status.h"
#include <cstddef>
#include <dlfcn.h>
#include <string>

namespace ude {

class Module {
public:
  using dl_handle = void *;

  /// Empty string to load from current process
  Module(const std::string &path) : path(path) { // NOLINT
  }

  Status load(bool checkAbi = false) {
    // nullptr for the current process
    dl_handle handle = dlopen(path.empty() ? nullptr : path.c_str(), RTLD_LAZY);

    if (handle == nullptr) {
      std::string errMsg = std::string("dlopen failed, due to ") + dlerror();
      return Status::failure(true, errMsg.c_str());
    }

    this->handle = handle;

    auto f = reinterpret_cast<UdeLibrary *(*)(uint64_t &, uint64_t &)>(dlsym(handle, "UDE_LIBRARY_MAIN"));

    if (f == nullptr) {
      return Status::failure(true, "No found symbol UDE_LIBRARY_MAIN in this shared library.");
    }

    uint64_t major;
    uint64_t minor;
    auto *lib = f(major, minor);

    if (UDE_VERSION_HEX(major, minor) < UDE_MIN_REQUIRED) {
      return Status::failure(true, ("Don't meet the minimum version library requirement." +
                                    std::string("The minimum required version is " + std::to_string(UDE_MIN_REQUIRED)))
                                       .c_str());
    }

    if (checkAbi) {
      auto getLibAbiNumFunc = reinterpret_cast<size_t (*)()>(dlsym(handle, "UDE_ABI_CHECK"));
      if (getLibAbiNumFunc == nullptr) {
        return Status::failure(true, "No found symbol UDE_ABI_CHECK in this shared library.");
      }
      if (getLibAbiNumFunc() != getABINumber()) {
        return Status::failure(
            true, "The ABI version loaded by ude is incompatible with the ABI version used by the current library.");
      }
    }

    this->lib = lib;
    return Status::success();
  }

  // handle release by dispatcher
  ~Module() {
    if (handle != nullptr) {
      dlclose(handle);
    }
  }

  UDE_NODISCARD const UdeLibrary *accessLib() const { return lib; }

  UDE_NODISCARD const std::string &getPath() const { return path; }

  UDE_NODISCARD dl_handle getHandle() const { return handle; }

private:
  std::string path{};
  UdeLibrary *lib = nullptr;
  dl_handle handle = nullptr;
};

} // namespace ude
