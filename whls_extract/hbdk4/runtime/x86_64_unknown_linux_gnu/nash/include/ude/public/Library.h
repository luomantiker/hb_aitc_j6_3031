#pragma once

#include "Common.h"
#include "Compiler.h"
#include "Kernel.h"
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ude {

class UdeLibraryInit;

class UDE_EXPORTED UdeLibrary {
  friend UdeLibraryInit;

public:
  using KernelsType = std::vector<std::unique_ptr<Kernel>>;
  /// for kernels need config
  template <size_t outs, typename FuncConfig, typename FuncImpl, typename... Extras>
  std::enable_if_t<is_function_v<FuncImpl>, UdeLibrary &> def(const char *schema, FuncConfig config, FuncImpl impl,
                                                              const Extras &...args) {
    std::unique_ptr<Kernel> kernel =
        std::make_unique<KernelImpl<outs, std::remove_pointer_t<FuncConfig>, std::remove_pointer_t<FuncImpl>>>(
            key, config, impl, name(schema), file(file_), line(line_), args...);
    addKernel(std::move(kernel));
    return *this;
  }

  /// for kernels don't need config
  template <size_t outs, typename FuncImpl, typename... Extras>
  UdeLibrary &def(const char *schema, FuncImpl impl, const Extras &...args) {
    std::unique_ptr<Kernel> kernel =
        std::make_unique<KernelImpl<outs, std::remove_pointer_t<FuncImpl>, std::remove_pointer_t<FuncImpl>>>(
            key, impl, name(schema), file(file_), line(line_), args...);
    addKernel(std::move(kernel));
    return *this;
  }

  UDE_NODISCARD const KernelsType *getKernels() const { return &keepAlives; }

protected:
  UdeLibrary(const char *fn, DispatchKey key, size_t line) : file_(fn), key(key), line_(line) {}

private:
  void addKernel(std::unique_ptr<Kernel> &&kernel) { keepAlives.push_back(std::move(kernel)); }

  KernelsType keepAlives;
  const char *file_ = nullptr;
  // Class who defined kernel
  DispatchKey key;
  size_t line_{};
};

class UdeLibraryInit {
  using InitFuncType = void (*)(UdeLibrary &);

public:
  UdeLibraryInit(InitFuncType initFunc, DispatchKey key, const char *file, size_t line) : lib(file, key, line) {
    initFunc(lib);
  }

  UdeLibrary lib;
};

inline size_t combineHash(size_t h1, size_t h2, size_t h3, size_t h4, size_t h5) {
  h1 ^= h2 + 0x9e3779b9 + (h1 << 6U) + (h1 >> 2U);
  h1 ^= h3 + 0xfe3779cd + (h1 << 6U) + (h1 >> 2U);
  h1 ^= h4 + 0x9e3779ba + (h1 << 6U) + (h1 >> 2U);
  h1 ^= h5 + 0x9e3779de + (h1 << 6U) + (h1 >> 2U);
  return h1;
}

inline size_t getABINumber() {
  return combineHash(sizeof(Kernel), sizeof(TensorRef), sizeof(VectorRef), sizeof(Variable), sizeof(UdeLibrary));
}

#define UDE_LIBRARY(name, key)                                                                                         \
  static void UDE_CONCAT(UDE_LIBRARY_init_, name)(ude::UdeLibrary & m);                                                \
  const ude::UdeLibraryInit UDE_CONCAT(UDE_LIBRARY_static_init_, name)(UDE_CONCAT(UDE_LIBRARY_init_, name),            \
                                                                       ude::DispatchKey::key, __FILE__, __LINE__);     \
  extern "C" {                                                                                                         \
  UDE_EXPORTED const ude::UdeLibrary *UDE_LIBRARY_MAIN(uint64_t &major, uint64_t &minor) {                             \
    major = UDE_MAJOR_VERSION;                                                                                         \
    minor = UDE_MINOR_VERSION;                                                                                         \
    return &UDE_CONCAT(UDE_LIBRARY_static_init_, name).lib;                                                            \
  }                                                                                                                    \
  UDE_EXPORTED size_t UDE_ABI_CHECK() { return ude::getABINumber(); }                                                  \
  }                                                                                                                    \
  void UDE_CONCAT(UDE_LIBRARY_init_, name)(ude::UdeLibrary & m)

#define UDE_LIBRARY_INTERNAL(name, key)                                                                                \
  static void UDE_CONCAT(UDE_LIBRARY_init_, name)(ude::UdeLibrary & m);                                                \
  const ude::UdeLibraryInit UDE_CONCAT(UDE_LIBRARY_static_init_, name)(UDE_CONCAT(UDE_LIBRARY_init_, name),            \
                                                                       ude::DispatchKey::key, __FILE__, __LINE__);     \
  void UDE_CONCAT(UDE_LIBRARY_init_, name)(ude::UdeLibrary & m)

} // namespace ude
