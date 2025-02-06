#pragma once

#include "ArrayRef.h"
#include "Caster.h"
#include "Common.h"
#include "Compiler.h"
#include "Status.h"
#include "Types.h"
#include "Variable.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ude {

struct Kernel {

  Kernel() = default;
  /// Function name, ns + name
  const char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

  // User-specified documentation string
  const char *doc = nullptr;

  ArrayRef<ude::Type> getInputTypes() const { return {inTypes.begin(), ins}; }

  ArrayRef<ude::Type> getOutputTypes() const { return {outTypes.begin(), outs}; }

  std::array<Type, 5> outTypes{};
  std::array<Type, 40> inTypes{};

  /// Backend
  const char *backend = nullptr;

  /// defined file name
  const char *file = nullptr;

  /// defined line number
  size_t line = 0;

  /// who defined this kernel
  DispatchKey key = DispatchKey::REFERENCE;

  /// List of arguments's descr
  std::vector<const char *> args;

  /// Storage for the wrapped kernel function pointer, if any
  void *kernelImpl = nullptr;

  /// Storage for the wrapped kernel's config function pointer, if any
  void *kernelConfig = nullptr;

  /// Number of outputs
  size_t outs{0};
  size_t ins{0};

  Status invoke(MutableArrayRef<Variable *> outArgs, ArrayRef<Variable *> inArgs) const {
    return invokeImpl(outArgs, inArgs);
  }

  Status invoke(MutableArrayRef<Variable> outArgs, ArrayRef<Variable> inArgs) const {
    std::vector<Variable *> uArgs;
    uArgs.reserve(outs);
    for (auto &out : outArgs) {
      uArgs.push_back(&out);
    }

    std::vector<Variable *> uIArgs;
    uArgs.reserve(inArgs.size());
    for (const auto &in : inArgs) {
      uIArgs.push_back(const_cast<Variable *>(&in));
    }

    return invokeImpl(uArgs, uIArgs);
  }

  Status config(MutableArrayRef<Variable> outArgs, ArrayRef<Variable> inArgs) const {
    std::vector<Variable *> uArgs;
    uArgs.reserve(outs);
    for (auto &out : outArgs) {
      uArgs.push_back(&out);
    }

    std::vector<Variable *> uIArgs;
    uArgs.reserve(inArgs.size());
    for (const auto &in : inArgs) {
      uIArgs.push_back(const_cast<Variable *>(&in));
    }

    return configImpl(uArgs, uIArgs);
  }

  Status config(MutableArrayRef<Variable *> outs, ArrayRef<Variable *> ins) const { return configImpl(outs, ins); }

  Status config(MutableArrayRef<Variable *> stack) const {
    return configImpl(stack.take_front(outs), stack.drop_front(outs));
  }
  Status invoke(MutableArrayRef<Variable *> stack) const {
    return invokeImpl(stack.take_front(outs), stack.drop_front(outs));
  }
  // NOTE:
  // LLVM code standard: Provide a Virtual Method Anchor for Classes in Headers
  // This prevent vtable symbol existing in all files that uses this header
  //
  // This also work around some strange bugs in static lib in ARM build (See MR2061)
  virtual void anchor() {}

  virtual ~Kernel() = default;

private:
  virtual Status configImpl(MutableArrayRef<Variable *> outs, ArrayRef<Variable *> ins) const = 0;
  virtual Status invokeImpl(MutableArrayRef<Variable *> outs, ArrayRef<Variable *> ins) const = 0;
};

template <typename T, typename SFINAE = void> struct ProcessAttribute;

#define PROCESS(type)                                                                                                  \
  template <> struct ProcessAttribute<type> {                                                                          \
    static void init(const type &n, Kernel *r) { r->type = n.value; }                                                  \
  };

PROCESS(name)
PROCESS(doc)
PROCESS(backend)
PROCESS(file)
PROCESS(line)
#undef PROCESS

template <> struct ProcessAttribute<arg> {
  static void init(const arg &n, Kernel *r) { r->args.push_back(n.value); }
};

// process(arg)
template <typename... Args> struct ProcessAttributes {
  static void init(const Args &...args, Kernel *r) {
    using expander = int[]; // NOLINT
    (void)expander{0, ((void)ProcessAttribute<typename std::decay_t<Args>>::init(args, r), 0)...};
  }
};

template <size_t outs, typename ConfigFunc, typename ImplFunc> class KernelImpl;

/// Kernel wrapper class.
/// Only support raw function, it doesn't support lambda and class member function now.
/// We suppose kernel needs two functions, one is implement, and the other one is config.
/// If the config function doesn't exist, we just need use implement function's prototype
/// to replace config's function type. By this means, we can share the same derived class.
/// In C++17, this class template parameters can be deduce from construct function. However,
/// it can't work in C++14 and C++11.
template <size_t outNum, typename RetConfig, typename... ArgsConfig, typename RetImpl, typename... ArgsImpl>
class KernelImpl<outNum, RetConfig(ArgsConfig...), RetImpl(ArgsImpl...)> : public Kernel {
public:
  using ImplFunctionType = RetImpl (*)(ArgsImpl...);
  using ConfigFunctionType = RetConfig (*)(ArgsConfig...);
  using CastIn = ArgLoader<outNum, ArgsImpl...>;
  using CastOut = MakeCaster<RetImpl>;
  /// outNum = 2, {0, 1} -> outputs, {2, 3, ...} -> inputs
  using OutputsSeq = std::make_index_sequence<outNum>;
  using InputSeq = typename offset_sequence<outNum, std::make_index_sequence<sizeof...(ArgsImpl) - outNum>>::type;

  static_assert(outNum < 5UL, "kernel's wrapper don't support output nums greater than 5 now.");
  static_assert((sizeof...(ArgsImpl) - outNum) < 20UL,
                "kernel's wrapper don't support input nums greater than 20 now.");
  // TODO(Weiguozhao)
  /// When use ude::Status replace hbtl::LogicalResult, make the static_assert work.
  // static_assert(
  //     std::is_same_v<RetImpl, Status>,
  //     "KernelImpl wrapper wants impl's return type is Status, because it can check the call go well or not and trace
  //     debug info");
  // static_assert(std::is_same_v<RetConfig, Status>, "KernelImpl wrapper wants config's return type is"
  //                                                       "Status, because it can check the call go well or not and
  //                                                       trace debug info");

  template <typename... Extras>
  KernelImpl(DispatchKey key, ConfigFunctionType fc, ImplFunctionType fi, const Extras &...args) {
    this->outs = outNum;
    this->ins = sizeof...(ArgsImpl) - outNum;
    this->key = key;
    ProcessAttributes<Extras...>::init(args..., this);
    this->outTypes = CastIn::genOutIds(OutputsSeq{});
    this->inTypes = CastIn::genInIds(InputSeq{});
    this->kernelImpl = reinterpret_cast<void *>(fi);
    this->kernelConfig = reinterpret_cast<void *>(fc);
  }

  template <typename... Extras> explicit KernelImpl(DispatchKey key, ImplFunctionType fi, const Extras &...args) {
    this->outs = outNum;
    this->ins = sizeof...(ArgsImpl) - outNum;
    this->key = key;
    ProcessAttributes<Extras...>::init(args..., this);
    this->outTypes = CastIn::genOutIds(OutputsSeq{});
    this->inTypes = CastIn::genInIds(InputSeq{});
    this->kernelImpl = reinterpret_cast<void *>(fi);
  }

  ~KernelImpl() override = default;

private:
  Status configImpl(MutableArrayRef<Variable *> outs, ArrayRef<Variable *> ins) const override {
    assert(this->kernelConfig != nullptr && "Kernel has't config function");
    assert(sizeof...(ArgsConfig) == (ins.size() + outs.size()) &&
           "Kernel config function's number of args is not equal to defined");
    ArgLoader<outNum, ArgsConfig...> argsConvert;
    auto loadStatus = argsConvert.loadInArgs(ins);
    if (loadStatus.failed()) {
      auto msg = "(Config invoke) The input params: " + loadStatus.getMsg();
      return Status::failure(true, msg.c_str());
    }

    loadStatus = argsConvert.loadOutArgs(outs);
    if (loadStatus.failed()) {
      auto msg = "(Config invoke) The output params: " + loadStatus.getMsg();
      return Status::failure(true, msg.c_str());
    }

    auto status =
        std::move(argsConvert).template call<RetConfig>(reinterpret_cast<ConfigFunctionType>(this->kernelConfig));

    if (!status) {
      if UDE_CONSTEXPR_IF (has_msg_trait_v<RetConfig>) {
        return Status::failure(true,
                               (std::string("kernel ") + name + " config function call failure! " + "It's defined at " +
                                file + ":" + std::to_string(line) + ". " + "Due to " + status.getMsg())
                                   .c_str());
      } else {
        return Status::failure(true, (std::string("kernel ") + name + " config function call failure!" +
                                      "It's defined at " + file + " " + std::to_string(line))
                                         .c_str());
      }
    }

    auto syncStatus = argsConvert.syncOutArgs(outs);
    if (syncStatus.failed()) {
      return syncStatus;
    }
    return Status::success();
  }

  Status invokeImpl(MutableArrayRef<Variable *> outs, ArrayRef<Variable *> ins) const override {
    assert(this->kernelImpl != nullptr && "Kernel has't implement function");
    assert(sizeof...(ArgsImpl) == (outs.size() + ins.size()) &&
           "Kernel implement function's number of args is not equal to defined");
    ArgLoader<outNum, ArgsImpl...> argsConvert;
    auto loadStatus = argsConvert.loadInArgs(ins);
    if (loadStatus.failed()) {
      auto msg = "(Impl invoke) The input params: " + loadStatus.getMsg();
      return Status::failure(true, msg.c_str());
    }

    loadStatus = argsConvert.loadOutArgs(outs);
    if (loadStatus.failed()) {
      auto msg = "(Impl invoke) The output params: " + loadStatus.getMsg();
      return Status::failure(true, msg.c_str());
    }

    auto status = std::move(argsConvert).template call<RetImpl>(reinterpret_cast<ImplFunctionType>(this->kernelImpl));

    if (!status) {
      if UDE_CONSTEXPR_IF (has_msg_trait_v<RetImpl>) {
        return Status::failure(true,
                               (std::string("kernel ") + name + " impl function call failure!" + "It's defined at " +
                                file + " " + std::to_string(line) + ". " + "Due to " + status.getMsg())
                                   .c_str());
      } else {
        return Status::failure(true, (std::string("kernel ") + name + " impl function call failure!" +
                                      "It's defined at " + file + " " + std::to_string(line))
                                         .c_str());
      }
    }

    auto syncStatus = argsConvert.syncOutArgs(outs);
    if (syncStatus.failed()) {
      return syncStatus;
    }
    return Status::success();
  }
};

} // namespace ude
