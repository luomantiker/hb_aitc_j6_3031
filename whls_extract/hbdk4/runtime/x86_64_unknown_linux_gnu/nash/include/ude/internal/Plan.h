#pragma once
#include "ude/internal/Dispatcher.h"
#include "ude/internal/Schema.h"
#include "ude/public/ArrayRef.h"
#include "ude/public/Common.h"
#include "ude/public/Types.h"
#include "ude/public/Variable.h"
#include <cstddef>

namespace ude {

class Plan;

class Task {
public:
  ude::Status launch() { return kernel->invoke(stack); }

  ude::Status infer() { return kernel->config(stack); }

  [[nodiscard]] const ude::Kernel *touchKernel() const { return kernel; }

  ude::ArrayRef<ude::Variable *> touchVariable() const { return stack; }
  explicit operator bool() { return kernel != nullptr; }

protected:
  friend class Plan;
  Task(ude::Kernel *kernel, ude::MutableArrayRef<ude::Variable> &&args) : kernel(kernel) {
    for (auto &v : args) {
      stack.push_back(&v);
    }
  }

  Task() { kernel = nullptr; }

  ude::Kernel *kernel;
  std::vector<ude::Variable *> stack{}; // reference from plan
};

class Plan {
public:
  Plan(ude::Schema schema, std::vector<ude::Variable> args) : schema(std::move(schema)), args(std::move(args)) {}
  Plan(const std::string &nsAndName, std::vector<ude::Variable> outArgs, std::vector<ude::Variable> inArgs) {
    // old hbm's signatures is ns + name + type, we only care ns + name now.
    const auto pos = nsAndName.find_first_of('(');
    auto actualNsAndName = nsAndName.substr(0, pos);
    std::vector<Type> outTypes;
    outTypes.reserve(outArgs.size());
    for (auto &var : outArgs) {
      outTypes.push_back(var.getType());
    }
    std::vector<Type> inTypes;
    inTypes.reserve(inArgs.size());
    for (auto &var : inArgs) {
      inTypes.push_back(var.getType());
    }
    schema = ude::Schema(actualNsAndName, outTypes, inTypes);
    for (auto &oArgs : outArgs) {
      args.push_back(std::move(oArgs));
    }
    for (auto &iArgs : inArgs) {
      args.push_back(std::move(iArgs));
    }
  }

  Task createTask(const ude::Dispatcher &disp, ude::DispatchKey key, bool enableFallback = true) {
    auto candKernels = disp.findWithSchema(schema);
    if (candKernels.empty()) {
      return Task{};
    }

    ude::Kernel *kernel = nullptr;
    ude::Kernel *generalKernel = nullptr;
    for (size_t i = candKernels.size(); i > 0; --i) {
      if (candKernels[i - 1]->key == key) {
        kernel = candKernels[i - 1];
        break;
      }

      if (enableFallback && candKernels[i - 1]->key == ude::DispatchKey::REFERENCE) {
        generalKernel = candKernels[i - 1];
      }
    }

    if (kernel == nullptr && generalKernel == nullptr) {
      return Task{};
    }

    if (kernel != nullptr) {
      return Task{kernel, args};
    } else {
      return Task{generalKernel, args};
    }
  }

  UDE_NODISCARD ude::Schema getSchema() const { return schema; }

private:
  ude::Schema schema;
  std::vector<ude::Variable> args;
};

} // namespace ude
