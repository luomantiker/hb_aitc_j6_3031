#pragma once

#include "ude/public/ArrayRef.h"
#include "ude/public/Common.h"
#include "ude/public/Compiler.h"
#include "ude/public/Kernel.h"
#include "ude/public/Types.h"
#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>

namespace ude {

template <size_t outs> class Tag {};

// In order to replace python and hbrt conveniently
struct Schema {
  Schema() = default;
  template <size_t outs, typename ImplRet, typename... ImplArgs>
  Schema(std::string name, Tag<outs> tag, ImplRet (*func)(ImplArgs...)) {
    using funcType = ImplRet(ImplArgs...);
    std::unique_ptr<ude::Kernel> kernel =
        std::make_unique<ude::KernelImpl<outs, funcType, funcType>>(ude::DispatchKey::CUSTOM, func);
    nameAndNs = std::move(name);
    inTypeIds = kernel->inTypes;
    outTypeIds = kernel->outTypes;
    outNums = kernel->outs;
    inNums = kernel->ins;
  }

  explicit Schema(std::string str, std::vector<Type> outTypes, std::vector<Type> inTypes) : nameAndNs(std::move(str)) {
    outNums = outTypes.size();
    inNums = inTypes.size();
    memcpy(outTypeIds.data(), outTypes.data(), outNums * sizeof(Type));
    memcpy(inTypeIds.data(), inTypes.data(), inNums * sizeof(Type));
  }

  explicit Schema(const Kernel *kernel) {
    nameAndNs = kernel->name;
    inTypeIds = kernel->inTypes;
    outTypeIds = kernel->outTypes;
    outNums = kernel->outs;
    inNums = kernel->ins;
  }

  UDE_NODISCARD ude::Type getOutputType(size_t idx) const {
    assert(idx < outNums && "Try to access output's type greater than Number of outs");
    return outTypeIds[idx];
  }

  UDE_NODISCARD ude::Type getInputType(size_t idx) const {
    assert(idx < inNums && "Try to access input's type greater than Number of inputs");
    return inTypeIds[idx];
  }

  UDE_NODISCARD std::string getNsAndName() const { return nameAndNs; }

  ArrayRef<ude::Type> getInputTypes() const { return {inTypeIds.begin(), inNums}; }

  ArrayRef<ude::Type> getOutputTypes() const { return {outTypeIds.begin(), outNums}; }

  UDE_NODISCARD std::string getSchema() const {
    std::string sig(nameAndNs);
    sig += "(";

    if (inNums > 0) {
      for (size_t i = 0U; i < inNums - 1; ++i) {
        sig += toString(inTypeIds[i]) + std::string(", ");
      }
      sig += std::string(toString(inTypeIds[inNums - 1]));
    }
    sig += ") -> (";
    if (outNums > 0) {
      for (size_t i = 0U; i < outNums - 1; ++i) {
        sig += toString(outTypeIds[i]) + std::string(", ");
      }
      sig += std::string(toString(outTypeIds[outNums - 1]));
    }
    sig += ")";
    return sig;
  }

  UDE_NODISCARD std::string getInputSchema() const {
    std::string sig;
    sig += "(";

    if (inNums > 0) {
      for (size_t i = 0U; i < inNums - 1; ++i) {
        sig += toString(inTypeIds[i]) + std::string(", ");
      }
      sig += std::string(toString(inTypeIds[inNums - 1]));
    }
    sig += ")";
    return sig;
  }

  std::array<ude::Type, 5> outTypeIds{};
  std::array<ude::Type, 40> inTypeIds{};
  std::string nameAndNs;
  size_t outNums = 0;
  size_t inNums = 0;
};

} // namespace ude
