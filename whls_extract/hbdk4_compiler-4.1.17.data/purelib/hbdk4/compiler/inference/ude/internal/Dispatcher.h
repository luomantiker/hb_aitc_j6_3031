#pragma once

#include "ude/internal/Module.h"
#include "ude/internal/Schema.h"
#include "ude/public/ArrayRef.h"
#include "ude/public/Compiler.h"
#include "ude/public/Kernel.h"
#include "ude/public/Library.h"
#include "ude/public/Status.h"
#include "ude/public/Types.h"
#include <map>

namespace ude {

class Dispatcher {
public:
  Status remove(Kernel *kernel) {
    const auto *kernelName = kernel->name;
    auto nsAndName = parseNsAndName(kernelName);
    auto nsIter = kernelManagers.find(nsAndName.first);
    if (nsIter == kernelManagers.end()) {
      return Status::failure(
          true, ("Dispatcher kernel manager don't count the namespace " + std::string(nsAndName.first)).c_str());
    }
    auto nameIter = nsIter->second.find(nsAndName.second);
    if (nameIter == nsIter->second.end()) {
      return Status::failure(
          true, ("Dispatcher kernel manager don't count the function " + std::string(nsAndName.second)).c_str());
    }
    auto &kernelVec = nameIter->second;

    for (auto it = kernelVec.begin(); it != kernelVec.end();) {
      const auto *kv = *it;
      if ((kv->getInputTypes() == kernel->getInputTypes() && kv->getOutputTypes() == kernel->getOutputTypes()) &&
          (kv->key == kernel->key)) {
        it = kernelVec.erase(it);
      } else {
        ++it;
      }
    }
    return Status::success();
  }

  Status insert(Kernel *kernel) {
    const auto *kernelName = kernel->name;
    auto nsAndName = parseNsAndName(kernelName);
    if (nsAndName.first.empty()) {
      return Status::failure(true, "ns parse error");
    }

    auto nsIter = kernelManagers.find(nsAndName.first);
    if (nsIter == kernelManagers.end()) {
      auto resNsInsert = kernelManagers.insert({nsAndName.first, std::map<std::string, std::vector<Kernel *>>()});
      if (!resNsInsert.second) {
        return Status::failure(true, "insert kernel failed");
      }
      nsIter = resNsInsert.first;
    }

    auto nameIter = nsIter->second.find(nsAndName.second);
    if (nameIter == nsIter->second.end()) {
      auto resNameInsert = nsIter->second.insert({nsAndName.second, std::vector<Kernel *>()});
      if (!resNameInsert.second) {
        return Status::failure();
      }
      nameIter = resNameInsert.first;
    }

    nameIter->second.push_back(kernel);
    return Status::success();
  }

  UDE_NODISCARD ArrayRef<Kernel *> find(const std::string &ns, const std::string &name) const {
    if (kernelManagers.count(ns) == 0) {
      return {};
    }
    auto nsClass = kernelManagers.find(ns);

    if (nsClass->second.count(name) == 0) {
      return {};
    }

    auto it = nsClass->second.find(name);

    assert(it != nsClass->second.end() && "It should have value");
    return {it->second};
  }

  UDE_NODISCARD ArrayRef<Kernel *> find(const std::string &schema) const {
    auto nsAndName = parseNsAndName(schema.c_str());
    return find(nsAndName.first, nsAndName.second);
  }

  UDE_NODISCARD std::vector<Kernel *> findWithSchema(const Schema &schema) const {
    auto firstRes = find(schema.getNsAndName());
    std::vector<Kernel *> secondRes;
    for (auto *kernel : firstRes) {
      if (kernel->getOutputTypes() == schema.getOutputTypes() && kernel->getInputTypes() == schema.getInputTypes()) {
        secondRes.push_back(kernel);
      }
    }
    return secondRes;
  }

  Status load(const Module &m) {
    const auto *lib = m.accessLib();
    for (const auto &kernel : *lib->getKernels()) {
      auto status = insert(kernel.get());
      if (failed(status)) {
        return Status::failure(true, "insert kernel failed from module");
      }
    }
    return Status::success();
  }

  Status load(const UdeLibrary *lib) {
    for (const auto &kernel : *lib->getKernels()) {
      auto status = insert(kernel.get());
      if (failed(status)) {
        return Status::failure(true, "insert kernel failed");
      }
    }
    return Status::success();
  }

  Status unload(const Module &m) {
    const auto *lib = m.accessLib();
    for (const auto &kernel : *lib->getKernels()) {
      auto status = remove(kernel.get());
      if (failed(status)) {
        return Status::failure(true, "remove kernel failed");
      }
    }
    return Status::success();
  }

  Status dump() const {
    std::stringstream ss;
    for (const auto &[k, vs] : kernelManagers) {
      for (const auto &[kk, vss] : vs) {
        for (const auto *ks : vss) {
          ss << "namespace: " << k << ", name: " << kk << ", DispatchKey: " << (int)ks->key
             << ", signature: " << Schema(ks).getSchema() << std::endl;
        }
      }
    }

    return Status::success(true, ss.str().empty() ? nullptr : ss.str().c_str());
  }

  UDE_NODISCARD std::map<std::string, int32_t> getAllSchemas() const {
    std::map<std::string, int32_t> schemas;
    for (const auto &[k, vs] : kernelManagers) {
      for (const auto &[kk, vss] : vs) {
        for (const auto *ks : vss) {
          auto schema = Schema(ks).getSchema();
          if (schemas.count(schema) == 0) {
            schemas.emplace(std::move(schema), 1);
          } else {
            schemas[schema] += 1;
          }
        }
      }
    }
    return schemas;
  }

private:
  /// We suppose kernelName looks like namespace::functionName.
  /// This function help to do spilt.
  /// If namespace is not specify, use global namespace to replace.
  UDE_NODISCARD std::pair<std::string, std::string> parseNsAndName(const char *kernelName) const {
    if (kernelName == nullptr) {
      return {"", ""};
    }

    const auto size = strlen(kernelName);
    size_t lastNsPos = size - 1;

    while (lastNsPos != 0 && kernelName[lastNsPos] != ':') {
      --lastNsPos;
    }

    // KernelName hasn't specify namespace, use :: replace it.
    if (lastNsPos == 0) {
      std::string ns("::");
      std::string name(kernelName);
      return {ns, name};
    } else if (lastNsPos == 1) {
      std::string ns("::");
      std::string name(kernelName + lastNsPos + 1);
      return {ns, name};
    } else {
      // Note: ns start from 0, ns length = lastNsPos - 0 + 1.
      std::string ns(kernelName, 0, lastNsPos - 1);
      // Note: name start from lastNsPos + 1, length is size - (lastNsPos + 1) + 1.
      std::string name(kernelName, lastNsPos + 1, size - lastNsPos);
      return {ns, name};
    }
  }

  // KernelManager uses namespace of kernel as the first classification standard,
  // and takes the name of kernel as the second one.
  std::map<std::string, std::map<std::string, std::vector<Kernel *>>> kernelManagers;
};

} // namespace ude
