#include "hbdk_type.h"
#include "hbtl/Dispatch/Dispatcher.h"
#include "hbtl/Dispatch/External.h"
#include "hbtl/Dispatch/SliverC.h"
#include "hbtl/Support/ErrorHandling.h"
#include "hbtl_c_api.h"
#include "ude/internal/Dispatcher.h"
#include "ude/internal/Plan.h"
#include "ude/public/Common.h"
#include "ude/public/Protocols.h"
#include "ude/public/Status.h"
#include "ude/public/Types.h"
#include "ude/public/Variable.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

inline ude::Type convertHbrtTypeToUdeType(hbrt_element_type_t type) {
  switch (type) {
  case ELEMENT_TYPE_INT8: {
    return ude::Type::si8;
  }
  case ELEMENT_TYPE_INT16: {
    return ude::Type::si16;
  }
  case ELEMENT_TYPE_INT32: {
    return ude::Type::si32;
  }
  case ELEMENT_TYPE_INT64: {
    return ude::Type::si64;
  }
  case ELEMENT_TYPE_FLOAT16: {
    return ude::Type::f16;
  }
  case ELEMENT_TYPE_FLOAT32: {
    return ude::Type::f32;
  }
  case ELEMENT_TYPE_FLOAT64: {
    return ude::Type::f64;
  }
  case ELEMENT_TYPE_UINT8: {
    return ude::Type::ui8;
  }
  case ELEMENT_TYPE_UINT16: {
    return ude::Type::ui16;
  }
  case ELEMENT_TYPE_UINT32: {
    return ude::Type::ui32;
  }
  case ELEMENT_TYPE_UINT64: {
    return ude::Type::ui64;
  }
  case ELEMENT_TYPE_BOOL8: {
    return ude::Type::bool8;
  }
  default: {
    return ude::Type::invalid;
  }
  }
}

template <typename T> inline std::unique_ptr<ude::Variable> convertScalarToUdeVariable(const char *data) {
  auto num = *(reinterpret_cast<const T *>(data));
  return std::make_unique<ude::Variable>(num);
}

template <typename T>
inline std::unique_ptr<ude::Variable> convertArrayToUdeVariable(const char *data, size_t len, ude::Type type) {
  ude::VectorRef vecRef{data, len, ude::getType<T>()};
  return std::make_unique<ude::Variable>(vecRef, type);
}

std::unique_ptr<ude::Variable> createUdeVariable(const HbtlVariableWrapper *variableWrapper, bool &needInferShape) {
  auto generateUdeTensorFromVariableWrapper = [&](const HbtlVariableWrapper *variable) {
    ude::TensorRef tf{variable->data,
                      variable->dataLen,
                      convertHbrtTypeToUdeType(variable->hbrtDataType),
                      (int64_t)variable->rank,
                      ude::ArrayRef<int64_t>{variable->size, variable->rank},
                      ude::ArrayRef<int64_t>{variable->stride, variable->rank}};
    return tf;
  };

  auto convertInt8ToString = [](const char *data) {
    return std::make_unique<ude::Variable>(ude::StringRef{data, strlen(data)});
  };
  switch (variableWrapper->hbrtCpuArgsType) {
  case BOOL: {
    return convertScalarToUdeVariable<bool>(variableWrapper->data);
  }
  case INT64: {
    return convertScalarToUdeVariable<int64_t>(variableWrapper->data);
  }
  case FLOAT: {
    return convertScalarToUdeVariable<float>(variableWrapper->data);
  }
  case DOUBLE: {
    return convertScalarToUdeVariable<double>(variableWrapper->data);
  }
  case TENSOR: {
    needInferShape |= variableWrapper->isDynamicVariable;
    return std::make_unique<ude::Variable>(generateUdeTensorFromVariableWrapper(variableWrapper));
  }
  case I64_ARRAY: {
    return convertArrayToUdeVariable<int64_t>(variableWrapper->data, (variableWrapper->dataLen), ude::Type::Int64Vec);
  }
  case F32_ARRAY: {
    return convertArrayToUdeVariable<float>(variableWrapper->data, (variableWrapper->dataLen), ude::Type::F32Vec);
  }
  case F64_ARRAY: {
    return convertArrayToUdeVariable<double>(variableWrapper->data, (variableWrapper->dataLen), ude::Type::F64Vec);
  }
  case STRING: {
    return convertInt8ToString(variableWrapper->data);
  }
  case TENSOR_ARRAY: {
    std::vector<ude::Variable> tensorVec;
    tensorVec.reserve(variableWrapper->dataLen);
    const auto *const childVar =
        static_cast<const HbtlVariableWrapper *>(static_cast<const void *>(variableWrapper->data));
    for (uint32_t idx = 0U; idx < variableWrapper->dataLen; idx++) {
      needInferShape |= childVar[idx].isDynamicVariable;
      tensorVec.emplace_back(generateUdeTensorFromVariableWrapper(childVar + idx));
    }
    return std::make_unique<ude::Variable>(std::move(tensorVec), ude::Type::TensorVec);
  }
  default: {
    return nullptr;
  }
  }
}

struct ModuleCmp {
  bool operator()(const std::unique_ptr<ude::Module> &a, const std::unique_ptr<ude::Module> &b) const {
    return a->getPath() < b->getPath();
  }
};

class Dispatcher {
public:
  static Dispatcher &get() {
    static Dispatcher disp;
    return disp;
  }

  Dispatcher(const Dispatcher &) = delete;
  Dispatcher &operator=(const Dispatcher &) = delete;

  ude::Status update(const std::string &path);

  ude::Dispatcher &getWorker() { return *disp; }

private:
  Dispatcher(); // NOLINT
  std::set<std::unique_ptr<ude::Module>, ModuleCmp> addedModule;
  ude::Dispatcher *disp;
};

Dispatcher::Dispatcher() {
  disp = hbtl::Dispatcher::singleton();
  auto status = disp->load(hbtl::getSliverCHandle());
  assert(status.succeeded());
#if ONEDNN
  status = disp->load(hbtl::getExternalHandle());
  assert(status.succeeded());
#endif
}

ude::Status Dispatcher::update(const std::string &path) {
  auto mod = std::make_unique<ude::Module>(path);
  if (addedModule.count(mod) != 0) {
    return ude::Status::success();
  }
  auto status = mod->load();
  assert(status.succeeded());
  auto dStatus = disp->load(*mod);
  assert(dStatus.succeeded());
  addedModule.insert(std::move(mod));
  return ude::Status::success();
}

} // namespace

extern "C" void runHbtlKernel(HbtlVariableWrapperArray *output, HbtlVariableWrapperArray *input, HbtlArray *signature) {
  using namespace hbtl;
  std::vector<ude::Variable> outVar;
  bool needInferShape = false;
  for (uint32_t idx = 0U; idx < output->number; idx++) {
    needInferShape |= output->variableWrapper[idx].isDynamicVariable;
    auto oneVar = createUdeVariable(&(output->variableWrapper[idx]), needInferShape);
    if (oneVar == nullptr) {
      hbtl_trap("Cannot generate variable for ude");
    }
    // Don't copy ude::variable
    outVar.push_back(std::move(*oneVar));
  }
  std::vector<ude::Variable> inVar;
  for (uint32_t idx = 0U; idx < input->number; idx++) {
    auto oneVar = createUdeVariable(&(input->variableWrapper[idx]), needInferShape);
    if (oneVar == nullptr) {
      hbtl_trap("Cannot generate variable for ude");
    }
    // Don't copy ude::variable
    inVar.push_back(std::move(*oneVar));
  }
  std::string sigString(signature->data);

  auto plan = ude::Plan(sigString, std::move(outVar), std::move(inVar));
  auto task = plan.createTask(::Dispatcher::get().getWorker(), ude::DispatchKey::CUSTOM, true);
  if (!task) {
    std::string s = std::string("No task found for signature ") + sigString;
    hbtl_trap(s.c_str());
  }
  if (needInferShape) {
    auto inferStatus = task.infer();
    if (ude::failed(inferStatus)) {
      std::string errMsg = "Infer kernel failed, for signature " + sigString + " error msg is " + inferStatus.getMsg();
      hbtl_trap(errMsg.c_str());
      return;
    }
  }
  auto status = task.launch();
  if (failed(status)) {
    hbtl_trap("Run kernel failed");
  }

  // The content of task.touchVariable is a set of `outVar` and `inVar`.
  auto ude_output_vars = task.touchVariable();

  // TODO(wurudiong): Current version code is so stupid. Someday I will delete it and reconstruct with an uniform.
  for (uint32_t idx = 0U; idx < output->number; idx++) {
    if (output->variableWrapper[idx].isDynamicVariable) {
      size_t rank = output->variableWrapper[idx].rank;

      auto ude_output_type = ude_output_vars[idx]->getType();
      if (ude_output_type == ude::Type::tensor) {
        auto tensor_ref = ude_output_vars[idx]->getRef<ude::TensorRef>();
        auto new_rank = static_cast<size_t>(tensor_ref.rank);
        if (rank != new_rank) {
          hbtl_trap("Rank should be same in hbtl kernel running");
        }
        auto *size = output->variableWrapper[idx].dynamicSize;
        if (size == nullptr) {
          hbtl_trap("Dynamic shape dst data cannot be nullptr");
        }
        auto new_size = tensor_ref.shape;
        for (size_t r = 0; r < rank; r++) {
          size[r] = new_size[r];
        }
      }
    }
  }
}

extern "C" void jitRegister(void *kernel) {
  using namespace hbtl;
  auto status = ::hbtl::Dispatcher::singleton()->insert(reinterpret_cast<ude::Kernel *>(kernel));
  if (failed(status)) {
    hbtl_trap("insert kernel failed");
  }
}

extern "C" void registerCustom(const char *path) {
  using namespace hbtl;
  auto status = ::Dispatcher::get().update(path);
  if (failed(status)) {
    hbtl_trap("Register custom op failed");
  }
}
