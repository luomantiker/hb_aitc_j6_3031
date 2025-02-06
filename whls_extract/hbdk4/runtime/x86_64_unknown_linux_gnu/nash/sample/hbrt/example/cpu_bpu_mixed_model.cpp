/// \file
/// Example code to run cpu and bpu mixed model
///
/// \author Horizon Robotics, Inc
/// \copyright All rights reserved

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include "cpu_bpu_mixed_model.h"
#include "hbdk_type.h"
#include "hbrt4/hbrt4.hpp"
#include "hbtl/Dispatch/Dispatcher.h"
#include "hbtl_c_api.h"

#ifdef EXAMPLE_B25
#include "plat_cnn.h"
#endif
#ifdef EXAMPLE_B30
#include "hb_bpu_mem.h"
#include <hb_bpu.h>
#endif

// Only used for storing roi resizer input args
struct resizerArgs {
  char *address;
  size_t size;
  const ptrdiff_t *dims_data;
  size_t dims_len;
  const ptrdiff_t *strides_data;
  size_t stride_len;
};
// resizerGlobalVec is initalized in `prepareInputDataResizer()`
// and it is used in `runCpuNode()`
std::vector<resizerArgs> resizerGlobalVec;

// NOLINTNEXTLINE(modernize-macro-to-enum)
#define DEFAULT_CORE_MASK (0xffffffff)

#define CHECK_STATUS(expr)                                                                                             \
  {                                                                                                                    \
    const auto local_status = (expr);                                                                                  \
    if (local_status) {                                                                                                \
      std::cerr << "API errors\n";                                                                                     \
      abort();                                                                                                         \
    }                                                                                                                  \
  }

static ssize_t hbrtLogToStderr(Hbrt4ArrayRef message, const char *moduleName, Hbrt4LogLevel level, uint64_t messageId,
                               Hbrt4LoggerData logData, void *userData) {
  /// Only `message` is used in this function
  // cast others to void to avoid unused variable warnings
  (void)moduleName;
  (void)level;
  (void)messageId;
  (void)logData;
  (void)userData;
  auto ret = ::write(STDERR_FILENO, message.data, message.len);
  if (ret != static_cast<ssize_t>(message.len)) {
    return ret;
  }
  ::write(STDERR_FILENO, "\n", 1);
  return ret;
}

/// C API example to run a model with mixed CPU and BPU operations
/// For simplicity, `hbrt4Status` is not checked

static Hbrt4Logger createLogger() {
  Hbrt4LoggerInitializer initializer = {};
  initializer.handler = hbrtLogToStderr;
  initializer.handlerUserdata = nullptr;
  initializer.level = HBRT4_LOG_LEVEL_INFO;
  Hbrt4Logger logger;
  hbrt4LoggerCreate(initializer, &logger);
  return logger;
}

void jitRegister(void *kernel) {
  auto status = hbtl::Dispatcher::singleton()->insert(reinterpret_cast<ude::Kernel *>(kernel));
}
static Hbrt4Instance createInstance(Hbrt4Logger logger) {
  Hbrt4InstanceBuilder builder;
#ifdef EXAMPLE_B25
  hbrt4InstanceBuilderCreate(HBRT4_BPU_MARCH_BAYES2, &builder);
#endif
#ifdef EXAMPLE_B30
  hbrt4InstanceBuilderCreate(HBRT4_BPU_MARCH_BPU30G, &builder);
#endif
  hbrt4InstanceBuilderRegisterLogger(builder, logger);
  hbrt4InstanceBuilderRegisterJitKernel(builder, jitRegister);
  Hbrt4Instance instance;
  hbrt4InstanceBuilderInto(&builder, &instance);
  return instance;
}

static Hbrt4Hbm loadHbm(const char *filename, Hbrt4Instance instance) {
  Hbrt4Hbm hbm;
  CHECK_STATUS(hbrt4HbmCreateByFilename(instance, filename, Hbrt4HbmCreateFlag{}, &hbm));
  return hbm;
}

static Hbrt4Graph getGraph(Hbrt4Hbm hbm, const char *graphName) {
  Hbrt4Graph graph;
  hbrt4HbmGetGraphByName(hbm, graphName, &graph);
  return graph;
}

static void getVariableShape(Hbrt4Variable var, Hbrt4PtrdiffTArrayRef *dims, Hbrt4PtrdiffTArrayRef *strides) {
  Hbrt4Type type;
  hbrt4VariableGetType(var, &type);
  Hbrt4TypeTag typeTag;
  hbrt4TypeGetTag(type, &typeTag);
  assert((typeTag == HBRT4_TYPE_TAG_TENSOR) && "Function getVariableShape() only supports Tensor Type.");
  hbrt4TypeGetTensorDims(type, dims);
  hbrt4TypeGetTensorStrides(type, strides);
}

static void tensorDataPadding(char *src, char *dst, Hbrt4PtrdiffTArrayRef dims, Hbrt4PtrdiffTArrayRef strides) {
  size_t inner_type_size = strides.data[strides.len - 1];
  size_t dims_len = dims.len;
  size_t strides_len = strides.len;
  assert((dims_len == strides_len) && "The dims_length must be equal to strides_length.");
  size_t dst_len = dims.data[0] * strides.data[0];
  size_t src_element = 1;
  for (size_t i = 0; i < dims_len; ++i) {
    src_element *= dims.data[i];
  }

  std::memset(dst, 0, dst_len);

  auto getOutputIdx = [=](size_t input_idx) -> size_t {
    size_t res = 0;
    size_t idx = input_idx;
    for (size_t i = 0; i < dims_len; ++i) {
      size_t i_dims = dims.data[dims_len - i - 1];
      size_t i_coord = idx % i_dims;
      idx /= i_dims;
      size_t i_stride = strides.data[dims_len - i - 1];
      res += i_coord * i_stride;
    }
    return res;
  };

  for (size_t input_idx = 0; input_idx < src_element; ++input_idx) {
    size_t input_offset = input_idx * inner_type_size;
    size_t output_offset = getOutputIdx(input_idx);
    std::memcpy(dst + output_offset, src + input_offset, inner_type_size);
  }
}

static void prepareTensorData(Hbrt4Variable var, Hbrt4Buffer buffer, size_t offsetInMemspace, char *inputData) {
  void *address;
  hbrt4BufferGetAddress(buffer, &address);

  auto *tensorAddr = offsetInMemspace + (char *)address;
  Hbrt4PtrdiffTArrayRef dims;
  Hbrt4PtrdiffTArrayRef strides;
  getVariableShape(var, &dims, &strides);
  tensorDataPadding(inputData, tensorAddr, dims, strides);
}

static void runBpuNode(Hbrt4Node node,
                       std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> &buffers,
                       std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> &values,
                       Hbrt4Instance instance) {
  size_t numMemspaces;
  hbrt4NodeGetNumMemspaces(node, &numMemspaces);

  Hbrt4CommandBuilder builder;
  hbrt4CommandBuilderCreate(node, &builder);
  for (size_t i = 0; i < numMemspaces; ++i) {
    Hbrt4Memspace memspace;
    hbrt4NodeGetMemspace(node, i, &memspace);
    assert(buffers.count(memspace) > 0);
    CHECK_STATUS(hbrt4CommandBuilderBindBuffer(builder, buffers.at(memspace)));
  }

  size_t numInputVariables;
  hbrt4NodeGetNumInputVariables(node, &numInputVariables);
  for (size_t i = 0; i < numInputVariables; ++i) {
    Hbrt4Variable variable;
    hbrt4NodeGetInputVariable(node, i, &variable);
    auto iter = values.find(variable);
    if (iter != values.end()) {
      CHECK_STATUS(hbrt4CommandBuilderBindValue(builder, values.at(variable)));
    }
  }

  Hbrt4Command command;
  CHECK_STATUS(hbrt4CommandBuilderInto(&builder, &command));

  // Used to fuse multiple bpu command into one pipeline
  // This example run command one by one
  Hbrt4PipelineBuilder pipelineBuilder;
  hbrt4PipelineBuilderCreate(instance, &pipelineBuilder);
  hbrt4PipelineBuilderPushCommand(pipelineBuilder, command);

  Hbrt4Pipeline pipeline;
  CHECK_STATUS(hbrt4PipelineBuilderInto(&pipelineBuilder, &pipeline));

#ifdef EXAMPLE_B25
  Hbrt4Funccall fc;
  hbrt4PipelineGetBpuFunccall(pipeline, &fc);

  cnn_core_set_fc(fc.funccalls, static_cast<int>(fc.numFunccalls), DEFAULT_CORE_MASK, fc.fcDoneCallback);
  cnn_core_check_fc_done(DEFAULT_CORE_MASK, fc.interruptNumber, -1);
#endif

#ifdef EXAMPLE_B30
  hb_bpu_core_t core = {0};
  Hbrt4BpuTask bpuTask;
  hbrt4PipelineGetBpuTask(pipeline, Hbrt4__hb_task_type_t::HBRT4__TASK_TYPE_SYNC, &bpuTask);
  hbrt4__hb_bpu_task_t *driverHandle;
  hbrt4BpuTaskGetDriverHandle(bpuTask, &driverHandle);

  hb_bpu_task_t *task = (hb_bpu_task_t *)driverHandle;
  hb_bpu_core_process(core, *task);
#endif
}

inline hbrt_element_type_t convertHbrtTypeTagToElementType(Hbrt4TypeTag typeTag) {
  switch (typeTag) {
  case HBRT4_TYPE_TAG_SI8:
    return ELEMENT_TYPE_INT8;
  case HBRT4_TYPE_TAG_SI16:
    return ELEMENT_TYPE_INT16;
  case HBRT4_TYPE_TAG_SI32:
    return ELEMENT_TYPE_INT32;
  case HBRT4_TYPE_TAG_SI64:
    return ELEMENT_TYPE_INT64;
  case HBRT4_TYPE_TAG_UI8:
    return ELEMENT_TYPE_UINT8;
  case HBRT4_TYPE_TAG_UI16:
    return ELEMENT_TYPE_UINT16;
  case HBRT4_TYPE_TAG_UI32:
    return ELEMENT_TYPE_UINT32;
  case HBRT4_TYPE_TAG_UI64:
    return ELEMENT_TYPE_UINT64;
  case HBRT4_TYPE_TAG_F16:
    return ELEMENT_TYPE_FLOAT16;
  case HBRT4_TYPE_TAG_F32:
    return ELEMENT_TYPE_FLOAT32;
  case HBRT4_TYPE_TAG_F64:
    return ELEMENT_TYPE_FLOAT64;
  default:
    return ELEMENT_TYPE_UNKNOWN;
  }
}
inline hbrt_cpu_args_type_t convertHbrtTypeTagToCpuArgsType(Hbrt4TypeTag typeTag, Hbrt4TypeTag elementTypeTag) {
  switch (typeTag) {
  case HBRT4_TYPE_TAG_BOOL:
    return BOOL;
  case HBRT4_TYPE_TAG_SI64:
    return INT64;
  case HBRT4_TYPE_TAG_F32:
    return FLOAT;
  case HBRT4_TYPE_TAG_F64:
    return DOUBLE;
  case HBRT4_TYPE_TAG_STRING:
    return STRING;
  case HBRT4_TYPE_TAG_TENSOR:
    return TENSOR;
  case HBRT4_TYPE_TAG_ARRAY: {
    switch (elementTypeTag) {
    case HBRT4_TYPE_TAG_SI64:
      return I64_ARRAY;
    case HBRT4_TYPE_TAG_F32:
      return F32_ARRAY;
    case HBRT4_TYPE_TAG_F64:
      return F64_ARRAY;
    default:
      return NONE;
    }
  }
  default:
    return NONE;
  }
}
static void runCpuNode(Hbrt4Node node,
                       std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> &buffers) {
  size_t numOutput;
  hbrt4NodeGetNumOutputVariables(node, &numOutput);
  size_t numParam;
  hbrt4NodeGetNumParameterVariables(node, &numParam);

  std::vector<HbtlVariableWrapper> paramVec(numParam);
  std::vector<HbtlVariableWrapper> tupleVec;

  for (size_t i = 0; i < numParam; ++i) {
    Hbrt4Variable var;
    hbrt4NodeGetParameterVariable(node, i, &var);
    Hbrt4Type type;
    hbrt4VariableGetType(var, &type);
    Hbrt4TypeTag typeTag;
    hbrt4TypeGetTag(type, &typeTag);
    if (typeTag == HBRT4_TYPE_TAG_TUPLE) {
      size_t numChild;
      hbrt4VariableGetTupleNumChildren(var, &numChild);
      for (size_t j = 0; j < numChild; j++) {
        Hbrt4Variable childVar;
        hbrt4VariableGetTupleChild(var, j, &childVar);
        Hbrt4Type childType;
        hbrt4VariableGetType(childVar, &childType);
        Hbrt4Type eleType;
        hbrt4TypeGetElementType(childType, &eleType);
        Hbrt4TypeTag childTypeTag;
        hbrt4TypeGetTag(eleType, &childTypeTag);

        auto hbrtDataType = convertHbrtTypeTagToElementType(childTypeTag);

        HbtlVariableWrapper childVarWrapper = {resizerGlobalVec[j].address,      resizerGlobalVec[j].size,
                                               resizerGlobalVec[j].strides_data, resizerGlobalVec[j].dims_data,
                                               resizerGlobalVec[j].stride_len,   hbrtDataType,
                                               hbrt_cpu_args_type_t::NONE};
        tupleVec.push_back(childVarWrapper);
      }
      HbtlVariableWrapper tupleWrapper = {reinterpret_cast<char *>(tupleVec.data()),
                                          tupleVec.size(),
                                          reinterpret_cast<const int64_t *>(0),
                                          reinterpret_cast<const int64_t *>(0),
                                          0,
                                          ELEMENT_TYPE_UNKNOWN,
                                          TENSOR_ARRAY};
      paramVec[i] = tupleWrapper;
    } else {
      Hbrt4Memspace memspace;
      hbrt4VariableGetMemspace(var, &memspace);
      assert((buffers.count(memspace) > 0) && "None Memspace in buffers");
      const auto buffer = buffers.at(memspace);
      void *baseAddr;
      size_t size;
      hbrt4BufferGetAddress(buffer, &baseAddr);
      hbrt4BufferGetSize(buffer, &size);
      size_t variableAddrOffset;
      hbrt4VariableGetOffsetInMemspace(var, &variableAddrOffset);
      auto *variableAddr = reinterpret_cast<char *>(baseAddr) + variableAddrOffset;

      // elementTypeTag only used for Tensor and Array
      Hbrt4Type elementType;
      hbrt4TypeGetElementType(type, &elementType);
      Hbrt4TypeTag elementTypeTag;
      hbrt4TypeGetTag(elementType, &elementTypeTag);
      auto hbrtDataType = convertHbrtTypeTagToElementType(elementTypeTag);

      Hbrt4PtrdiffTArrayRef dims;
      Hbrt4PtrdiffTArrayRef strides;
      hbrt4TypeGetTensorDims(type, &dims);
      hbrt4TypeGetTensorStrides(type, &strides);

      auto hbrtCpuArgsType = convertHbrtTypeTagToCpuArgsType(typeTag, elementTypeTag);

      HbtlVariableWrapper normalWrapper = {variableAddr, size,         strides.data,   dims.data,
                                           dims.len,     hbrtDataType, hbrtCpuArgsType};
      paramVec[i] = normalWrapper;
    }
  }

  HbtlVariableWrapperArray outputWrapperArray = {paramVec.data(), numOutput};
  HbtlVariableWrapperArray inputWrapperArray = {paramVec.data() + numOutput, numParam - numOutput};

  const char *operationName;
  hbrt4NodeGetOperationName(node, &operationName);
  std::string operationString(operationName);

  HbtlArray signature = {operationString.data(), operationString.size()};

  runHbtlKernel(&outputWrapperArray, &inputWrapperArray, &signature);
}

static void runNode(Hbrt4Node node, std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> &buffers,
                    std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> &values,
                    Hbrt4Instance instance) {

  Hbrt4NodePseudoTag pseudoTag;
  hbrt4NodeGetPseudoTag(node, &pseudoTag);
  if (pseudoTag != HBRT4_NODE_PSEUDO_TAG_UNKNOWN) {
    // pseudo node. No need to execute. Node just for analysis purpose
    return;
  }

  Hbrt4DeviceCategory category;
  hbrt4NodeGetDeviceCategory(node, &category);
  if (category == HBRT4_DEVICE_CATEGORY_BPU) {
    runBpuNode(node, buffers, values, instance);
  } else {
    assert(category == HBRT4_DEVICE_CATEGORY_CPU);
    runCpuNode(node, buffers);
  }
}

static void prepareInputDataDdr(Hbrt4Graph graph,
                                std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> buffers,
                                char *inputData) {
  size_t numInputVariables;
  hbrt4GraphGetNumInputVariables(graph, &numInputVariables);
  for (size_t i = 0; i < numInputVariables; ++i) {
    Hbrt4Variable variable;
    hbrt4GraphGetInputVariable(graph, i, &variable);
    Hbrt4Type type;
    hbrt4VariableGetType(variable, &type);

    Hbrt4TypeTag typeTag;
    hbrt4TypeGetTag(type, &typeTag);
    assert((typeTag == HBRT4_TYPE_TAG_TENSOR) && "Ddr example only handles tensor");

    Hbrt4VariableInputSemantic input_semantic = HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN;
    hbrt4VariableGetInputSemantic(variable, &input_semantic);
    assert((input_semantic == HBRT4_VARIABLE_INPUT_SEMANTIC_NORMAL) && "Ddr example only supports Normal Semantic");

    Hbrt4Memspace memspace;
    hbrt4VariableGetMemspace(variable, &memspace);
    auto buffer = buffers.at(memspace);

    size_t offsetInMemspace;
    hbrt4VariableGetOffsetInMemspace(variable, &offsetInMemspace);

    prepareTensorData(variable, buffer, offsetInMemspace, inputData);
  }
}

static void prepareInputDataPyramid(Hbrt4Graph graph,
                                    std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> &values,
                                    char *data_y_uv, std::vector<std::vector<size_t>> &new_strides) {
  size_t numInputVariables;
  hbrt4GraphGetNumInputVariables(graph, &numInputVariables);
  for (size_t i = 0; i < numInputVariables; ++i) {
    Hbrt4Variable par_variable;
    hbrt4GraphGetInputVariable(graph, i, &par_variable);

    Hbrt4VariableInputSemantic input_semantic = HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN;
    hbrt4VariableGetInputSemantic(par_variable, &input_semantic);

    // This function only supports Pyramid input model
    if (input_semantic == HBRT4_VARIABLE_INPUT_SEMANTIC_PYRAMID) {
      // Pyramid input includes memory allocation and data padding
      Hbrt4Type type;
      hbrt4VariableGetType(par_variable, &type);
      Hbrt4TypeTag typeTag;
      hbrt4TypeGetTag(type, &typeTag);
      assert((typeTag == HBRT4_TYPE_TAG_TUPLE) && "The Tag of Pyramid input must be tuple.");

      size_t num_children = 0;
      hbrt4VariableGetTupleNumChildren(par_variable, &num_children);
      assert((num_children == 2) && "The input number of Pyramid must be 2.");

      Hbrt4ValueBuilder par_value_builder;
      hbrt4ValueBuilderCreate(par_variable, &par_value_builder);
      for (size_t j = 0; j < num_children; ++j) {
        Hbrt4Variable child_variable;
        hbrt4VariableGetTupleChild(par_variable, j, &child_variable);

        Hbrt4PtrdiffTArrayRef dims;
        Hbrt4PtrdiffTArrayRef ori_strides;
        getVariableShape(child_variable, &dims, &ori_strides);
        size_t strides_len = ori_strides.len;
        assert((strides_len == 4) && "The length of Pyramid strides must equal to 4.");
        new_strides[j][2] = ori_strides.data[2];
        new_strides[j][3] = ori_strides.data[3];

        Hbrt4VariableInputSemantic child_semantic = HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN;
        hbrt4VariableGetInputSemantic(child_variable, &child_semantic);

        // update other strides(optional)
        new_strides[j][0] = dims.data[1] * new_strides[j][1];

        size_t mem_size = dims.data[0] * new_strides[j][0];
        assert((mem_size > 0) && "Allocation size must larger than zero.");
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        void *address = reinterpret_cast<void *>(hb_bpu_mem_alloc(static_cast<int>(mem_size), 0));
        Hbrt4Memspace child_memspace;
        Hbrt4Buffer child_buffer;
        hbrt4VariableGetMemspace(child_variable, &child_memspace);
        hbrt4BufferCreateWithSize(child_memspace, address, mem_size, &child_buffer);

        Hbrt4PtrdiffTArrayRef set_strides = {reinterpret_cast<const ptrdiff_t *>(new_strides[j].data()),
                                             new_strides[j].size()};

        Hbrt4ValueBuilder child_value_builder;
        hbrt4ValueBuilderCreate(child_variable, &child_value_builder);
        hbrt4ValueBuilderSetBuffer(child_value_builder, child_buffer);
        hbrt4ValueBuilderSetTensorStrides(child_value_builder, set_strides);
        Hbrt4Value child_value;
        hbrt4ValueBuilderInto(&child_value_builder, &child_value);

        hbrt4ValueBuilderSetSubValue(par_value_builder, j, child_value);

        tensorDataPadding(data_y_uv, reinterpret_cast<char *>(address), dims, set_strides);
        data_y_uv += mem_size;
      } // end of `num_children`

      Hbrt4Value par_value;
      hbrt4ValueBuilderInto(&par_value_builder, &par_value);
      values[par_variable] = par_value;
    }
  }
}

static void prepareInputDataResizer(Hbrt4Graph graph,
                                    std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> &values,
                                    char *data_y_uv_roi, std::vector<std::vector<size_t>> &resizer_dims,
                                    std::vector<std::vector<size_t>> &resizer_strides) {
  size_t numInputVariables;
  hbrt4GraphGetNumInputVariables(graph, &numInputVariables);
  assert((numInputVariables > 0) && "The number of input variable should larger than 0.");
  for (size_t i = 0; i < numInputVariables; ++i) {
    Hbrt4Variable par_variable;
    hbrt4GraphGetInputVariable(graph, i, &par_variable);

    Hbrt4VariableInputSemantic input_semantic = HBRT4_VARIABLE_INPUT_SEMANTIC_UNKNOWN;
    hbrt4VariableGetInputSemantic(par_variable, &input_semantic);
    if (input_semantic == HBRT4_VARIABLE_INPUT_SEMANTIC_RESIZER) {
      Hbrt4Type type;
      hbrt4VariableGetType(par_variable, &type);
      Hbrt4TypeTag typeTag;
      hbrt4TypeGetTag(type, &typeTag);
      assert((typeTag == HBRT4_TYPE_TAG_TUPLE) && "The Tag of Resizer input must be tuple.");
      size_t num_children = 0;
      hbrt4VariableGetTupleNumChildren(par_variable, &num_children);
      assert((num_children == 3) && "The input number of Resizer must be 3.");

      Hbrt4ValueBuilder par_value_builder;
      hbrt4ValueBuilderCreate(par_variable, &par_value_builder);
      for (size_t j = 0; j < num_children; ++j) {
        Hbrt4Variable child_variable;
        hbrt4VariableGetTupleChild(par_variable, j, &child_variable);
        Hbrt4Memspace memspace;
        hbrt4VariableGetMemspace(child_variable, &memspace);
        size_t mem_size = resizer_dims[j][0] * resizer_strides[j][0];
        assert((mem_size > 0) && "Allocation size must larger than zero.");
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        void *address = reinterpret_cast<void *>(hb_bpu_mem_alloc(static_cast<int>(mem_size), 0));
        Hbrt4Buffer buffer;
        CHECK_STATUS(hbrt4BufferCreateWithSize(memspace, address, mem_size, &buffer));

        Hbrt4PtrdiffTArrayRef set_dims = {reinterpret_cast<const ptrdiff_t *>(resizer_dims[j].data()),
                                          resizer_dims[j].size()};
        Hbrt4PtrdiffTArrayRef set_strides = {reinterpret_cast<const ptrdiff_t *>(resizer_strides[j].data()),
                                             resizer_strides[j].size()};

        // Resizer args are stored as global variable for `runCpuNode()`
        struct resizerArgs arg = {reinterpret_cast<char *>(address),
                                  mem_size,
                                  set_dims.data,
                                  set_dims.len,
                                  set_strides.data,
                                  set_strides.len};
        resizerGlobalVec.push_back(arg);

        Hbrt4ValueBuilder child_value_builder;
        hbrt4ValueBuilderCreate(child_variable, &child_value_builder);
        hbrt4ValueBuilderSetBuffer(child_value_builder, buffer);
        hbrt4ValueBuilderSetTensorDims(child_value_builder, set_dims);
        hbrt4ValueBuilderSetTensorStrides(child_value_builder, set_strides);
        Hbrt4Value child_value;
        hbrt4ValueBuilderInto(&child_value_builder, &child_value);

        hbrt4ValueBuilderSetSubValue(par_value_builder, j, child_value);

        tensorDataPadding(data_y_uv_roi, reinterpret_cast<char *>(address), set_dims, set_strides);

        data_y_uv_roi += mem_size;
      }
      Hbrt4Value par_value;
      hbrt4ValueBuilderInto(&par_value_builder, &par_value);
      values[par_variable] = par_value;
    }
  }
}

static void allocateBuffers(Hbrt4Graph graph,
                            std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> &buffers) {
  size_t numMemspaces;
  hbrt4GraphGetNumMemspaces(graph, &numMemspaces);
  assert((numMemspaces > 0) && "The number of memspaces should larger than zero.");

  for (size_t i = 0; i < numMemspaces; ++i) {
    Hbrt4Memspace memspace;
    hbrt4GraphGetMemspace(graph, i, &memspace);

    size_t alignment = 0;
    size_t size = 0;
    Hbrt4Status ret_status = HBRT4_STATUS_UNKNOWN;

    hbrt4MemspaceGetAlignment(memspace, &alignment);
    ret_status = hbrt4MemspaceGetSize(memspace, &size);

    Hbrt4MemspaceUsage usage;
    hbrt4MemspaceGetUsage(memspace, &usage);
    Hbrt4Buffer buffer;

    if (usage == HBRT4_MEMSPACE_USAGE_CONSTANT) {
      hbrt4MemspaceGetConstantBuffer(memspace, &buffer);
    } else if (ret_status != HBRT4_STATUS_DYNAMIC_VALUE) {
      /// If size < 0, the actually size depend on input
      assert(size >= 0);

      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      void *address = reinterpret_cast<void *>(hb_bpu_mem_alloc(static_cast<int>(size), 0));
      assert((reinterpret_cast<size_t>(address) % alignment) == 0);

      hbrt4BufferCreate(memspace, address, &buffer);
    }

    buffers[memspace] = buffer;
  }
}

static void freeBuffers(std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> &buffers) {
  for (auto &b : buffers) {
    auto memspace = b.first;
    Hbrt4MemspaceUsage usage;
    size_t size;
    hbrt4MemspaceGetUsage(memspace, &usage);
    auto ret_status = hbrt4MemspaceGetSize(memspace, &size);
    if (usage != HBRT4_MEMSPACE_USAGE_CONSTANT && ret_status != HBRT4_STATUS_DYNAMIC_VALUE) {
      auto buffer = b.second;
      void *address;
      hbrt4BufferGetAddress(buffer, &address);
      hb_bpu_mem_free(reinterpret_cast<size_t>(address));
    }
  }
}

static void freeValues(std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> &values) {
  for (auto &v : values) {
    auto value = v.second;
    size_t num_sub_values = 0;
    hbrt4ValueGetNumSubValues(value, &num_sub_values);
    for (size_t i = 0; i < num_sub_values; ++i) {
      Hbrt4Value sub_value;
      hbrt4ValueGetSubValue(value, i, &sub_value);
      Hbrt4Buffer sub_buffer;
      hbrt4ValueGetBuffer(sub_value, &sub_buffer);
      void *address;
      hbrt4BufferGetAddress(sub_buffer, &address);
      hb_bpu_mem_free(reinterpret_cast<size_t>(address));
    }
  }
}

static void runGraph(Hbrt4Graph graph,
                     std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> buffers,
                     std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> values,
                     Hbrt4Instance instance) {
  size_t numNodes;
  hbrt4GraphGetNumNodes(graph, &numNodes);

  for (size_t i = 0; i < numNodes; ++i) {
    Hbrt4Node node;
    hbrt4GraphGetNode(graph, i, &node);
    runNode(node, buffers, values, instance);
  }
}

static void saveOutput(Hbrt4Graph graph,
                       std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> buffers,
                       const char *output_path) {
  size_t numOutputs;
  hbrt4GraphGetNumOutputVariables(graph, &numOutputs);

  for (size_t i = 0; i < numOutputs; ++i) {
    Hbrt4Variable variable;
    hbrt4GraphGetOutputVariable(graph, i, &variable);
    Hbrt4Memspace memspace;
    hbrt4VariableGetMemspace(variable, &memspace);
    size_t offset = 0;
    size_t size = 0;
    hbrt4VariableGetOffsetInMemspace(variable, &offset);
    hbrt4MemspaceGetSize(memspace, &size);
    auto buffer = buffers.at(memspace);
    void *address;
    hbrt4BufferGetAddress(buffer, &address);

    std::string output_filename(output_path);
    output_filename += "output_" + std::to_string(i) + ".bin";
    std::ofstream output_file(output_filename, std::ios::out | std::ios::binary | std::ios::trunc);
    assert(output_file.is_open() && "Cannot open outputfile");
    if (!output_file.is_open()) {
      return;
    }
    output_file.write(reinterpret_cast<char *>(address) + offset, static_cast<long>(size));
    output_file.close();
  }
}

void core_open_wrapper(size_t core_index) {
#ifdef EXAMPLE_B25
  cnn_core_open(core_index);
#endif

#ifdef EXAMPLE_B30
  hb_bpu_core_t core = {core_index};
  hb_bpu_core_open(&core, DEFAULT_CORE_MASK, hb_bpu_choose_t::CHOOSE_BY_CAP);
#endif
}

void core_close_wrapper(size_t core_index) {
#ifdef EXAMPLE_B25
  cnn_core_close(core_index);
#endif

#ifdef EXAMPLE_B30
  hb_bpu_core_t core = {core_index};
  hb_bpu_core_close(core);
#endif
}

void cpu_bpu_mixed_model_normal_example(char *hbm_name, char *graph_name, char *inputData, const char *output_path) {
  Hbrt4Logger logger = createLogger();
  Hbrt4Instance instance = createInstance(logger);
  Hbrt4Hbm hbm = loadHbm(hbm_name, instance);

  // global_buffers is used for mapping `Memspace` to `Buffer`,
  // and program can get `Address` from variable's Memspace.
  std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> global_buffers;
  std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> global_values_not_used;

  core_open_wrapper(0);

  Hbrt4Graph graph = getGraph(hbm, graph_name);
  allocateBuffers(graph, global_buffers);
  prepareInputDataDdr(graph, global_buffers, inputData);
  runGraph(graph, global_buffers, global_values_not_used, instance);
  saveOutput(graph, global_buffers, output_path);

  freeBuffers(global_buffers);

  core_close_wrapper(0);

  hbrt4HbmDestroy(&hbm);
  hbrt4InstanceDestroy(&instance);
  hbrt4LoggerDestroy(&logger);
}

void cpu_bpu_mixed_model_pyramid_example(char *hbm_name, char *graph_name, char *data_y_uv, size_t new_stride_w,
                                         const char *output_path) {
  Hbrt4Logger logger = createLogger();
  Hbrt4Instance instance = createInstance(logger);
  Hbrt4Hbm hbm = loadHbm(hbm_name, instance);

  std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> global_buffers;
  std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> global_values;

  // Pyramid Y stride array is similar with [xxx, xx, 1, 1]
  // Pyramid UV stride array is similar with [xxx, xx, 2, 1]
  // User should set xx with new_stride_w, and update xxx.
  std::vector<std::vector<size_t>> new_strides(2, std::vector<size_t>(4));
  new_strides[0][1] = new_stride_w;
  new_strides[1][1] = new_stride_w;

  core_open_wrapper(0);

  Hbrt4Graph graph = getGraph(hbm, graph_name);
  allocateBuffers(graph, global_buffers);
  prepareInputDataPyramid(graph, global_values, data_y_uv, new_strides);
  runGraph(graph, global_buffers, global_values, instance);
  saveOutput(graph, global_buffers, output_path);

  freeBuffers(global_buffers);
  freeValues(global_values);

  core_close_wrapper(0);

  hbrt4HbmDestroy(&hbm);
  hbrt4InstanceDestroy(&instance);
  hbrt4LoggerDestroy(&logger);
}

void cpu_bpu_mixed_model_resizer_example(char *hbm_name, char *graph_name, char *data_y_uv_roi, const char *output_path,
                                         std::vector<std::vector<size_t>> &resizer_dims,
                                         std::vector<std::vector<size_t>> &resizer_strides) {
  Hbrt4Logger logger = createLogger();
  Hbrt4Instance instance = createInstance(logger);
  Hbrt4Hbm hbm = loadHbm(hbm_name, instance);

  std::map<Hbrt4Memspace, Hbrt4Buffer, hbrt4::CmpObjectByPtr<Hbrt4Memspace>> global_buffers;
  std::map<Hbrt4Variable, Hbrt4Value, hbrt4::CmpObjectByPtr<Hbrt4Variable>> global_values;

  core_open_wrapper(0);

  Hbrt4Graph graph = getGraph(hbm, graph_name);
  allocateBuffers(graph, global_buffers);
  prepareInputDataResizer(graph, global_values, data_y_uv_roi, resizer_dims, resizer_strides);
  runGraph(graph, global_buffers, global_values, instance);
  saveOutput(graph, global_buffers, output_path);

  freeBuffers(global_buffers);
  freeValues(global_values);

  core_close_wrapper(0);

  hbrt4HbmDestroy(&hbm);
  hbrt4InstanceDestroy(&instance);
  hbrt4LoggerDestroy(&logger);
}
