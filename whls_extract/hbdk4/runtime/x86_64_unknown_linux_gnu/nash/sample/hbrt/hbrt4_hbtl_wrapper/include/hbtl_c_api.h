#pragma once
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdio>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#endif // __cplusplus

#ifndef HBTL_WARPER_C_API
#define HBTL_WARPER_C_API __attribute__((__visibility__("default")))
#endif
#include "hbdk_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data;
  size_t len;
} HbtlArray;

typedef struct {
  char *data;
  size_t dataLen;
  const int64_t *stride;
  const int64_t *size;
  size_t rank;
  hbrt_element_type_t hbrtDataType;
  hbrt_cpu_args_type_t hbrtCpuArgsType;
  bool isDynamicVariable;
  int64_t *dynamicSize; // FIXME(wuruidong): It should be updated through `run_hbtl_kernel` in dynamic_shape running.
} HbtlVariableWrapper;

typedef struct {
  HbtlVariableWrapper *variableWrapper;
  size_t number;
} HbtlVariableWrapperArray;

HBTL_WARPER_C_API extern void runHbtlKernel(HbtlVariableWrapperArray *output, HbtlVariableWrapperArray *input,
                                            HbtlArray *signature);

HBTL_WARPER_C_API extern void jitRegister(void *kernel);
void registerCustom(const char *path);
#ifdef __cplusplus
}
#endif
