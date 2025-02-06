#pragma once
#include <cstddef>
#include <vector>
#ifndef CPU_BPU_MIXED_MODEL_H
#define CPU_BPU_MIXED_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

void cpu_bpu_mixed_model_normal_example(char *hbm_name, char *graph_name, char *inputData, const char *output_path);
void cpu_bpu_mixed_model_pyramid_example(char *hbm_name, char *graph_name, char *data_y_uv, size_t new_stride_w,
                                         const char *output_path);
void cpu_bpu_mixed_model_resizer_example(char *hbm_name, char *graph_name, char *data_y_uv_roi, const char *output_path,
                                         std::vector<std::vector<size_t>> &resizer_dims,
                                         std::vector<std::vector<size_t>> &resizer_strides);

#ifdef __cplusplus
}
#endif
#endif
