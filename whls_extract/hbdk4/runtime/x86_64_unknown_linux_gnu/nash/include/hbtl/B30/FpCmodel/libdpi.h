#pragma once
#ifndef HBTL_SUPPORT_FPCMODEL_LIBDPI_H_
#define HBTL_SUPPORT_FPCMODEL_LIBDPI_H_

#include "hbtl/Support/Compiler.h"

#ifdef __cplusplus
extern "C" {
#endif
HBTL_NAMESPACE_BEGIN {
  namespace b30 {
  namespace bpufp {

  extern void fadd(int fin0, int fin1, int round_mode, vpu_fp_type_e fp_type, int *out_data, int ieee = 0);
  extern void fmul(int fin0, int fin1, int round_mode, vpu_fp_type_e fp_type, int *out_data, int i_eee = 1);
  extern void get_idx(char *str, int *idx);
  extern void fp2int(vpu_fp_type_e fp_type, int fdin, vpu_int_type_e int_type, int round_mode, int *int_out,
                     int ieee = 1);
  extern void int2fp(vpu_int_type_e int_type, int dint, vpu_fp_type_e fp_type, int round_mode, int *fp_out);
  extern void fp2fp(vpu_fp_type_e ftype_in, int fp_in, vpu_fp_type_e ftype_out, int round_mode, int *fp_out);
  extern void vrcp(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  extern void vsqrt(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  extern void vexp(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  extern void vlog(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  extern void vsin(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  extern void vcos(vpu_fp_type_e ftype_in, int fp_in, int *fp_out);
  } // namespace bpufp
  } // namespace b30
}
#ifdef __cplusplus
}
#endif

#endif
