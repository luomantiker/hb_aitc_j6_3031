#pragma once
#ifndef HBTL_SUPPORT_FPCMODEL_FPU_H_
#define HBTL_SUPPORT_FPCMODEL_FPU_H_

#include "hbtl/Support/Compiler.h"

HBTL_NAMESPACE_BEGIN {
  namespace b30 {
  namespace bpufp {

  enum vpu_fp_type_e { FP16 = 0x50a, FP24 = 0x80f, FP_24 = 0x710, FP32 = 0x817, BF16 = 0x807 };

  enum vpu_int_type_e {
    UINT_8 = 0x008,
    SINT_8 = 0x108,
    UINT_16 = 0x010,
    SINT_16 = 0x110,
    UINT_32 = 0x020,
    SINT_32 = 0x120
  };
  } // namespace bpufp
  } // namespace b30
}

#endif
