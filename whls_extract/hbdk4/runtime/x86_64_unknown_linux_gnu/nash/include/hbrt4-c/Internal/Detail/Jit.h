/// Internal testing helper
/// Not part of public API
/// May change at any time

// IWYU pragma: private, include "hbrt4-c/Internal/AllInternal.h"

#pragma once

#if !defined(__clangd__) && !defined(HBRT4_INTERNAL_DETAIL_GUARD)
#error "This file should not be directly included. Include hbrt4-c/Internal/AllInternal.h instead"
#endif

#include "hbrt4-c/hbrt4-c.h"

typedef struct {
  uint64_t *data;
  size_t capacity;
  size_t size;
} JitVector;

typedef struct {
  int64_t ih;
  int64_t iw;
  int64_t ic;
  int64_t istride;
} JitImage;

typedef struct {
  int64_t oh;
  int64_t ow;
  int64_t oc;
  int64_t ostride;
} JitOutput;

typedef struct {
  int64_t w_begin;
  int64_t h_begin;
  int64_t w_end;
  int64_t h_end;
} JitRoi;

enum PadMode {
  PAD_NEAREST,
  PAD_CONST,
};

typedef struct {
  int64_t y_pad;
  int64_t uv_pad;
} PadValue;

#ifdef __cplusplus
extern "C" {
#endif
/// Function pointers to internal APIs
Hbrt4Status Hbrt4RoiResizeGenerator(JitVector *instVec, JitImage image, JitOutput output, JitRoi roi,
                                    enum PadMode pad_mode, PadValue pad_value);

Hbrt4Status Hbrt4BatchRoiResizeGenerator(JitVector *instVec, const JitImage images[], const JitRoi rois[],
                                         JitOutput output, enum PadMode pad_mode, PadValue pad_value,
                                         size_t batch_size);

Hbrt4Status Hbrt4AppendNop(JitVector *instVec);
Hbrt4Status Hbrt4SetEmptyInst(JitVector *instVec);
#ifdef __cplusplus
}
#endif

extern const struct Hbrt4JitIAPI HBRT4_JIT_IAPI;
