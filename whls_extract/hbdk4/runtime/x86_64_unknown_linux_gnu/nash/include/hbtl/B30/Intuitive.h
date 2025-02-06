/// B30 kernels
#pragma once

#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace b30 {

  HBTL_EXPORTED LogicalResult Conv2d(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                     const Tensor &sumin, const Tensor &padVal, const std::vector<int64_t> &kernel,
                                     const std::vector<int64_t> &pad, const std::vector<int64_t> &stride, int64_t group,
                                     bool scaleEn, bool roundEn, bool reluEn);

  HBTL_EXPORTED LogicalResult Conv2dOneDNN(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                           const Tensor &sumin, const Tensor &padVal,
                                           const std::vector<int64_t> &kernel, const std::vector<int64_t> &pad,
                                           const std::vector<int64_t> &stride, int64_t group, bool scaleEn,
                                           bool roundEn, bool reluEn);

  HBTL_EXPORTED LogicalResult ReduceMaxV(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                         const std::vector<int64_t> &axes, int64_t startC);
  HBTL_EXPORTED LogicalResult ReduceMaxI(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                         const std::vector<int64_t> &axes, int64_t startC);
  HBTL_EXPORTED LogicalResult ReduceMaxVI(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                          const std::vector<int64_t> &axes, int64_t startC);

  HBTL_EXPORTED LogicalResult ReduceSum(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                        const std::vector<int64_t> &axes, int64_t preRShift, bool scaleEn,
                                        int64_t scale, int64_t postRShift, RoundMode round, bool satEn);

  namespace compiler {
  /**
   * @brief B30 EncodeBias kernel.
   *
   *
   * @param quantInfo encoded quantInfo for bpu conv
   * @param bias optional conv bias, dtype is si32.
   * @param finScale conv fin scale.
   * @param weightScales conv weight scales.
   * @param foutScales conv fout scales.
   * @param suminScales conv sumin scales. if conv don't have sumin, then this vector should be empty.
   * @param foutBitWidth conv output element type bit width
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult EncodeBias(Tensor &quantInfo, const Tensor &bias, double finScale,
                                         ArrayRef<double> weightScales, ArrayRef<double> foutScales,
                                         ArrayRef<double> suminScales, int64_t foutBitWidth);

  HBTL_EXPORTED LogicalResult zigzagWeight(Tensor &fout, const Tensor &fin);

  } // namespace compiler
  } // namespace b30
}
HBTL_NAMESPACE_END
