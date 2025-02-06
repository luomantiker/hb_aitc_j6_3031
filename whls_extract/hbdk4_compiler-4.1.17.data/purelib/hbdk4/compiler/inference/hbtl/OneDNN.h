/// oneDNN kernels
#pragma once

#include "hbtl/Native.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace oneDNN {

  HBTL_EXPORTED LogicalResult Conv2dNHWC(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                         const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult Conv2dNHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                               const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                               const std::vector<int64_t> &dilation, int64_t group);

  HBTL_EXPORTED LogicalResult QuantizedConv2d(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                              int64_t padVal, const std::vector<int64_t> &stride,
                                              const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                              int64_t group);

  HBTL_EXPORTED LogicalResult QuantizedConv2dPP(Tensor &fout, const Tensor &psum, const Tensor &sumin,
                                                const Tensor &quantInfo, bool foutScaleEn, bool foutRoundEn,
                                                bool foutReluEn);

  HBTL_EXPORTED LogicalResult B25ConvMacPP(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                           const Tensor &quantInfo, const Tensor &sumin, int64_t padVal,
                                           const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                                           const std::vector<int64_t> &dilation, int64_t group, bool scaleEn,
                                           bool roundEn, bool reluEn);

  HBTL_EXPORTED LogicalResult B30ConvMacPP(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                           const Tensor &quantInfo, const Tensor &sumin, const Tensor &lut,
                                           int64_t padVal, const std::vector<int64_t> &pad,
                                           const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                           int64_t group, bool scaleEn, bool roundEn, bool reluEn);

  HBTL_EXPORTED LogicalResult Linear(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias);
  HBTL_EXPORTED LogicalResult LinearNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight);

  HBTL_EXPORTED LogicalResult B30ComplexBinary(Tensor &fout, const Tensor &lhs, const Tensor &rhs,
                                               const Tensor &quantInfo, const std::string &mode,
                                               const std::string &round, bool satEn);
  } // namespace oneDNN
}
HBTL_NAMESPACE_END
