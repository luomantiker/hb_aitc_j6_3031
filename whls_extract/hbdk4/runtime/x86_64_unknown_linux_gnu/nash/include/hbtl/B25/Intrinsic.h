#pragma once

#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace b25 {
  namespace intrinsic {

  // ------------------- bayes Conv Kernels ------------------- //

  /// native conv mac. weight in [k,r,s,c] order
  HBTL_EXPORTED LogicalResult ConvMac(Tensor &psum, const Tensor &fin, const Tensor &weight, ArrayRef<int64_t> pad,
                                      int64_t padValue, ArrayRef<int64_t> stride, ArrayRef<int64_t> dilation,
                                      int64_t group);

  /// zigzag weight from [k,r,s,c] to [k,g,c]
  HBTL_EXPORTED LogicalResult ConvZigzagWeight(Tensor &fout, const Tensor &fin);

  /// zigzagged conv mac, weight in [{n},k,g,c] order, n is broadcastable
  HBTL_EXPORTED LogicalResult ConvZigzagMac(Tensor &psum, const Tensor &fin, const Tensor &weight,
                                            ArrayRef<int64_t> kernel, ArrayRef<int64_t> pad, int64_t padValue,
                                            ArrayRef<int64_t> stride, int64_t intputCPerGroup, int64_t outputCPerGroup);

  HBTL_EXPORTED LogicalResult ConvSparseMac(Tensor &psum, const Tensor &fin, const Tensor &weight, const Tensor &bmap,
                                            ArrayRef<int64_t> kernel, ArrayRef<int64_t> pad, int64_t padValue,
                                            ArrayRef<int64_t> stride);

  /// intuitive conv post process. quantinfo for each k are [bias, suminScale, suminRShift, preRShift, scale,
  /// postRShift]
  HBTL_EXPORTED LogicalResult ConvPP(Tensor &fout, const Tensor &psum, const Tensor &sumin, const Tensor &quantInfo,
                                     bool foutScaleEn, bool foutRoundEn, bool foutReluEn);

  /// generate quant info from scales and dtypes
  HBTL_EXPORTED LogicalResult GenQuantInfo(Tensor &quantInfo, const Tensor &bias, const Tensor &weight,
                                           ArrayRef<double> finScales, int64_t finZero, ArrayRef<double> weightScales,
                                           ArrayRef<double> foutScales, int64_t foutZero, int64_t foutBitWidth,
                                           ArrayRef<double> suminScales, int64_t suminZero, int64_t suminBitWidth);

  /// encode quantInfo into hw format
  HBTL_EXPORTED LogicalResult EncodeQuantInfo(Tensor &encode, const Tensor &decode);

  /// decode quantInfo from hw format
  HBTL_EXPORTED LogicalResult DecodeQuantInfo(Tensor &decode, const Tensor &encode, int64_t suminBitWidth);

  /// conv post process with encoded quant info
  HBTL_EXPORTED LogicalResult ConvPPEncoded(Tensor &fout, const Tensor &psum, const Tensor &sumin,
                                            const Tensor &quantInfo, bool foutScaleEn, bool foutRoundEn,
                                            bool foutReluEn);

  /// one-shot intuitive conv kernel. weight in [k,h,w,c]. quantInfo in [bias, suminScale, suminRShift, preRShift,
  /// scale, postRShift]
  HBTL_EXPORTED LogicalResult ConvMacPP(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &quantInfo,
                                        const Tensor &sumin, const Tensor &lut, int64_t padVal,
                                        const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                                        const std::vector<int64_t> &dilation, int64_t group, int64_t blockByteSize,
                                        bool scaleEn, bool roundEn, bool reluEn);

  // ------------------- bayes Dsu Kernels ------------------- //

  HBTL_EXPORTED LogicalResult DsuResize(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &start,
                                        const std::vector<int64_t> &step, const std::string &interp, bool roundEn,
                                        bool padBorder, int64_t padValue);

  HBTL_EXPORTED LogicalResult DsuWarp(Tensor &fout, const Tensor &fin, const Tensor &move, int64_t padValue,
                                      const std::string &interp, const std::vector<int64_t> &stride,
                                      const std::vector<int64_t> &start, bool roundEn, bool isMoveYx,
                                      int64_t moveFracBitNum, bool padBorder);

  // ------------------- bayes Dsu Elt Kernels ------------------- //

  /// one-shot max pool kernel
  HBTL_EXPORTED LogicalResult DsuEltMaxPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            int64_t padValue, bool padBorder);

  /// one-shot avg pool kernel
  HBTL_EXPORTED LogicalResult DsuEltAvgPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            int64_t padValue, bool padBorder, int64_t preRShift, int64_t scale,
                                            int64_t postRShift, const std::string &round, bool satEn);

  HBTL_EXPORTED LogicalResult DsuEltAvgPoolAcc(Tensor &fout, const Tensor &fin, ArrayRef<int64_t> kernel,
                                               ArrayRef<int64_t> stride, ArrayRef<int64_t> pad, int64_t padValue,
                                               bool padBorder);

  HBTL_EXPORTED LogicalResult DsuEltAvgPoolPP(Tensor &fout, const Tensor &psum, int64_t preRShift, int64_t scale,
                                              int64_t postRShift, RoundMode roundMode, bool satEn);

  // ------------------- bayes Elt Eltwise Kernels ------------------- //

  /// complex binary ops: add, sub, mul, sadd
  HBTL_EXPORTED LogicalResult EltComplexBinary(Tensor &fout, const Tensor &lhs, const Tensor &rhs,
                                               const Tensor &quantInfo, const std::string &mode,
                                               const std::string &round, bool satEn);

  /// compare and logical binary ops
  HBTL_EXPORTED LogicalResult EltBinary(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const std::string &mode);

  /// unary ops: abs and shift
  HBTL_EXPORTED LogicalResult EltUnary(Tensor &fout, const Tensor &fin, const Tensor &quantInfo,
                                       const std::string &mode, const std::string &round, bool satEn);

  /// the only ternary op: select(where)
  HBTL_EXPORTED LogicalResult EltSelect(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &sel);

  HBTL_EXPORTED LogicalResult EltReduceSum(Tensor &fout, const Tensor &psum, const Tensor &fin, RoundMode round,
                                           bool satEn, uint32_t rshift);

  // ------------------- bayes Elt Lut Kernels ------------------- //

  HBTL_EXPORTED LogicalResult EltLutSimple(Tensor &fout, const Tensor &fin, const Tensor &entries);

  HBTL_EXPORTED LogicalResult EltLut(Tensor &fout, const Tensor &fin, const Tensor &entries, const Tensor &ranges,
                                     const std::string &sym, const std::string &round, bool satEn);

  HBTL_EXPORTED LogicalResult ConvLut(Tensor &fout, const Tensor &fin, ArrayRef<int16_t> entries,
                                      ArrayRef<int64_t> ranges);

  // ------------------- bayes Elt Reduce Kernels ------------------- //

  enum class ReduceMaxFunc {
    MAX = 3,        // MAX(A)
    ARGMAX = 4,     // ARGMAX(A)
    ARGMAX_MAX = 5, // ARGMAX(A), MAX(A)
    UNKNOWN,
  };

  HBTL_EXPORTED LogicalResult EltReduceMax(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                           ReduceMaxFunc reduceMaxFunc, int64_t startC);

  enum class BinaryReduceAccFunc {
    DOT = 0, // SUM(A * B) >> S2
    L1 = 1,  // SUM(|A - B|) >> S2
  };

  HBTL_EXPORTED LogicalResult EltBinaryReduceAcc(Tensor &fout, const Tensor &psum, const Tensor &lhs, const Tensor &rhs,
                                                 BinaryReduceAccFunc binaryReduceAccFunc, RoundMode round, bool satEn,
                                                 uint32_t rshift);

  // ------------------- bayes Load store Kernels ------------------- //

  HBTL_EXPORTED LogicalResult Nv12ToYuv444(Tensor &yuv, const Tensor &y, const Tensor &uv);

  HBTL_EXPORTED LogicalResult RunLengthEncode(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult RunLengthEncodePyBinding(Tensor &length, Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult RunLengthEncodePyBindingInfer(Tensor &length, Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult RunLengthDecode(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult FilterScore(Tensor &fout, const Tensor &score, const Tensor &fin, ArrayRef<int64_t> start,
                                          int16_t scoreThresh, bool isScoreFirst);

  HBTL_EXPORTED LogicalResult CompressPacket(Tensor &compress, const Tensor &raw, int8_t maskVal);

  HBTL_EXPORTED LogicalResult DecompressPacket(const Tensor &raw, Tensor &compress, int8_t maskVal);

  } // namespace intrinsic
  } // namespace b25
}
HBTL_NAMESPACE_END
