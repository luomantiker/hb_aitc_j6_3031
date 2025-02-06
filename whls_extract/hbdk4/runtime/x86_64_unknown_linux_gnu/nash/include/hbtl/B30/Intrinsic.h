/// B30 kernels
#pragma once

#include "hbtl/B30/FP.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace b30 {
  namespace intrinsic {

  // ------------------- nash Tae Kernels ------------------- //

  /// native conv mac. weight in [k,r,s,c] order
  HBTL_EXPORTED LogicalResult TaeConvMac(Tensor &psum, const Tensor &fin, const Tensor &weight, ArrayRef<int64_t> pad,
                                         int64_t padValue, ArrayRef<int64_t> stride, ArrayRef<int64_t> dilation,
                                         int64_t group);

  /// zigzag weight from [k,r,s,c] to [k,g,c]
  HBTL_EXPORTED LogicalResult TaeZigzagWeight(Tensor &fout, const Tensor &fin);

  /// zigzagged conv mac, weight in [{n},k,g,c] order, n is broadcastable
  HBTL_EXPORTED LogicalResult TaeConvZigzagMac(Tensor &psum, const Tensor &fin, const Tensor &weight,
                                               ArrayRef<int64_t> kernel, ArrayRef<int64_t> pad, int64_t padValue,
                                               ArrayRef<int64_t> stride, int64_t inputCPerGroup,
                                               int64_t outputCPerGroup);

  HBTL_EXPORTED LogicalResult TaeConvSparseMac(Tensor &psum, const Tensor &fin, const Tensor &weight,
                                               const Tensor &bmap, ArrayRef<int64_t> kernel, ArrayRef<int64_t> pad,
                                               int64_t padValue, ArrayRef<int64_t> stride);

  /// intuitive conv post process. quantinfo for each k are [bias, suminScale, suminRShift, preRShift, scale,
  /// postRShift]
  HBTL_EXPORTED LogicalResult TaeConvPP(Tensor &fout, const Tensor &psum, const Tensor &sumin, const Tensor &quantInfo,
                                        bool foutScaleEn, bool foutRoundEn, bool foutReluEn);

  /// intuitive conv float post process. quantinfo for each k are [bias, suminScale, outputScale]
  HBTL_EXPORTED LogicalResult TaeConvFPP(Tensor &fout, const Tensor &psum, const Tensor &sumin, const Tensor &quantInfo,
                                         bool DirectPsumEn, bool foutRoundEn, bool foutReluEn);

  /// generate quant info from scales and dtypes
  HBTL_EXPORTED LogicalResult GenQuantInfo(Tensor &quantInfo, const Tensor &bias, const Tensor &weight,
                                           ArrayRef<double> finScales, int64_t finZero, ArrayRef<double> weightScales,
                                           ArrayRef<double> foutScales, int64_t foutZero, int64_t foutBitWidth,
                                           ArrayRef<double> suminScales, int64_t suminZero, int64_t suminBitWidth);

  /// generate float quant info from scales and dtypes
  HBTL_EXPORTED LogicalResult GenFloatQuantInfo(Tensor &quantInfo, const Tensor &bias, const Tensor &weight,
                                                ArrayRef<double> finScales, int64_t finZero,
                                                ArrayRef<double> weightScales, ArrayRef<double> foutScales,
                                                int64_t foutZero, int64_t foutBitWidth, ArrayRef<double> suminScales,
                                                int64_t suminZero, int64_t suminBitWidth);

  /// encode quantInfo into hw format
  HBTL_EXPORTED LogicalResult EncodeQuantInfo(Tensor &encode, const Tensor &decode, bool hasSuminPart);

  /// decode quantInfo from hw format
  HBTL_EXPORTED LogicalResult DecodeQuantInfo(Tensor &decode, const Tensor &encode, bool hasSuminPart);

  /// conv post process with encoded quant info
  HBTL_EXPORTED LogicalResult TaeConvPPEncoded(Tensor &fout, const Tensor &psum, const Tensor &sumin,
                                               const Tensor &quantInfo, bool foutScaleEn, bool foutRoundEn,
                                               bool foutReluEn);

  /// one-shot intuitive conv kernel. weight in [k,h,w,c]. quantInfo in [bias, suminScale, suminRShift, preRShift,
  /// scale, postRShift]
  HBTL_EXPORTED LogicalResult TaeConvMacPP(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                           const Tensor &quantInfo, const Tensor &sumin, const Tensor &lut,
                                           int64_t padVal, const std::vector<int64_t> &pad,
                                           const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                           int64_t group, bool scaleEn, bool roundEn, bool reluEn);

  /// one-shot intuitive conv kernel. weight in [k,h,w,c]. quantInfo in [bias, suminScale, outputScale]
  HBTL_EXPORTED LogicalResult TaeConvMacFPP(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                            const Tensor &quantInfo, const Tensor &sumin, const Tensor &lut,
                                            int64_t padVal, const std::vector<int64_t> &pad,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                            int64_t group, bool DirectPsumEn, bool roundEn, bool reluEn);

  HBTL_EXPORTED LogicalResult TaeFP24Lut(Tensor &fout, const Tensor &fin, ArrayRef<int32_t> entries,
                                         ArrayRef<int32_t> ranges, bool foutReluEn);
  // ------------------- nash Dsu Kernels ------------------- //

  HBTL_EXPORTED LogicalResult AaeResize(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &start,
                                        const std::vector<int64_t> &step, const std::string &interp, bool roundEn,
                                        bool padBorder, int64_t padValue);

  HBTL_EXPORTED LogicalResult AaeResizeNearest(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &start,
                                               const std::vector<int64_t> &step, bool padBorder, int64_t padValue);

  HBTL_EXPORTED LogicalResult AaeWarp(Tensor &fout, const Tensor &fin, const Tensor &move, int64_t padValue,
                                      const std::vector<int64_t> &stride, const std::vector<int64_t> &start,
                                      const std::vector<int64_t> &a, const std::vector<int64_t> &b, int64_t s,
                                      const std::string &interp, bool roundEn, bool isMoveYx, int64_t moveFracBitNum,
                                      bool padBorder, bool isTopPadInvalid, bool isBotPadInValid, bool isAbsoluteMv,
                                      bool isNormAbsoluteMv);

  HBTL_EXPORTED LogicalResult AaeWarpNearest(Tensor &fout, const Tensor &fin, const Tensor &move, int64_t padValue,
                                             const std::vector<int64_t> &stride, const std::vector<int64_t> &start,
                                             const std::vector<int64_t> &a, const std::vector<int64_t> &b, int64_t s,
                                             bool isMoveYx, int64_t moveFracBitNum, bool padBorder,
                                             bool isTopPadInvalid, bool isBotPadInValid, bool isAbsoluteMv,
                                             bool isNormAbsoluteMv);

  HBTL_EXPORTED LogicalResult AaeGather(Tensor &fout, const Tensor &fin, const Tensor &move, int64_t padValue,
                                        const std::vector<int64_t> &start, bool roundEn, bool isMoveYx, bool padBorder,
                                        bool isTopPadInvalid, bool isBotPadInvalid, bool is1D);

  HBTL_EXPORTED LogicalResult AaeGatherMultiType(Tensor &fout, const Tensor &fin, const Tensor &move, int64_t padValue,
                                                 const std::vector<int64_t> &start, bool isMoveYx, bool padBorder,
                                                 bool isTopPadInvalid, bool isBotPadInvalid, bool is1D);

  HBTL_EXPORTED LogicalResult SaeScatter(Tensor &fout, const Tensor &fin, const Tensor &indices,
                                         const std::string &scatterReduceMode, const std::vector<int64_t> &start,
                                         bool isMoveYx, const std::string &scatterRoundMode);

  // ------------------- nash Aae Vae Kernels ------------------- //

  HBTL_EXPORTED LogicalResult AaeVaeMaxPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            int64_t padValue, bool padBorder);

  /// one-shot avg pool kernel
  HBTL_EXPORTED LogicalResult AaeVaeAvgPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            int64_t padValue, bool padBorder, int64_t preRShift, int64_t scale,
                                            int64_t postRShift, const std::string &round, bool satEn);

  HBTL_EXPORTED LogicalResult AaeVaeAvgPoolAcc(Tensor &fout, const Tensor &fin, ArrayRef<int64_t> kernel,
                                               ArrayRef<int64_t> stride, ArrayRef<int64_t> pad, int64_t padValue,
                                               bool padBorder);

  HBTL_EXPORTED LogicalResult AaeVaeAvgPoolPP(Tensor &fout, const Tensor &psum, int64_t preRShift, int64_t scale,
                                              int64_t postRShift, RoundMode roundMode, bool satEn);

  // ------------------- nash Vae Eltwise Kernels ------------------- //

  HBTL_EXPORTED LogicalResult EltDequantFp(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &quantInfo,
                                           const std::string &mode, const std::string &round);

  HBTL_EXPORTED LogicalResult EltComplexBinaryFp(Tensor &fout, const Tensor &lhs, const Tensor &rhs,
                                                 const Tensor &quantInfo, const std::string &mode,
                                                 const std::string &round, bool satEn);

  /// complex binary ops: add, sub, mul, sadd
  HBTL_EXPORTED LogicalResult VaeComplexBinary(Tensor &fout, const Tensor &lhs, const Tensor &rhs,
                                               const Tensor &quantInfo, const std::string &mode,
                                               const std::string &round, bool satEn);

  /// compare and logical binary ops
  HBTL_EXPORTED LogicalResult VaeBinary(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const std::string &mode,
                                        const std::string &round);

  /// unary ops: abs and shift
  HBTL_EXPORTED LogicalResult VaeUnary(Tensor &fout, const Tensor &fin, const Tensor &quantInfo,
                                       const std::string &mode, const std::string &round, bool satEn);

  /// the only ternary op: select(where)
  HBTL_EXPORTED LogicalResult VaeSelect(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &sel);

  /// binary half without quant
  HBTL_EXPORTED LogicalResult EltBinaryFp(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const std::string &mode,
                                          const std::string &round);

  /// unary half
  HBTL_EXPORTED LogicalResult EltUnaryFp(Tensor &fout, const Tensor &lhs, const Tensor &quantInfo,
                                         const std::string &mode, const std::string &round, bool satEn);

  /// ternary select
  HBTL_EXPORTED LogicalResult EltSelectFp(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &sel);

  /// max pool half
  LogicalResult EltMaxPoolFp(Tensor &fout, const Tensor &fin, ArrayRef<int64_t> kernel, ArrayRef<int64_t> stride,
                             ArrayRef<int64_t> pad, half_t padValue, bool padBorder);

  // ------------------- nash Vae Lut Kernels ------------------- //

  /**
   * @brief B30 Vae Simple Lut Kernel
   *           v = fin[...,c]
   *           fout[...,c] = entries[v / 64 + v % 64] >> (c % 2 ==0) ? 8 : 0
   *
   * @param fout output tensor, dtype could be si8.
   * @param fin input tensor, dtype could be si8.
   * @param entries look-up table.
   * @param p verbose detailed calculate steps on the trace point of fout.
   */

  HBTL_EXPORTED LogicalResult VaeLutSimple(Tensor &fout, const Tensor &fin, const Tensor &entries);

  /**
   * @brief B30 Vae Lut Kernel. Calculate result with index transform table and interpolation table.
   *
   * @param fout output tensor, dtype could be si8, si16.
   * @param fin input tensor, dtype could be si8.
   * @param entries interpolation table
   * @param ranges index transform table, two linear table. and six look-up tables.
   * @param symMode symmetric mode of value could be y symmetric ,center symmetric and no symmetric.
   * @param round floor, round or ceil when right shift.
   * @param satEn saturate to fout dtype or truncate msb.
   * @param p verbose detailed calculate steps on the trace point of fout.
   */

  HBTL_EXPORTED LogicalResult VaeLut(Tensor &fout, const Tensor &fin, const Tensor &entries, const Tensor &ranges,
                                     const Tensor &quant_info, const std::string &sym, const std::string &roundMode,
                                     const std::string &lutMode, bool satEn);

  // ------------------- nash Vae Reduce Kernels ------------------- //

  /**
   * @brief B30 Vae ReduceSum Kernel. Performs following calculation
   *            accu = psum[...,0] + \sum_{c=0}^{C} fin[...,c]
   *            fout[...,0] = accu >> rshift
   *
   * @param fout output tensor, dtype could be si32, si16 or si8.
   * @param psum optional psum tensor, dtype must be si32. use zero to when invalid.
   * @param fin input tensor, dtype could be si16 or si8.
   * @param round floor, round or ceil when right shift.
   * @param satEn saturate to fout dtype or truncate msb.
   * @param rshift number of lsb of accu to be shifted.
   * @param p verbose detailed calculate steps on the trace point of fout.
   */
  HBTL_EXPORTED LogicalResult VaeReduceSum(Tensor &fout, const Tensor &psum, const Tensor &fin, RoundMode round,
                                           bool satEn, uint32_t rshift);

  enum class ReduceMaxFunc {
    MAX = 3,        // MAX(A)
    ARGMAX = 4,     // ARGMAX(A)
    ARGMAX_MAX = 5, // ARGMAX(A), MAX(A)
    UNKNOWN,
  };

  /**
   * @brief B30 Vae ReduceMax Kernel. Performs following calculation
   *            val = \max_{c=0}^{C} fin[...,c]
   *            fout[...,0] = max(val,psum[...,0])
   *
   * @param fout output tensor, dtype could be si16 or si8.
   * @param psum optional psum tensor, dtype must be si32. use zero to when invalid.
   * @param fin input tensor, dtype could be si16 or si8.
   * @param reduceMaxFunc function of reduce could be max, argmax and argmax_max
   * @param startC the real c index for the first using input channel
   * @param p verbose detailed calculate steps on the trace point of fout.
   */
  HBTL_EXPORTED LogicalResult VaeReduceMax(Tensor &fout, const Tensor &psum, const Tensor &fin,
                                           ReduceMaxFunc reduceMaxFunc, int64_t startC);

  enum class BinaryReduceAccFunc {
    DOT = 0, // SUM(A * B) >> S2
    L1 = 1,  // SUM(|A - B|) >> S2
  };
  /**
   * @brief B30 Vae Binary Reduce Kernel. Performs following calculation
   *            accu = psum[...,0] + \sum_{c=0}^{C} lhs[...,c] * rhs[...,c] or |lhs[...,c] - rhs[...,c] |
   *            fout[...,0] = accu >> rshift
   *
   * @param fout output tensor, dtype could be si16 or si8.
   * @param psum optional psum tensor, dtype must be si32. use zero to when invalid.
   * @param lhs input tensor, dtype could be si16 or si8.
   * @param rhs input tensor, dtype could be si16 or si8.
   * @param binaryReduceAccFunc function of reduce could dot and l1.
   * @param round floor, round or ceil when right shift.
   * @param satEn saturate to fout dtype or truncate msb.
   * @param rshift number of lsb of accu to be shifted.
   * @param p verbose detailed calculate steps on the trace point of fout.
   */
  HBTL_EXPORTED LogicalResult VaeBinaryReduceAcc(Tensor &fout, const Tensor &psum, const Tensor &lhs, const Tensor &rhs,
                                                 BinaryReduceAccFunc binaryReduceAccFunc, RoundMode round, bool satEn,
                                                 uint32_t rshift);

  HBTL_EXPORTED LogicalResult Nv12ToYuv444(Tensor &yuv, const Tensor &y, const Tensor &uv);

  /**
   * @brief B30 Load/Store crc32 kernel.
   *
   *
   * @param crc uint32_t, CRC32-bit check.
   * @param fin input tensor, dtype could be ui8.
   * @param p verbose detailed calculate steps on the trace point of crc32.
   * @return uint32_t crc
   */
  HBTL_EXPORTED uint32_t crc32(uint32_t crc, const Tensor &fin);

  /**
   * @brief B30 Load/Store RunLengthEncode kernel.
   *
   *
   * @param fout fout tensor, dtype could be si8 or si16.
   * @param fin input tensor, dtype could be si8.
   * @param busAlign fout offect, LOAD_STORE_BUS_ALIGNMENT / sizeof(int8_t).
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult RunLengthEncode(Tensor &fout, const Tensor &fin, int64_t busAlign);

  HBTL_EXPORTED LogicalResult RunLengthEncodePostProcess(Tensor &foutCnt, Tensor &foutData, const Tensor &finCnt,
                                                         uint64_t invalidNum);

  HBTL_EXPORTED LogicalResult RlePostProcess(Tensor &encodeCount, Tensor &encodeData, const Tensor &fin,
                                             int64_t busAlign);

  HBTL_EXPORTED LogicalResult RlePostProcessConfig(Tensor &encodeCount, Tensor &encodeData, const Tensor &fin,
                                                   int64_t busAlign);

  /**
   * @brief B30 Load/Store RunLengthDecode kernel.
   *
   *
   * @param fout fout tensor, dtype could be si8.
   * @param fin input tensor, dtype could be si8 or si16.
   * @param busAlign fout offect, LOAD_STORE_BUS_ALIGNMENT / sizeof(int8_t).
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult RunLengthDecode(Tensor &fout, const Tensor &fin, int64_t busAlign);

  /**
   * @brief B30 Load/Store FilterScore kernel.
   *
   *
   * @param fout fout tensor, dtype could be si8.
   * @param fin input tensor, dtype could be si8 or si16.
   * @param score input tensor, dtype could be si16.
   * @param start the start offset
   * @param scoreThresh
   * @param isScoreFirst true for the first of Score tensor
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult FilterScore(Tensor &fout, const Tensor &score, const Tensor &fin,
                                          std::vector<int64_t> &start, int16_t scoreThresh, bool isScoreFirst);

  HBTL_EXPORTED LogicalResult FilterOp(Tensor &fout, const Tensor &score, const Tensor &fin, Tensor &partialCnt,
                                       Tensor &partialData, std::vector<int64_t> &start, int64_t scoreThresh,
                                       bool isScoreFirst, bool hasPartial);

  constexpr uint32_t defaultCrcInit = 0xFFFFFFFF;
  static const uint32_t LOAD_STORE_BUS_ALIGNMENT = 32U;
  static const uint32_t STORE_ALIGNMENT = 64U;
#define LOAD_STORE_PACKET_LANE_NUM 4U
  static const uint32_t LOAD_STORE_PACKET_BYTE_SIZE = 16U;

  static const uint32_t L1M_CHANNEL_NUM = 16U;
  static const uint32_t L1M_BANK_NUM = 8U;
  static const uint32_t L1M_BANK_DEPTH = 2048U;
  static const uint32_t L1M_CHANNEL_BYTE_SIZE = 16U;
  static const uint32_t L1M_TOTAL_BYTE_SIZE = L1M_CHANNEL_BYTE_SIZE * L1M_BANK_DEPTH * L1M_BANK_NUM * L1M_CHANNEL_NUM;
  static const uint32_t L1M_BLOCK_BYTE_SIZE = L1M_CHANNEL_NUM * L1M_CHANNEL_BYTE_SIZE;
  static const uint32_t L1M_ADDRESS_RANGE = L1M_BANK_NUM * L1M_BANK_DEPTH;

  /**
   * @brief B30 Load/Store CompressPacketLane kernel.
   *
   *
   * @param laneBuffer fout tensor, dtype could be ui32.
   * @param finTensor input tensor, dtype could be si8.
   * @param lane
   * @param laneNum
   * @param maskval
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return uint32_t
   */
  HBTL_EXPORTED uint32_t CompressPacketLane(uint32_t lane, Tensor &laneBuffer, const Tensor &finTensor, int8_t maskVal,
                                            uint32_t laneNum, uint32_t l1mBlockByteSize, uint32_t l1mChannelNum,
                                            uint32_t l1mChannelByteSize, bool enXor = false);

  HBTL_EXPORTED LogicalResult CompressPacketLanePyBinding(Tensor &length, Tensor &laneBuffer, const Tensor &finTensor,
                                                          int64_t lane, int64_t maskVal, int64_t laneNum,
                                                          int64_t l1mBlockByteSize, int64_t l1mChannelNum,
                                                          int64_t l1mChannelByteSize, bool enXor = false);

  HBTL_EXPORTED LogicalResult CompressPacketLanePyBindingInfer(Tensor &length, Tensor &laneBuffer,
                                                               const Tensor &finTensor, int64_t lane, int64_t maskVal,
                                                               int64_t laneNum, int64_t l1mBlockByteSize,
                                                               int64_t l1mChannelNum, int64_t l1mChannelByteSize,
                                                               bool enXor = false);

  /**
   * @brief B30 Load/Store DecompressPacketLane kernel.
   *
   *
   * @param laneBuffer input tensor, dtype could be ui32.
   * @param calTensor fout tensor, dtype could be si8.
   * @param lane
   * @param laneNum
   * @param maskval
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return uint32_t
   */
  HBTL_EXPORTED uint32_t DecompressPacketLane(uint32_t lane, const Tensor &laneBuffer, Tensor &calTensor,
                                              int8_t maskVal, uint32_t laneNum, uint32_t l1mBlockByteSize,
                                              uint32_t l1mChannelNum, uint32_t l1mChannelByteSize, bool enXor = false);

  } // namespace intrinsic
  } // namespace b30
}
HBTL_NAMESPACE_END
