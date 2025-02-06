/// Quantization kernels
#pragma once

#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

HBTL_NAMESPACE_BEGIN {
  class Tensor;
  namespace quant {

  /**
   * @brief QuantizeLinear.
   *        Acting like `onnx::QuantizeLinear` op, performs following calculation
   *        fout[i] = clamp(round(fin[i] / scales[?] + zeros[?]))
   *        rounding to the nearest even.
   *
   * @param fout output tensor, dtype could be i8/i16
   * @param fin input tensor, dtype could be float/double
   * @param scales double vector of scales to use, size should match fin.size(quantizedAxis)
   * @param zeros integer vector of offset to use, size should match fin.size(quantizedAxis)
   * @param axis dimension on which apply per-axis quantization
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */

  HBTL_EXPORTED LogicalResult Quantize(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                       const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis, bool narrowRange);

  /**
   * @brief Dequantize, performs following calculation
   *        fout[i] = (fin[i] - zeroPoint[?]) * scale[?]
   *
   * @param fout output tensor, dtype could be float/double
   * @param fin input tensor, dtype could be i8/i16
   * @param scales double vector of scales to use, size should match fin.size(axis)
   * @param zeros integer vector of offset to use, size should match fin.size(axis)
   * @param axis dimension on which apply per-axis quantization
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult Dequantize(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                         const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis);
  HBTL_EXPORTED LogicalResult DequantizeConfig(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                               const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis);
  /**
   * @brief Quantize.
   *        Acting like `torch.quantize_per_channel` op, performs following calculation
   *        fout[i] = clamp(round(fin[i] / scales[?] + zeros[?]))
   *
   * @param fout output tensor, dtype could be i8/i16
   * @param fin input tensor, dtype could be float/double
   * @param scales double vector of scales to use, size should match fin.size(quantizedAxis)
   * @param zeros integer vector of offset to use, size should match fin.size(quantizedAxis)
   * @param axis dimension on which apply per-axis quantization
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */

  HBTL_EXPORTED LogicalResult Qcast(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                    const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis, bool narrowRange,
                                    bool nearestRound);

  /**
   * @brief Dequantize, performs following calculation
   *        fout[i] = (fin[i] - zeroPoint[?]) * scale[?]
   *
   * @param fout output tensor, dtype could be float/double
   * @param fin input tensor, dtype could be i8/i16
   * @param scales double vector of scales to use, size should match fin.size(axis)
   * @param zeros integer vector of offset to use, size should match fin.size(axis)
   * @param axis dimension on which apply per-axis quantization
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult Dcast(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                    const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis);
  HBTL_EXPORTED LogicalResult DcastConfig(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                          const std::vector<int64_t> &zeros, bool hasAxis, int64_t axis);
  /**
   * @brief Quantize.
   *        Acting like `torch.quantize_per_channel` op, performs following calculation
   *        fout[i] = clamp(round(fin[i] / scales[?] + zeros[?]))
   *
   * @param fout output tensor, dtype could be i8/i16
   * @param fin input tensor, dtype could be float/double
   * @param scales double vector of scales to use, size should match fin.size(quantizedAxis)
   * @param zeros integer vector of offset to use, size should match fin.size(quantizedAxis)
   * @param axis dimension on which apply per-axis quantization
   * @param p verbose detailed calculate steps on the trace point of fout.
   * @return LogicalResult
   */

  HBTL_EXPORTED LogicalResult DynamicQuantize(Tensor &fout, Tensor &scales, const Tensor &fin, int64_t bitWidth,
                                              bool symmetric, bool hasAxis, int64_t axis, bool hasBlockSize,
                                              int64_t blockSize, bool nearestRound);

  HBTL_EXPORTED LogicalResult DynamicQuantizeConfig(Tensor &fout, Tensor &scales, const Tensor &fin, int64_t bitWidth,
                                                    bool symmetric, bool hasAxis, int64_t axis, bool hasBlockSize,
                                                    int64_t blockSize, bool nearestRound);

  HBTL_EXPORTED LogicalResult FuseDynamicDequantizeConfig(Tensor &fout, const Tensor &fin, const Tensor &inputScale,
                                                          const Tensor &weightScale);

  HBTL_EXPORTED LogicalResult FuseDynamicDequantize(Tensor &fout, const Tensor &fin, const Tensor &inputScale,
                                                    const Tensor &weightScale);

  HBTL_EXPORTED LogicalResult DynamicDequantize(Tensor &fout, const Tensor &fin, const Tensor &scales, bool symmetric,
                                                bool hasAxis, int64_t axis);

  HBTL_EXPORTED LogicalResult DynamicDequantizeConfig(Tensor &fout, const Tensor &fin, const Tensor &scales,
                                                      bool symmetric, bool hasAxis, int64_t axis);

  HBTL_EXPORTED LogicalResult FakeQuant(Tensor &fout, const Tensor &fin, const std::vector<double> &scales,
                                        const std::vector<int64_t> &zeros, bool isSigned, int64_t bitWidth,
                                        bool hasAxis, int64_t axis, bool narrowRange, bool nearestRound);

  HBTL_EXPORTED FailureOr<std::tuple<Tensor, Tensor>> MaxCalibrate(const Tensor &fin, ::std::optional<int64_t> axis,
                                                                   bool symmetric);

  } // namespace quant
}
HBTL_NAMESPACE_END
