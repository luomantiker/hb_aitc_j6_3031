#pragma once

#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/LogicalResult.h"
#include <string>
#include <utility>

HBTL_NAMESPACE_BEGIN {
  class Tensor;

  HBTL_EXPORTED hbtl::LogicalResult AllIdentical(const hbtl::Tensor &cal, const hbtl::Tensor &ref,
                                                 bool detailView = false);

  HBTL_EXPORTED hbtl::LogicalResult AllClose(const hbtl::Tensor &cal, const hbtl::Tensor &ref, float rtol = 1e-6,
                                             float atol = 1e-6, bool equalNaN = false, bool detailView = false);

  using AnalysisResult = std::pair<float, std::string>;

  /// return signal to quantization-noise ratio.
  HBTL_EXPORTED FailureOr<AnalysisResult> SignalQuantizationNoiseRatio(const Tensor &a, const Tensor &b);

  /// return cosine distance.
  HBTL_EXPORTED FailureOr<AnalysisResult> CosineDistance(const Tensor &a, const Tensor &b);

  /// return root-mean-square error
  HBTL_EXPORTED FailureOr<AnalysisResult> RootMeanSquareDeviation(const Tensor &a, const Tensor &b);

  /// return maximum quantization noise.
  HBTL_EXPORTED FailureOr<AnalysisResult> MaximumQuantizationNoise(const Tensor &a, const Tensor &b);
}
HBTL_NAMESPACE_END
