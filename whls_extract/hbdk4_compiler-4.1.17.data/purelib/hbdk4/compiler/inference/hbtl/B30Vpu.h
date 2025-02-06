/// B30 Vpu kernels
#pragma once

#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/LogicalResult.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace b30vpu {

  /// Binary vpu ops
  HBTL_EXPORTED LogicalResult BinaryEltwiseAdd(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseAddConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseSub(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseSubConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMul(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMulConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseDiv(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseDivConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMax(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMaxConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMin(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMinConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwisePow(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwisePowConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseMod(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseModConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseRem(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BinaryEltwiseRemConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);

  /// Unary vpu ops
  HBTL_EXPORTED LogicalResult UnaryEltwiseExp(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseExpConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseReciprocal(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseReciprocalConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseRsqrt(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseRsqrtConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSqrt(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSqrtConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSigmoid(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSigmoidConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseLeakyReLU(Tensor &fout, const Tensor &fin, double slop);
  HBTL_EXPORTED LogicalResult UnaryEltwiseLeakyReLUConfig(Tensor &fout, const Tensor &fin, double slop);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSoftplus(Tensor &fout, const Tensor &fin, double beta, double threshold);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSoftplusConfig(Tensor &fout, const Tensor &fin, double beta,
                                                         double threshold);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAbs(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAbsConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseNeg(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseNegConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseClip(Tensor &fout, const Tensor &fin, double low, double high);
  HBTL_EXPORTED LogicalResult UnaryEltwiseClipConfig(Tensor &fout, const Tensor &fin, double low, double high);
  HBTL_EXPORTED LogicalResult UnaryEltwiseGELU(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseGELUConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAtan(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAtanConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAsin(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAsinConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAcos(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseAcosConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSin(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseSinConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseCos(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseCosConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseLog(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseLogConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseCast(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult UnaryEltwiseCastConfig(Tensor &fout, const Tensor &fin);

  // Reduce vpu ops
  HBTL_EXPORTED LogicalResult ReduceSum(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                        bool keepdim);

  HBTL_EXPORTED LogicalResult ReduceMean(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                         bool keepdim, int64_t reduceNum);

  HBTL_EXPORTED LogicalResult ReduceMaxK(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                         bool keepDim, int64_t reduceNum, int64_t k);

  HBTL_EXPORTED LogicalResult ReduceArgMaxK(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                            bool keepDim, int64_t reduceNum, int64_t k);

  HBTL_EXPORTED LogicalResult ReduceMaxWithPartialSum(Tensor &fout, const Tensor &fin, const Tensor &partial,
                                                      const std::vector<int64_t> &dims, bool keepDim);

  HBTL_EXPORTED LogicalResult ReduceSumWithPartialSum(Tensor &fout, const Tensor &fin, const Tensor &partial,
                                                      const std::vector<int64_t> &dims, bool keepDim);

  HBTL_EXPORTED LogicalResult ReduceMeanWithPartialSum(Tensor &fout, const Tensor &fin, const Tensor &partial,
                                                       const std::vector<int64_t> &dims, bool keepDim,
                                                       int64_t reduceNum);

  HBTL_EXPORTED LogicalResult Quant(Tensor &fout, const Tensor &in, const std::vector<double> &scales,
                                    const std::vector<int64_t> &zeros, const std::vector<int64_t> &constShape);

  HBTL_EXPORTED LogicalResult Dequant(Tensor &fout, const Tensor &in, const std::vector<double> &scales,
                                      const std::vector<int64_t> &zeros, const std::vector<int64_t> &constShape);

  } // namespace b30vpu
}

HBTL_NAMESPACE_END
