/// Native kernels
#pragma once

#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace native {

  /**
   * @brief Native Conv2d Kernel.
   *
   * @param fout output tensor
   * @param fin input tensor
   * @param weight weight tensor
   * @param bias
   * @param stride
   * @param pad pad size on top left
   * @param dilation
   */
  HBTL_EXPORTED LogicalResult Conv2dNHWC(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                         const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult Conv2dNHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                               const Tensor &bias, const std::vector<int64_t> &stride,
                                               const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                               int64_t group);
  HBTL_EXPORTED LogicalResult Conv2dNHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                               const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                               const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult Conv2dNHWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                     const std::vector<int64_t> &stride,
                                                     const std::vector<int64_t> &pad,
                                                     const std::vector<int64_t> &dilation, int64_t group);

  /**
   * @brief Native Conv3d Kernel.
   *
   * @param fout output tensor
   * @param fin input tensor
   * @param weight weight tensor
   * @param bias
   * @param stride
   * @param pad pad size on depth top left
   * @param dilation
   */
  HBTL_EXPORTED LogicalResult Conv3dNDHWC(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                          const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                          const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult Conv3dNDHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                const Tensor &bias, const std::vector<int64_t> &stride,
                                                const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                                int64_t group);
  HBTL_EXPORTED LogicalResult Conv3dNDHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                                const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult Conv3dNDHWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                      const std::vector<int64_t> &stride,
                                                      const std::vector<int64_t> &pad,
                                                      const std::vector<int64_t> &dilation, int64_t group);

  /**
   * @brief Native Conv Kernel.
   *
   * @param fout output tensor
   * @param fin input tensor
   * @param weight weight tensor
   * @param bias
   * @param stride
   * @param pad
   * @param dilation
   */
  HBTL_EXPORTED LogicalResult Conv(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                   const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                   const std::vector<int64_t> &dilation, int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvConfig(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                         const std::vector<int64_t> &dilation, int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                         const std::vector<int64_t> &dilation, int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                               const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                               const std::vector<int64_t> &dilation, int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult Conv1dNLC(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                        const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                        const std::vector<int64_t> &dilation, int64_t group);
  HBTL_EXPORTED LogicalResult DeformConv2dNHWC(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                               const Tensor &offset, const Tensor &mask, const Tensor &bias,
                                               const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                               const std::vector<int64_t> &dilation, int64_t group, int64_t offsetGroup,
                                               bool useMask);
  HBTL_EXPORTED LogicalResult DeformConv2dNHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                     const Tensor &offset, const Tensor &mask, const Tensor &bias,
                                                     const std::vector<int64_t> &stride,
                                                     const std::vector<int64_t> &pad,
                                                     const std::vector<int64_t> &dilation, int64_t group,
                                                     int64_t offsetGroup, bool useMask);
  HBTL_EXPORTED LogicalResult DeformConv2dNHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                     const Tensor &offset, const Tensor &mask,
                                                     const std::vector<int64_t> &stride,
                                                     const std::vector<int64_t> &pad,
                                                     const std::vector<int64_t> &dilation, int64_t group,
                                                     int64_t offsetGroup, bool useMask);
  HBTL_EXPORTED LogicalResult DeformConv2dNHWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                           const Tensor &offset, const Tensor &mask,
                                                           const std::vector<int64_t> &stride,
                                                           const std::vector<int64_t> &pad,
                                                           const std::vector<int64_t> &dilation, int64_t group,
                                                           int64_t offsetGroup, bool useMask);
  /**
   * @brief Native Conv1dTranspose Kernel.
   *
   * @param fout output tensor
   * @param fin input tensor
   * @param weight weight tensor
   * @param bias
   * @param stride
   * @param pad clip on output [h_top, w_left, h_bottom, w_right]
   * @param dilation
   */
  HBTL_EXPORTED LogicalResult Conv1dTransposeNWC(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                 const Tensor &bias, const std::vector<int64_t> &stride,
                                                 const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                                 int64_t group, bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv1dTransposeNWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                       const Tensor &bias, const std::vector<int64_t> &stride,
                                                       const std::vector<int64_t> &pad,
                                                       const std::vector<int64_t> &dilation, int64_t group,
                                                       bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv1dTransposeNWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                       const std::vector<int64_t> &stride,
                                                       const std::vector<int64_t> &pad,
                                                       const std::vector<int64_t> &dilation, int64_t group,
                                                       bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv1dTransposeNWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                             const std::vector<int64_t> &stride,
                                                             const std::vector<int64_t> &pad,
                                                             const std::vector<int64_t> &dilation, int64_t group,
                                                             bool illegalWeight);
  HBTL_EXPORTED LogicalResult ConvAdd(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &sumin,
                                      const Tensor &bias, const std::vector<int64_t> &stride,
                                      const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                      int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvAddConfig(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &sumin,
                                            const Tensor &bias, const std::vector<int64_t> &stride,
                                            const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                            int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvAddNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &sumin,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            const std::vector<int64_t> &dilation, int64_t group, bool channelLast);
  HBTL_EXPORTED LogicalResult ConvAddNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                  const Tensor &sumin, const std::vector<int64_t> &stride,
                                                  const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                                  int64_t group, bool channelLast);

  /**
   * @brief Native Conv2dTranspose Kernel.
   *
   * @param fout output tensor
   * @param fin input tensor
   * @param weight weight tensor
   * @param bias
   * @param stride
   * @param pad clip on output [h_top, w_left, h_bottom, w_right]
   * @param dilation
   */
  HBTL_EXPORTED LogicalResult Conv2dTransposeNHWC(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                  const Tensor &bias, const std::vector<int64_t> &stride,
                                                  const std::vector<int64_t> &pad, const std::vector<int64_t> &dilation,
                                                  int64_t group, bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv2dTransposeNHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                        const Tensor &bias, const std::vector<int64_t> &stride,
                                                        const std::vector<int64_t> &pad,
                                                        const std::vector<int64_t> &dilation, int64_t group,
                                                        bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv2dTransposeNHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                        const std::vector<int64_t> &stride,
                                                        const std::vector<int64_t> &pad,
                                                        const std::vector<int64_t> &dilation, int64_t group,
                                                        bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv2dTransposeNHWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                              const std::vector<int64_t> &stride,
                                                              const std::vector<int64_t> &pad,
                                                              const std::vector<int64_t> &dilation, int64_t group,
                                                              bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv3dTransposeNDHWC(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                   const Tensor &bias, const std::vector<int64_t> &stride,
                                                   const std::vector<int64_t> &pad,
                                                   const std::vector<int64_t> &dilation, int64_t group,
                                                   bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv3dTransposeNDHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                         const Tensor &bias, const std::vector<int64_t> &stride,
                                                         const std::vector<int64_t> &pad,
                                                         const std::vector<int64_t> &dilation, int64_t group,
                                                         bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv3dTransposeNDHWCNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                         const std::vector<int64_t> &stride,
                                                         const std::vector<int64_t> &pad,
                                                         const std::vector<int64_t> &dilation, int64_t group,
                                                         bool illegalWeight);
  HBTL_EXPORTED LogicalResult Conv3dTransposeNDHWCNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                                               const std::vector<int64_t> &stride,
                                                               const std::vector<int64_t> &pad,
                                                               const std::vector<int64_t> &dilation, int64_t group,
                                                               bool illegalWeight);

  HBTL_EXPORTED LogicalResult Linear(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias);
  HBTL_EXPORTED LogicalResult LinearConfig(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias);
  HBTL_EXPORTED LogicalResult LinearNoBias(Tensor &fout, const Tensor &fin, const Tensor &weight);
  HBTL_EXPORTED LogicalResult LinearNoBiasConfig(Tensor &fout, const Tensor &fin, const Tensor &weight);

  /**
   * @brief Nd average pooling
   *        Acting like torch.nn.AvgPool2d
   *
   * @param fout  output tensor, dtype could be float/double
   * @param fin  input tensor, dtype could be float/double
   * @param kernel  the size of the kernel window
   * @param stride  the stride of the kernel window
   * @param pad  implicit zero padding to be added on single sides
   * @param dilation a parameter that controls the stride of elements in the window
   * @param p
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult AvgPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                      const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                      const std::vector<int64_t> &dilation, bool isCeilMode);
  HBTL_EXPORTED LogicalResult AvgPoolConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            const std::vector<int64_t> &dilation, bool isCeilMode);

  HBTL_EXPORTED LogicalResult MaskedSelect(Tensor &fout, const Tensor &fin, const Tensor &mask);

  HBTL_EXPORTED LogicalResult MaskedSelectConfig(Tensor &fout, const Tensor &fin, const Tensor &mask);

  /**
   * @brief 2d max pooling
   *        Acting like torch.nn.MaxPool2d
   *
   * @param fout  output tensor, dtype could be float/double
   * @param fin  input tensor, dtype could be float/double
   * @param kernel  the size of the kernel window
   * @param stride  the stride of the kernel window
   * @param pad  implicit -INF padding to be added on single sides
   * @param dilation  a parameter that controls the stride of elements in the window
   * @param p
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult MaxPool(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                      const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                      const std::vector<int64_t> &dilation, bool isCeilMode);
  HBTL_EXPORTED LogicalResult MaxPoolConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                            const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                            const std::vector<int64_t> &dilation, bool isCeilMode);

  HBTL_EXPORTED LogicalResult LpPoolChannelLast(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &kernel,
                                                const std::vector<int64_t> &stride, const std::vector<int64_t> &pad,
                                                const std::vector<int64_t> &dilation, float p, bool isCeilMode);
  HBTL_EXPORTED LogicalResult LpPoolChannelLastConfig(Tensor &fout, const Tensor &fin,
                                                      const std::vector<int64_t> &kernel,
                                                      const std::vector<int64_t> &stride,
                                                      const std::vector<int64_t> &pads,
                                                      const std::vector<int64_t> &dilation, float p, bool isCeilMode);
  HBTL_EXPORTED LogicalResult BevPoolV2(Tensor &fout, const Tensor &depth, const Tensor &feat,
                                        const Tensor &ranks_depth, const Tensor &ranks_feat, const Tensor &ranks_bev,
                                        const Tensor &interval_starts, const Tensor &interval_lengths,
                                        const std::vector<int64_t> &bev_feat_shape);

  HBTL_EXPORTED LogicalResult BevPoolV2Config(Tensor &fout, const Tensor &depth, const Tensor &feat,
                                              const Tensor &ranks_depth, const Tensor &ranks_feat,
                                              const Tensor &ranks_bev, const Tensor &interval_starts,
                                              const Tensor &interval_lengths,
                                              const std::vector<int64_t> &bev_feat_shape);

  /// binary series
  HBTL_EXPORTED LogicalResult Add(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult AddConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitShift(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitShiftConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseAnd(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseAndConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseOr(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseOrConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseXor(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult BitwiseXorConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Div(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult DivConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Equal(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult EqualConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Greater(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult GreaterConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult GreaterEqual(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult GreaterEqualConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Less(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LessConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LessEqual(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LessEqualConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalAnd(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalAndConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalOr(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalOrConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalXor(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult LogicalXorConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Max(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult MaxConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Min(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult MinConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Mul(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult MulConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult NotEqual(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult NotEqualConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Pow(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult PowConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Mod(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult ModConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Rem(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult RemConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Sub(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult SubConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Where(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &sls);
  HBTL_EXPORTED LogicalResult WhereConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs, const Tensor &sls);

  /// reduce series
  HBTL_EXPORTED LogicalResult ReduceMax(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                        bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceMaxConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceArgMax(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                           bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceArgMaxConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                 bool keepdim);

  HBTL_EXPORTED LogicalResult ReduceSum(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                        bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceSumConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceMean(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                         bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceMeanConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                               bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceMin(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                        bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceMinConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceArgMin(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                           bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceArgMinConfig(Tensor &foutType, const Tensor &finType,
                                                 const std::vector<int64_t> &dims, bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceLp(Tensor &fout, const Tensor &fin, int64_t p);
  HBTL_EXPORTED LogicalResult CumSum(Tensor &fout, const Tensor &fin, int64_t dims, bool exclusive, bool reverse);
  HBTL_EXPORTED LogicalResult CumSumConfig(Tensor &fout, const Tensor &fin, int64_t dims, bool exclusive, bool reverse);
  HBTL_EXPORTED LogicalResult HardMax(Tensor &fout, const Tensor &fin, int64_t dims);
  HBTL_EXPORTED LogicalResult ReduceLogSumExp(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceProd(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                         bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceProdConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                               bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceLogSum(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                           bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceLogSumConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                 bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceSumSquare(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceSumSquareConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                    bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceAll(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                        bool keepdim);
  HBTL_EXPORTED LogicalResult ReduceAllConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              bool keepdim);

  /// activation series
  HBTL_EXPORTED LogicalResult ReLU(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult ReLUConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult LeakyReLU(Tensor &fout, const Tensor &fin, double slop);
  HBTL_EXPORTED LogicalResult LeakyReLUConfig(Tensor &fout, const Tensor &fin, double slop);
  HBTL_EXPORTED LogicalResult Sigmoid(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SigmoidConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Tanh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult TanhConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Swish(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SwishConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Mish(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult MishConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult GELU(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult GELUConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult GELUV2(Tensor &fout, const Tensor &fin, const std::string &approximate);
  HBTL_EXPORTED LogicalResult GELUV2Config(Tensor &fout, const Tensor &fin, const std::string &approximate);
  HBTL_EXPORTED LogicalResult Softplus(Tensor &fout, const Tensor &fin, double beta, double threshold);
  HBTL_EXPORTED LogicalResult SoftplusConfig(Tensor &fout, const Tensor &fin, double beta, double threshold);

  HBTL_EXPORTED LogicalResult Hardsigmoid(Tensor &fout, const Tensor &fin, float alpha, float beta);
  HBTL_EXPORTED LogicalResult HardsigmoidConfig(Tensor &fout, const Tensor &fin, float alpha, float beta);
  HBTL_EXPORTED LogicalResult Elu(Tensor &fout, const Tensor &fin, float alpha);
  HBTL_EXPORTED LogicalResult EluConfig(Tensor &fout, const Tensor &fin, float alpha);
  HBTL_EXPORTED LogicalResult PRelu(Tensor &fout, const Tensor &fin, const Tensor &slope);
  HBTL_EXPORTED LogicalResult PReluConfig(Tensor &fout, const Tensor &fin, const Tensor &slope);
  HBTL_EXPORTED LogicalResult Lut(Tensor &fout, const Tensor &fin, const Tensor &config);
  HBTL_EXPORTED LogicalResult LutConfig(Tensor &fout, const Tensor &fin, const Tensor &config);

  HBTL_EXPORTED LogicalResult ClampF(Tensor &fout, const Tensor &fin, double lo, double hi);
  HBTL_EXPORTED LogicalResult ClampFConfig(Tensor &fout, const Tensor &fin, double lo, double hi);
  HBTL_EXPORTED LogicalResult Clamp(Tensor &fout, const Tensor &fin, int64_t lo, int64_t hi);
  HBTL_EXPORTED LogicalResult ClampConfig(Tensor &fout, const Tensor &fin, int64_t lo, int64_t hi);

  // arith series
  HBTL_EXPORTED LogicalResult Sqrt(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SqrtConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Rsqrt(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult RsqrtConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Reciprocal(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult ReciprocalConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Exp(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult ExpConfig(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult Abs(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AbsConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Acos(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AcosConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Acosh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AcoshConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Asin(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AsinConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Asinh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AsinhConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Atan(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AtanConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Atanh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult AtanhConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult BitwiseNot(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult BitwiseNotConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Ceil(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult CeilConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Cos(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult CosConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Cosh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult CoshConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Erf(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult ErfConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Floor(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult FloorConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Log(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult LogConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult LogicalNot(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult LogicalNotConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Neg(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult NegConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Round(Tensor &fout, const Tensor &fin, int64_t decimals = 0);
  HBTL_EXPORTED LogicalResult RoundConfig(Tensor &fout, const Tensor &fin, int64_t decimals = 0);
  HBTL_EXPORTED LogicalResult Selu(Tensor &fout, const Tensor &fin, float alpha, float gamma);
  HBTL_EXPORTED LogicalResult SeluConfig(Tensor &fout, const Tensor &fin, float alpha, float gamma);
  HBTL_EXPORTED LogicalResult Sign(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SignConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Sin(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SinConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Sinh(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SinhConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Softsign(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult SoftsignConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Tan(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult TanConfig(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult Cast(Tensor &fout, const Tensor &fin, bool saturate);
  HBTL_EXPORTED LogicalResult CastConfig(Tensor &fout, const Tensor &fin, bool saturate);
  HBTL_EXPORTED LogicalResult FakeCast(Tensor &fout, const Tensor &fin, const ElementType &dtype);
  HBTL_EXPORTED LogicalResult FakeCastConfig(Tensor &fout, const Tensor &fin, const ElementType &dtype);
  HBTL_EXPORTED LogicalResult NonZero(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult NonZeroConfig(Tensor &fout, const Tensor &fin);

  // normalize series
  HBTL_EXPORTED LogicalResult Softmax(Tensor &fout, const Tensor &fin, int64_t dim);
  HBTL_EXPORTED LogicalResult SoftmaxConfig(Tensor &fout, const Tensor &fin, int64_t dim);
  HBTL_EXPORTED LogicalResult LogSoftmax(Tensor &fout, const Tensor &fin, int64_t dim);
  HBTL_EXPORTED LogicalResult LogSoftmaxConfig(Tensor &fout, const Tensor &fin, int64_t dim);

  HBTL_EXPORTED LogicalResult BatchNormChannelLast(Tensor &fout, const Tensor &fin, const Tensor &mean,
                                                   const Tensor &var, const Tensor &weight, const Tensor &bias,
                                                   float eps = 1e-5);
  HBTL_EXPORTED LogicalResult BatchNormChannelLastConfig(Tensor &fout, const Tensor &fin, const Tensor &mean,
                                                         const Tensor &var, const Tensor &weight, const Tensor &bias,
                                                         float eps = 1e-5);
  HBTL_EXPORTED LogicalResult BatchNormChannelLastNoAffine(Tensor &fout, const Tensor &fin, const Tensor &mean,
                                                           const Tensor &var, float eps = 1e-5);
  HBTL_EXPORTED LogicalResult BatchNormChannelLastNoAffineConfig(Tensor &fout, const Tensor &fin, const Tensor &mean,
                                                                 const Tensor &var, float eps = 1e-5);

  HBTL_EXPORTED LogicalResult LayerNorm(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                        const std::vector<int64_t> &dims, float eps = 1e-5);
  HBTL_EXPORTED LogicalResult LayerNormConfig(Tensor &fout, const Tensor &fin, const Tensor &weight, const Tensor &bias,
                                              const std::vector<int64_t> &dims, float eps = 1e-5);
  HBTL_EXPORTED LogicalResult LayerNormNoAffine(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                float eps = 1e-5);
  HBTL_EXPORTED LogicalResult LayerNormNoAffineConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                      float eps = 1e-5);

  HBTL_EXPORTED LogicalResult RmsNorm(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                      const std::vector<int64_t> &dims, float eps = 1e-5);
  HBTL_EXPORTED LogicalResult RmsNormConfig(Tensor &fout, const Tensor &fin, const Tensor &weight,
                                            const std::vector<int64_t> &dims, float eps = 1e-5);
  HBTL_EXPORTED LogicalResult RmsNormNoAffine(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                              float eps = 1e-5);
  HBTL_EXPORTED LogicalResult RmsNormNoAffineConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                                    float eps = 1e-5);

  HBTL_EXPORTED LogicalResult LRN(Tensor &fout, const Tensor &fin, int64_t size, float alpha = 0.0001,
                                  float beta = 0.75, float bias = 1.0);
  HBTL_EXPORTED LogicalResult LpNormalize(Tensor &fout, const Tensor &fin, float p = 2.0, int64_t dim = 1,
                                          float eps = 1e-12);
  HBTL_EXPORTED LogicalResult LpNormalizeConfig(Tensor &fout, const Tensor &fin, float p = 2.0, int64_t dim = 1,
                                                float eps = 1e-12);

  // sample series
  // ratio and size just for infertype, hbtl not useless
  HBTL_EXPORTED LogicalResult Resize2dBilinearNHWC(Tensor &fout, const Tensor &fin, const std::vector<double> &offset,
                                                   const std::vector<double> &step, bool padBorder, double padValue,
                                                   const std::vector<double> &ratio = {},
                                                   const std::vector<int64_t> &sizes = {});
  HBTL_EXPORTED LogicalResult Resize2dBilinearNHWCConfig(Tensor &fout, const Tensor &fin,
                                                         const std::vector<double> &offset,
                                                         const std::vector<double> &step, bool padBorder,
                                                         double padValue, const std::vector<double> &ratio = {},
                                                         const std::vector<int64_t> &sizes = {});
  // ratio and size just for infertype, hbtl not useless
  HBTL_EXPORTED LogicalResult Resize2dNearestNHWC(Tensor &fout, const Tensor &fin, const std::vector<double> &offset,
                                                  const std::vector<double> &step, bool padBorder, double padValue,
                                                  const std::vector<double> &ratio = {},
                                                  const std::vector<int64_t> &sizes = {});

  HBTL_EXPORTED LogicalResult Resize2dNearestNHWCConfig(Tensor &fout, const Tensor &fin,
                                                        const std::vector<double> &offset,
                                                        const std::vector<double> &step, bool padBorder,
                                                        double padValue, const std::vector<double> &ratio = {},
                                                        const std::vector<int64_t> &sizes = {});

  HBTL_EXPORTED LogicalResult Resize2dBicubicNHWC(Tensor &fout, const Tensor &fin, const std::vector<double> &offset,
                                                  const std::vector<double> &step, bool padBorder, double padValue,
                                                  double cubicCoeffA = -0.75, const std::vector<double> &ratio = {},
                                                  const std::vector<int64_t> &sizes = {});

  HBTL_EXPORTED LogicalResult Resize2dBicubicNHWCConfig(Tensor &fout, const Tensor &fin,
                                                        const std::vector<double> &offset,
                                                        const std::vector<double> &step, bool padBorder,
                                                        double padValue, double cubicCoeffA = -0.75,
                                                        const std::vector<double> &ratio = {},
                                                        const std::vector<int64_t> &sizes = {});

  HBTL_EXPORTED LogicalResult GridSample2dBilinearNHWC(Tensor &fout, const Tensor &fin, const Tensor &grid,
                                                       bool padBorder, bool alignCorner, double padValue);
  HBTL_EXPORTED LogicalResult GridSample2dBilinearNHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &grid,
                                                             bool padBorder, bool alignCorner, double padValue);
  HBTL_EXPORTED LogicalResult GridSample2dNearestNHWC(Tensor &fout, const Tensor &fin, const Tensor &grid,
                                                      bool padBorder, bool alignCorner, double padValue);
  HBTL_EXPORTED LogicalResult GridSample2dNearestNHWCConfig(Tensor &fout, const Tensor &fin, const Tensor &grid,
                                                            bool padBorder, bool alignCorner, double padValue);

  HBTL_EXPORTED LogicalResult Warp2dBilinear(Tensor &fout, const Tensor &fin, const Tensor &move, bool padBorder,
                                             double padValue);
  HBTL_EXPORTED LogicalResult Warp2dBilinearConfig(Tensor &fout, const Tensor &fin, const Tensor &move, bool padBorder,
                                                   double padValue);
  HBTL_EXPORTED LogicalResult Warp2dNearest(Tensor &fout, const Tensor &fin, const Tensor &move, bool padBorder,
                                            double padValue);
  HBTL_EXPORTED LogicalResult Warp2dNearestConfig(Tensor &fout, const Tensor &fin, const Tensor &move, bool padBorder,
                                                  double padValue);

  HBTL_EXPORTED LogicalResult RoiResizeNV12(Tensor &fout, const Tensor &y, const Tensor &uv, const Tensor &roi,
                                            const std::vector<int64_t> &padValue, const std::string &interpolateMode,
                                            const std::string &expansionMode, const std::vector<int64_t> &size = {});
  HBTL_EXPORTED LogicalResult RoiResizeNV12Config(Tensor &fout, const Tensor &y, const Tensor &uv, const Tensor &roi,
                                                  const std::vector<int64_t> &padValue,
                                                  const std::string &interpolateMode, const std::string &expansionMode,
                                                  const std::vector<int64_t> &size = {});

  HBTL_EXPORTED LogicalResult RoiResizeY(Tensor &fout, const Tensor &y, const Tensor &roi,
                                         const std::vector<int64_t> &padValue, const std::string &interpolateMode,
                                         const std::string &expansionMode, const std::vector<int64_t> &size = {});
  HBTL_EXPORTED LogicalResult RoiResizeYConfig(Tensor &fout, const Tensor &y, const Tensor &roi,
                                               const std::vector<int64_t> &padValue, const std::string &interpolateMode,
                                               const std::string &expansionMode, const std::vector<int64_t> &size = {});

  // move series
  HBTL_EXPORTED LogicalResult Concat(Tensor &fout, const std::vector<Tensor> &fins, int64_t dim);
  HBTL_EXPORTED LogicalResult ConcatConfig(Tensor &fout, const std::vector<Tensor> &fins, int64_t dim);
  HBTL_EXPORTED LogicalResult PadBorder(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                        const std::vector<int64_t> &end);
  HBTL_EXPORTED LogicalResult PadBorderConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                              const std::vector<int64_t> &end);
  HBTL_EXPORTED LogicalResult PadConst(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                       const std::vector<int64_t> &end, int64_t padValue);
  HBTL_EXPORTED LogicalResult PadConstConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                             const std::vector<int64_t> &end, int64_t padValue);
  HBTL_EXPORTED LogicalResult PadConstF(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                        const std::vector<int64_t> &end, double padValue);
  HBTL_EXPORTED LogicalResult PadConstFConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                              const std::vector<int64_t> &end, double padValue);
  HBTL_EXPORTED LogicalResult ReshapeConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &shape);
  HBTL_EXPORTED LogicalResult Reshape(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &shape);
  HBTL_EXPORTED LogicalResult Select(Tensor &fout, const Tensor &fin, int64_t dim, int64_t index);
  HBTL_EXPORTED LogicalResult SelectConfig(Tensor &fout, const Tensor &fin, int64_t dim, int64_t index);
  HBTL_EXPORTED LogicalResult SliceConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                          const std::vector<int64_t> &step, const std::vector<int64_t> &end);
  HBTL_EXPORTED LogicalResult Slice(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &begin,
                                    const std::vector<int64_t> &step, const std::vector<int64_t> &end);
  HBTL_EXPORTED LogicalResult DynamicSliceConfig(Tensor &fout, const Tensor &fin, const Tensor &starts,
                                                 const Tensor &ends, const Tensor &axes, const Tensor &steps);
  HBTL_EXPORTED LogicalResult DynamicSlice(Tensor &fout, const Tensor &fin, const Tensor &starts, const Tensor &ends,
                                           const Tensor &axes, const Tensor &steps);
  HBTL_EXPORTED LogicalResult SliceScatterConfig(Tensor &fout, const Tensor &fin, const Tensor &src, int64_t dim,
                                                 int64_t start, int64_t end, int64_t step);
  HBTL_EXPORTED LogicalResult SliceScatter(Tensor &fout, const Tensor &fin, const Tensor &src, int64_t dim,
                                           int64_t start, int64_t end, int64_t step);
  HBTL_EXPORTED LogicalResult TileConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &multiplies);
  HBTL_EXPORTED LogicalResult Tile(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &multiplies);
  HBTL_EXPORTED LogicalResult TransposeConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &permutes);
  HBTL_EXPORTED LogicalResult Transpose(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &permutes);
  HBTL_EXPORTED LogicalResult Flip(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims);
  HBTL_EXPORTED LogicalResult FlipConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims);
  HBTL_EXPORTED LogicalResult Roll(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &shifts,
                                   const std::vector<int64_t> &dims);
  HBTL_EXPORTED LogicalResult RollConfig(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &shifts,
                                         const std::vector<int64_t> &dims);
  HBTL_EXPORTED LogicalResult Stack(Tensor &fout, const std::vector<Tensor> &fin, int64_t dim);
  HBTL_EXPORTED LogicalResult StackConfig(Tensor &fout, const std::vector<Tensor> &fin, int64_t dim);
  HBTL_EXPORTED LogicalResult Identical(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult IdenticalConfig(Tensor &fout, const Tensor &fin);
  HBTL_EXPORTED LogicalResult Sort(Tensor &fout, Tensor &indices, const Tensor &fin, const int64_t &dim,
                                   const bool &descending, const bool &stable);
  HBTL_EXPORTED LogicalResult SortConfig(Tensor &fout, Tensor &indices, const Tensor &fin, const int64_t &dim,
                                         const bool &descending, const bool &stable);

  // gather series
  HBTL_EXPORTED LogicalResult IndexSelect(Tensor &fout, const Tensor &fin, const Tensor &index, int64_t dim);
  HBTL_EXPORTED LogicalResult IndexSelectConfig(Tensor &fout, const Tensor &fin, const Tensor &index, int64_t dim);
  HBTL_EXPORTED LogicalResult GatherElements(Tensor &fout, const Tensor &fin, const Tensor &index, int64_t dim);
  HBTL_EXPORTED LogicalResult GatherElementsConfig(Tensor &fout, const Tensor &fin, const Tensor &index, int64_t dim);
  HBTL_EXPORTED LogicalResult GatherND(Tensor &fout, const Tensor &fin, const Tensor &indices, int64_t batchDim);
  HBTL_EXPORTED LogicalResult GatherNDConfig(Tensor &fout, const Tensor &fin, const Tensor &index, int64_t batchDim);

  HBTL_EXPORTED LogicalResult ScatterElement(Tensor &fout, const Tensor &data, const Tensor &indices,
                                             const Tensor &update, int64_t dim, const std::string &reduce);
  HBTL_EXPORTED LogicalResult ScatterElementConfig(Tensor &fout, const Tensor &data, const Tensor &indices,
                                                   const Tensor &update, int64_t dim, const std::string &reduce);
  HBTL_EXPORTED LogicalResult ScatterMean(Tensor &fout, const Tensor &data, const Tensor &indices, const Tensor &update,
                                          int64_t dim, const std::string &reduce);
  HBTL_EXPORTED LogicalResult ScatterMeanConfig(Tensor &fout, const Tensor &data, const Tensor &indices,
                                                const Tensor &update, int64_t dim, const std::string &reduce);
  HBTL_EXPORTED LogicalResult ScatterND(Tensor &fout, const Tensor &data, const Tensor &indices, const Tensor &update,
                                        const std::string &reduce);
  HBTL_EXPORTED LogicalResult ScatterNDConfig(Tensor &fout, const Tensor &fin, const Tensor &indices,
                                              const Tensor &update, const std::string &reduce);

  HBTL_EXPORTED LogicalResult MatMul(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult MatMulConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs);
  HBTL_EXPORTED LogicalResult Correlation(Tensor &fout, const Tensor &lhs, const Tensor &rhs, int64_t kernel_size = 1,
                                          int64_t pad_size = 0, int64_t stride1 = 1, int64_t stride2 = 1,
                                          int64_t max_displacement = 1, bool isMultiply = true);
  HBTL_EXPORTED LogicalResult CorrelationConfig(Tensor &fout, const Tensor &lhs, const Tensor &rhs,
                                                int64_t kernel_size = 1, int64_t pad_size = 0, int64_t stride1 = 1,
                                                int64_t stride2 = 1, int64_t max_displacement = 1,
                                                bool isMultiply = true);

  // recurrent series

  HBTL_EXPORTED LogicalResult Gru(Tensor &fout, Tensor &hn, const Tensor &fin, const std::vector<Tensor> &weightIh,
                                  const std::vector<Tensor> &biasIh, const std::vector<Tensor> &weightHh,
                                  const std::vector<Tensor> &biasHh, const std::vector<Tensor> &weightIhReverse,
                                  const std::vector<Tensor> &biasIhReverse, const std::vector<Tensor> &weightHhReverse,
                                  const std::vector<Tensor> &biasHhReverse, const Tensor &h0, int64_t numLayer,
                                  bool batchFirst, bool biDirectional);

  HBTL_EXPORTED LogicalResult Lstm(Tensor &fout, Tensor &hn, Tensor &cn, const Tensor &fin,
                                   const std::vector<Tensor> &weightIh, const std::vector<Tensor> &biasIh,
                                   const std::vector<Tensor> &weightHh, const std::vector<Tensor> &biasHh,
                                   const std::vector<Tensor> &weightIhReverse, const std::vector<Tensor> &biasIhReverse,
                                   const std::vector<Tensor> &weightHhReverse, const std::vector<Tensor> &biasHhReverse,
                                   const Tensor &h0, const Tensor &c0, int64_t numLayer, bool batchFirst,
                                   bool biDirectional);

  HBTL_EXPORTED LogicalResult Rnn(Tensor &fout, Tensor &hn, const Tensor &fin, const std::vector<Tensor> &weightIh,
                                  const std::vector<Tensor> &biasIh, const std::vector<Tensor> &weightHh,
                                  const std::vector<Tensor> &biasHh, const std::vector<Tensor> &weightIhReverse,
                                  const std::vector<Tensor> &biasIhReverse, const std::vector<Tensor> &weightHhReverse,
                                  const std::vector<Tensor> &biasHhReverse, const Tensor &h0, int64_t numLayer,
                                  const std::string &nonLinearity, bool batchFirst, bool biDirectional);

  HBTL_EXPORTED LogicalResult PointPillarScatter(Tensor &fout, const Tensor &voxels, const Tensor &coords,
                                                 const std::vector<int64_t> &outShape);
  HBTL_EXPORTED LogicalResult PointPillarScatterConfig(Tensor &fout, const Tensor &voxels, const Tensor &coords,
                                                       const std::vector<int64_t> &outShape);

  HBTL_EXPORTED LogicalResult PointPillarPreProcess(Tensor &voxels, Tensor &coords, const Tensor &points,
                                                    const std::vector<double> &pcRanges,
                                                    const std::vector<double> &normRanges,
                                                    const std::vector<int64_t> &normDims,
                                                    const std::vector<double> &voxelSizes, const int64_t &maxVoxelNum,
                                                    const int64_t &maxPointsPerVoxel);
  HBTL_EXPORTED LogicalResult PointPillarPreProcessConfig(Tensor &voxels, Tensor &coords, const Tensor &points,
                                                          const std::vector<double> &pcRanges,
                                                          const std::vector<double> &normRanges,
                                                          const std::vector<int64_t> &normDims,
                                                          const std::vector<double> &voxelSizes,
                                                          const int64_t &maxVoxelNum, const int64_t &maxPointsPerVoxel);

  // misc series
  inline const char *getNopName() { return "native::View"; }
  HBTL_EXPORTED LogicalResult View(Tensor &fout, const Tensor &fin0);
  HBTL_EXPORTED LogicalResult View(Tensor &fout, const Tensor &fin0, const Tensor &fin1); // do nothing
  HBTL_EXPORTED LogicalResult View(Tensor &fout, const Tensor &fin0, const Tensor &fin1,
                                   const Tensor &fin2); // do nothing
  HBTL_EXPORTED LogicalResult View(Tensor &fout, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2,
                                   const Tensor &fin3); // do nothing
  HBTL_EXPORTED LogicalResult Rope(Tensor &fout, const Tensor &fin, const Tensor &position_id);

  HBTL_EXPORTED LogicalResult RopeConfig(Tensor &fout, const Tensor &fins, const Tensor &position_id);

  HBTL_EXPORTED LogicalResult TopK(Tensor &values, Tensor &indices, const Tensor &fin, const int64_t &k,
                                   const int64_t &dim, const bool &largest, const bool &sorted);

  HBTL_EXPORTED LogicalResult TopKConfig(Tensor &values, Tensor &indices, const Tensor &fin, const int64_t &k,
                                         const int64_t &dim, const bool &largest, const bool &sorted);

  HBTL_EXPORTED LogicalResult Filter(Tensor &maxValue, Tensor &maxIndex, Tensor &filterCoord, Tensor &filterData,
                                     const Tensor &data, int64_t channelBegin, int64_t channelEnd, double threshold);
  HBTL_EXPORTED LogicalResult FilterConfig(Tensor &maxValue, Tensor &maxIndex, Tensor &filterCoord, Tensor &filterData,
                                           const Tensor &data, int64_t channelBegin, int64_t channelEnd,
                                           double threshold);

  HBTL_EXPORTED LogicalResult NanToNum(Tensor &fout, const Tensor &fin, double, double posInf, double negInf);

  HBTL_EXPORTED LogicalResult NanToNumConfig(Tensor &fout, const Tensor &fin, double nan, double posInf, double negInf);

  HBTL_EXPORTED LogicalResult NonMaxSuppression(Tensor &selectedIndices, const Tensor &boxes, const Tensor &scores,
                                                const std::string &boxType, float iouThreshold, float scoresThreshold,
                                                int64_t maxOutputBoxesPerClass);

  HBTL_EXPORTED LogicalResult NonMaxSuppressionConfig(Tensor &selectedIndices, const Tensor &boxes,
                                                      const Tensor &scores, const std::string &boxType,
                                                      float iouThreshold, float scoresThreshold,
                                                      int64_t maxOutputBoxesPerClass);

  /// unpool nd series
  /**
   * @brief 2d max pooling with indices
   *        Acting like torch.nn.MaxPool2d to produce output and indices
   *
   * @param fout  output tensor vector, dtype of first element could be float/double, dtype of second element is int64_t
   * @param fin  input tensor, dtype could be float/doßuble
   * @param kernel  the size of the kernel window
   * @param stride  the stride of the kernel window
   * @param pad  implicit -INF padding to be added on single sides
   * @param dilation  a parameter that controls the stride of elements in the window
   * @param p
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult MaxPoolWithIndicesChannelLast(Tensor &fout, Tensor &indices, const Tensor &fin,
                                                            const std::vector<int64_t> &kernel,
                                                            const std::vector<int64_t> &stride,
                                                            const std::vector<int64_t> &pad,
                                                            const std::vector<int64_t> &dilation, bool isCeilMode);
  HBTL_EXPORTED LogicalResult MaxPoolWithIndicesChannelLastConfig(Tensor &fout, Tensor &indices, const Tensor &fin,
                                                                  const std::vector<int64_t> &kernel,
                                                                  const std::vector<int64_t> &stride,
                                                                  const std::vector<int64_t> &pad,
                                                                  const std::vector<int64_t> &dilation,
                                                                  bool isCeilMode);

  /**
   * @brief 2d max unpool
   *        Acting like torch.nn.MaxUnpool2d to compute a partial inverse of torch.nn.MaxPool2d
   *
   * @param fout  output tensor vector, dtype could be float/double
   * @param fin  input tensor, dtype could be float/double
   * @param indices input tensor, dtype could be int64_t
   * @param kernel  the size of the kernel window
   * @param stride  the stride of the kernel window
   * @param pad  implicit -INF padding to be added on single sides
   * @param dilation  a parameter that controls the stride of elements in the window
   * @param p
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult MaxUnpoolChannelLast(Tensor &fout, const Tensor &fin, const Tensor &indices,
                                                   const std::vector<int64_t> &outputShape,
                                                   const std::vector<int64_t> &kernel,
                                                   const std::vector<int64_t> &stride, const std::vector<int64_t> &pad);
  HBTL_EXPORTED LogicalResult MaxUnpoolChannelLastConfig(Tensor &fout, const Tensor &fin, const Tensor &indices,
                                                         const std::vector<int64_t> &outputShape,
                                                         const std::vector<int64_t> &kernel,
                                                         const std::vector<int64_t> &stride,
                                                         const std::vector<int64_t> &pad);

  /**
   * @brief hbir::ShuffleFactorOp
   *
   * @param fout  output tensor
   * @param fin  input tensor
   * @param dims dims attribute, int64_t vector
   * @param factors factors attribute, int64_t vector
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult ShuffleFactor(Tensor &fout, const Tensor &fin, const std::vector<int64_t> &dims,
                                            const std::vector<int64_t> &factors);

  /**
   * @brief hbir::KvCacheUpdateOp
   *
   * @param fout  output tensor
   * @param updatedScale  output tensor
   * @param updatedInput  output tensor
   * @param fin  input tensor
   * @param cachedScale  input tensor
   * @param cachedInput  input tensor
   * @return LogicalResult
   */
  HBTL_EXPORTED LogicalResult KvCacheUpdate(Tensor &fout, Tensor &cachedOut, const Tensor &cachedInput,
                                            const Tensor &input, int64_t dim, int64_t repeatNum);
  HBTL_EXPORTED LogicalResult KvCacheUpdateConfig(Tensor &fout, Tensor &cachedOut, const Tensor &cachedInput,
                                                  const Tensor &input, int64_t dim, int64_t repeatNum);

  HBTL_EXPORTED LogicalResult DummyOutput(Tensor &fout);
  HBTL_EXPORTED LogicalResult DummyOutputConfig(Tensor &fout);
  } // namespace native
}
HBTL_NAMESPACE_END
