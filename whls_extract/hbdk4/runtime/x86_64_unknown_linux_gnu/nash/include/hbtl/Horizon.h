/// Native kernels
#pragma once

#include "hbtl/Support/Compiler.h"
#include "hbtl/Tensor.h"

HBTL_NAMESPACE_BEGIN {
  namespace horizon {

  // image
  HBTL_EXPORTED LogicalResult Nv12ToYuv444Sub128(Tensor &yuv, const Tensor &y, const Tensor &uv);
  HBTL_EXPORTED LogicalResult Nv12ToYuv444Sub128Config(Tensor &yuv, const Tensor &y, const Tensor &uv);

  HBTL_EXPORTED LogicalResult Pyramid(Tensor &yuv, const Tensor &y);
  HBTL_EXPORTED LogicalResult PyramidConfig(Tensor &yuv, const Tensor &y);

  HBTL_EXPORTED LogicalResult ImagePreProcess(Tensor &fout, const Tensor &fin, const std::string &cscMode,
                                              int64_t divisor, const std::vector<double> &mean,
                                              const std::vector<double> &sd);
  HBTL_EXPORTED LogicalResult ImagePreProcessConfig(Tensor &fout, const Tensor &fin, const std::string &cscMode,
                                                    int64_t divisor, const std::vector<double> &mean,
                                                    const std::vector<double> &sd);

  // postprocess series
  HBTL_EXPORTED LogicalResult
  DetectionPostProcessFloat(std::vector<Tensor> &outBoxes, std::vector<std::vector<float_t>> &outScores,
                            std::vector<std::vector<int64_t>> &outClsIndexes, const std::vector<Tensor> &fins,
                            const std::vector<Tensor> &anchors, const std::vector<int64_t> &clsOffsets,
                            float_t boxFilterThresh, float_t nmsIoUThresh, uint64_t nmsPreTopK, uint64_t nmsPostTopK);

  HBTL_EXPORTED LogicalResult DetectionPostProcessQuantized(
      Tensor &boxData, const std::vector<Tensor> &inputs, const std::vector<int64_t> &anchors,
      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin,
      const std::vector<int64_t> &clsIdxOffset, int64_t seed, const std::vector<int64_t> &useClipping,
      const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize, int64_t inputShift,
      int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult DetectionPostProcessQuantizedConfig(
      Tensor &boxData, const std::vector<Tensor> &inputs, const std::vector<int64_t> &anchors,
      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin,
      const std::vector<int64_t> &clsIdxOffset, int64_t seed, const std::vector<int64_t> &useClipping,
      const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize, int64_t inputShift,
      int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult DppQuantized1(Tensor &boxData, const Tensor &fin0, const std::vector<int64_t> &anchors,
                                            const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold,
                                            int64_t nmsThreshold, int64_t nmsMargin,
                                            const std::vector<int64_t> &clsIdxOffset, int64_t seed,
                                            const std::vector<int64_t> &useClipping,
                                            const std::vector<int64_t> &imageStrides,
                                            const std::vector<int64_t> &imageSize, int64_t inputShift,
                                            int64_t maxBoxNum);
  HBTL_EXPORTED LogicalResult
  DppQuantized1Config(Tensor &boxData, const Tensor &fin0, const std::vector<int64_t> &anchors,
                      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold,
                      int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset, int64_t seed,
                      const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult
  DppQuantized2(Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const std::vector<int64_t> &anchors,
                const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold,
                int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset, int64_t seed,
                const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);
  HBTL_EXPORTED LogicalResult
  DppQuantized2Config(Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const std::vector<int64_t> &anchors,
                      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold,
                      int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset, int64_t seed,
                      const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult DppQuantized3(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const std::vector<int64_t> &anchors,
      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin,
      const std::vector<int64_t> &clsIdxOffset, int64_t seed, const std::vector<int64_t> &useClipping,
      const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize, int64_t inputShift,
      int64_t maxBoxNum);
  HBTL_EXPORTED LogicalResult DppQuantized3Config(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const std::vector<int64_t> &anchors,
      const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin,
      const std::vector<int64_t> &clsIdxOffset, int64_t seed, const std::vector<int64_t> &useClipping,
      const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize, int64_t inputShift,
      int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult DppQuantized4(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const Tensor &fin3,
      const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold,
      int64_t nmsThreshold, int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset, int64_t seed,
      const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);
  HBTL_EXPORTED LogicalResult DppQuantized4Config(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const Tensor &fin3,
      const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec, int64_t boxFilterThreshold,
      int64_t nmsThreshold, int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset, int64_t seed,
      const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult DppQuantized5(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const Tensor &fin3,
      const Tensor &fin4, const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
      int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset,
      int64_t seed, const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);
  HBTL_EXPORTED LogicalResult DppQuantized5Config(
      Tensor &boxData, const Tensor &fin0, const Tensor &fin1, const Tensor &fin2, const Tensor &fin3,
      const Tensor &fin4, const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
      int64_t boxFilterThreshold, int64_t nmsThreshold, int64_t nmsMargin, const std::vector<int64_t> &clsIdxOffset,
      int64_t seed, const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
      const std::vector<int64_t> &imageSize, int64_t inputShift, int64_t maxBoxNum);

  HBTL_EXPORTED LogicalResult
  FilterPostProcess(Tensor &boxData, const std::vector<Tensor> &inputs, const std::vector<int64_t> &anchors,
                    const std::vector<int64_t> &anchorNumVec, const std::vector<int64_t> &clsIdxOffset,
                    const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                    const std::vector<int64_t> &imageSize, const std::vector<int64_t> &inputChannel,
                    const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult
  FilterPostProcessConfig(Tensor &boxData, const std::vector<Tensor> &inputs, const std::vector<int64_t> &anchors,
                          const std::vector<int64_t> &anchorNumVec, const std::vector<int64_t> &clsIdxOffset,
                          const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                          const std::vector<int64_t> &imageSize, const std::vector<int64_t> &inputChannel,
                          const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP1(Tensor &boxData, const Tensor &input0, const std::vector<int64_t> &anchors,
                                   const std::vector<int64_t> &anchorNumVec, const std::vector<int64_t> &clsIdxOffset,
                                   const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                                   const std::vector<int64_t> &imageSize, const std::vector<int64_t> &inputChannel,
                                   const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP1Config(Tensor &boxData, const Tensor &input0, const std::vector<int64_t> &anchors,
                                         const std::vector<int64_t> &anchorNumVec,
                                         const std::vector<int64_t> &clsIdxOffset,
                                         const std::vector<int64_t> &useClipping,
                                         const std::vector<int64_t> &imageStrides,
                                         const std::vector<int64_t> &imageSize,
                                         const std::vector<int64_t> &inputChannel,
                                         const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP2(Tensor &boxData, const Tensor &input0, const Tensor &input1,
                                   const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
                                   const std::vector<int64_t> &clsIdxOffset, const std::vector<int64_t> &useClipping,
                                   const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize,
                                   const std::vector<int64_t> &inputChannel, const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP2Config(Tensor &boxData, const Tensor &input0, const Tensor &input1,
                                         const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
                                         const std::vector<int64_t> &clsIdxOffset,
                                         const std::vector<int64_t> &useClipping,
                                         const std::vector<int64_t> &imageStrides,
                                         const std::vector<int64_t> &imageSize,
                                         const std::vector<int64_t> &inputChannel,
                                         const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP3(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2,
                                   const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
                                   const std::vector<int64_t> &clsIdxOffset, const std::vector<int64_t> &useClipping,
                                   const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize,
                                   const std::vector<int64_t> &inputChannel, const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult
  FPP3Config(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2,
             const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
             const std::vector<int64_t> &clsIdxOffset, const std::vector<int64_t> &useClipping,
             const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize,
             const std::vector<int64_t> &inputChannel, const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP4(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2,
                                   const Tensor &input3, const std::vector<int64_t> &anchors,
                                   const std::vector<int64_t> &anchorNumVec, const std::vector<int64_t> &clsIdxOffset,
                                   const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                                   const std::vector<int64_t> &imageSize, const std::vector<int64_t> &inputChannel,
                                   const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult
  FPP4Config(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2, const Tensor &input3,
             const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
             const std::vector<int64_t> &clsIdxOffset, const std::vector<int64_t> &useClipping,
             const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize,
             const std::vector<int64_t> &inputChannel, const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FPP5(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2,
                                   const Tensor &input3, const Tensor &input4, const std::vector<int64_t> &anchors,
                                   const std::vector<int64_t> &anchorNumVec, const std::vector<int64_t> &clsIdxOffset,
                                   const std::vector<int64_t> &useClipping, const std::vector<int64_t> &imageStrides,
                                   const std::vector<int64_t> &imageSize, const std::vector<int64_t> &inputChannel,
                                   const std::vector<int64_t> &fppSi64Param);
  HBTL_EXPORTED LogicalResult
  FPP5Config(Tensor &boxData, const Tensor &input0, const Tensor &input1, const Tensor &input2, const Tensor &input3,
             const Tensor &input4, const std::vector<int64_t> &anchors, const std::vector<int64_t> &anchorNumVec,
             const std::vector<int64_t> &clsIdxOffset, const std::vector<int64_t> &useClipping,
             const std::vector<int64_t> &imageStrides, const std::vector<int64_t> &imageSize,
             const std::vector<int64_t> &inputChannel, const std::vector<int64_t> &fppSi64Param);

  HBTL_EXPORTED LogicalResult FilterCopy(Tensor &fout0, Tensor &fout1, Tensor &fout2, Tensor &fout3, const Tensor &fin,
                                         const int64_t &channelNum, const int64_t &padByteSize);
  HBTL_EXPORTED LogicalResult FilterCopyConfig(Tensor &fout0, Tensor &fout1, Tensor &fout2, Tensor &fout3,
                                               const Tensor &fin, const int64_t &channelNum,
                                               const int64_t &padByteSize);

  HBTL_EXPORTED LogicalResult RoiAlign(Tensor &fout, const std::vector<Tensor> &fin, const std::vector<int64_t> &shape,
                                       const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                       const std::string &interpolateMode, int64_t canonicalBoxSize,
                                       int64_t canonicalLevel, const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlignConfig(Tensor &fout, const std::vector<Tensor> &fin,
                                             const std::vector<int64_t> &shape,
                                             const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                             const std::string &interpolateMode, int64_t canonicalBoxSize,
                                             int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult RoiAlign1(Tensor &fout, const Tensor &rois, const Tensor &fin,
                                        const std::vector<int64_t> &shape, const std::vector<int64_t> &featureStrides,
                                        int64_t samplingRation, const std::string &interpolateMode,
                                        int64_t canonicalBoxSize, int64_t canonicalLevel,
                                        const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlign1Config(Tensor &fout, const Tensor &rois, const Tensor &fin,
                                              const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                              const std::string &interpolateMode, int64_t canonicalBoxSize,
                                              int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult RoiAlign2(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                        const std::vector<int64_t> &shape, const std::vector<int64_t> &featureStrides,
                                        int64_t samplingRation, const std::string &interpolateMode,
                                        int64_t canonicalBoxSize, int64_t canonicalLevel,
                                        const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlign2Config(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                              const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                              const std::string &interpolateMode, int64_t canonicalBoxSize,
                                              int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult RoiAlign3(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                        const Tensor &fin2, const std::vector<int64_t> &shape,
                                        const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                        const std::string &interpolateMode, int64_t canonicalBoxSize,
                                        int64_t canonicalLevel, const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlign3Config(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                              const Tensor &fin2, const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                              const std::string &interpolateMode, int64_t canonicalBoxSize,
                                              int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult RoiAlign4(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                        const Tensor &fin2, const Tensor &fin3, const std::vector<int64_t> &shape,
                                        const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                        const std::string &interpolateMode, int64_t canonicalBoxSize,
                                        int64_t canonicalLevel, const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlign4Config(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                              const Tensor &fin2, const Tensor &fin3, const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                              const std::string &interpolateMode, int64_t canonicalBoxSize,
                                              int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult RoiAlign5(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                        const Tensor &fin2, const Tensor &fin3, const Tensor &fin4,
                                        const std::vector<int64_t> &shape, const std::vector<int64_t> &featureStrides,
                                        int64_t samplingRation, const std::string &interpolateMode,
                                        int64_t canonicalBoxSize, int64_t canonicalLevel,
                                        const std::vector<double> &boxClipRatio);
  HBTL_EXPORTED LogicalResult RoiAlign5Config(Tensor &fout, const Tensor &rois, const Tensor &fin0, const Tensor &fin1,
                                              const Tensor &fin2, const Tensor &fin3, const Tensor &fin4,
                                              const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &featureStrides, int64_t samplingRation,
                                              const std::string &interpolateMode, int64_t canonicalBoxSize,
                                              int64_t canonicalLevel, const std::vector<double> &boxClipRatio);

  HBTL_EXPORTED LogicalResult Correlation(Tensor &fout, const Tensor &lhs, const Tensor &rhs, int64_t kernelSize = 1,
                                          int64_t padSize = 0, int64_t stride1 = 1, int64_t stride2 = 1,
                                          int64_t maxDisplacement = 1, bool isMultiply = true, int64_t rshift = 0,
                                          int64_t startw = 0, int64_t processDone = 0);

  HBTL_EXPORTED LogicalResult RcnnPostProcess(Tensor &fout0, Tensor &fout1, const Tensor &bbox, const Tensor &score,
                                              const Tensor &delta, int64_t img_h, int64_t img_w, float nms_threshold,
                                              float score_threshold, int64_t cls_number, int64_t top_n,
                                              const std::vector<float> &bbox_delta_mean,
                                              const std::vector<float> &bbox_delta_std, bool image_size_fixed);
  HBTL_EXPORTED LogicalResult RcnnPostProcessConfig(Tensor &fout0, Tensor &fout1, const Tensor &bbox,
                                                    const Tensor &score, const Tensor &delta, int64_t img_h,
                                                    int64_t img_w, float nms_threshold, float score_threshold,
                                                    int64_t cls_number, int64_t top_n,
                                                    const std::vector<float> &bbox_delta_mean,
                                                    const std::vector<float> &bbox_delta_std, bool image_size_fixed);

  HBTL_EXPORTED LogicalResult RunLengthEncode(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult RunLengthEncodeConfig(Tensor &fout, const Tensor &fin);

  HBTL_EXPORTED LogicalResult RlePostProcess(Tensor &fout, const Tensor &fin, int64_t validNum, int64_t busAlign);
  HBTL_EXPORTED LogicalResult RlePostProcessConfig(Tensor &fout, const Tensor &fin, int64_t validNum, int64_t busAlign);

  } // namespace horizon
}
HBTL_NAMESPACE_END
