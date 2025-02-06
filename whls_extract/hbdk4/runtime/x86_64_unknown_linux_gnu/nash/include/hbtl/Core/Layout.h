#pragma once

#ifndef HBTL_SUPPORT_LAYOUT_H_
#define HBTL_SUPPORT_LAYOUT_H_

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/ADT/STLExtras.h"
#include "hbtl/ADT/iterator.h"
#include "hbtl/Core/ElementType.h"
#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/Compiler.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

HBTL_NAMESPACE_BEGIN {

  struct BlockAxis {
    BlockAxis() = default;
    BlockAxis(const BlockAxis &) = default;
    BlockAxis(BlockAxis &) = default;
    BlockAxis &operator=(const BlockAxis &) = default;
    BlockAxis &operator=(BlockAxis &) = default;

    int64_t axis = 0;
    int64_t size = 0;
  };

  class HBTL_EXPORTED Block {

  public:
    Block() = default;

    explicit Block(ArrayRef<BlockAxis> axes) : blockAxes({axes.begin(), axes.end()}) {}

    static Block create(ArrayRef<int64_t> axes, ArrayRef<int64_t> sizes) {
      assert(axes.size() == sizes.size() && "");

      std::vector<BlockAxis> blockAxes;
      blockAxes.reserve(axes.size());

      for (auto e : zip(axes, sizes)) {
        blockAxes.push_back({std::get<0>(e), std::get<1>(e)});
      }
      return Block(blockAxes);
    }

    [[nodiscard]] int64_t getBlockSize() const {
      int64_t blockSize = 1;
      for (const auto &blockAxis : blockAxes) {
        blockSize *= blockAxis.size;
      }
      return blockSize;
    }

    [[nodiscard]] std::vector<int64_t> getNativeShape(ArrayRef<int64_t> outerShape) const;
    [[nodiscard]] std::vector<int64_t> getOuterShape(ArrayRef<int64_t> nativeShape) const;

    [[nodiscard]] Tensor toNative(const Tensor &layout) const;
    [[nodiscard]] Tensor fromNative(const Tensor &native) const;

    [[nodiscard]] Block transpose(ArrayRef<int64_t> permutes) const {
      std::vector<BlockAxis> newBlockAxes;
      for (auto blockAxis : blockAxes) {
        const auto *found = find(permutes, blockAxis.axis + permutes.size());
        assert(found != permutes.end());
        blockAxis.axis = (found - permutes.begin()) - static_cast<int64_t>(permutes.size());
        newBlockAxes.push_back(blockAxis);
      }
      return Block{newBlockAxes};
    }

    [[nodiscard]] Block expand(int64_t curTensorRank, ElementType curTensorType) const {
      auto newBlockAxes = blockAxes;
      if (newBlockAxes.empty()) {        // native
        newBlockAxes.push_back({-1, 1}); // 1c
      }

      auto blockFactor = getByteSize(curTensorType) / getByteSize(hbtl::ElementType::opaque8);
      if (blockFactor > 1) {
        if (newBlockAxes.back().axis == -1 || newBlockAxes.back().axis == curTensorRank - 1) {
          newBlockAxes.back().size *= blockFactor;
        } else {
          hbtl::BlockAxis blockAxis;
          if (newBlockAxes.back().axis < 0) {
            blockAxis = hbtl::BlockAxis{-1, blockFactor};
          } else {
            blockAxis = hbtl::BlockAxis{curTensorRank - 1, blockFactor};
          }

          newBlockAxes.push_back(blockAxis);
        }
      }
      auto newBlockInfo = hbtl::Block(newBlockAxes);
      return newBlockInfo;
    }

    std::vector<BlockAxis> blockAxes;
  };
}
HBTL_NAMESPACE_END

#endif // HBTL_SUPPORT_LAYOUT_H_
