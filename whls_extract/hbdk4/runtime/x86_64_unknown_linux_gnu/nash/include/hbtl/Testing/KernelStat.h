#pragma once
#include "Perf.h"
#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/ADT/Random.h"
#include "hbtl/Core/ElementType.h"
#include "hbtl/Core/Tensor.h"
#include "hbtl/Support/ADTExtras.h"
#include "hbtl/Support/Context.h"
#include "hbtl/Support/LogicalResult.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

HBTL_NAMESPACE_BEGIN {

  template <size_t nums> class KernelStat {

  public:
    explicit KernelStat(const std::string &filename) : times(nums, 1), insts(nums, 0) { of = std::ofstream(filename); }

    template <typename T, typename... ArgTypes> void randomize(int64_t salt, T &&arg, ArgTypes &&...args) {
      if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
        if (isIntegral(arg.getType())) {
          runByElementType<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>(
              arg.getType(), [&](auto tv) {
                using DataType = typename decltype(tv)::type;
                auto carg = arg.contiguous();
                fillUniform<DataType>(arg.template getMutData<DataType>(), std::to_string(salt), 0,
                                      static_cast<DataType>(120));
                arg.copy(carg);
              });
        }
        if (isFloat(arg.getType())) {
          runByElementType<float, double>(arg.getType(), [&](auto tv) {
            using DataType = typename decltype(tv)::type;
            auto carg = arg.contiguous();
            fillNormal<DataType>(arg.template getMutData<DataType>(), 0, 1, std::to_string(salt));
            arg.copy(carg);
          });
        }
      }
      if constexpr (sizeof...(args) != 0) {
        randomize<ArgTypes...>(salt + 1, std::forward<ArgTypes>(args)...);
      }
    }

    template <typename FuncT, typename... ArgTypes> LogicalResult runner(FuncT impl, ArgTypes &&...args) {

      auto *ctx = Context::get();
      for (size_t i = 0; i < nums; ++i) {
        Context::get()->info(("run iter " + std::to_string(i)).c_str());
        randomize<ArgTypes...>(i * sizeof...(args), std::forward<ArgTypes>(args)...); // give each iter different values
        {                                                                             // measure times
          auto begin = std::chrono::high_resolution_clock::now();
          auto res = impl(args...);
          auto end = std::chrono::high_resolution_clock::now();
          if (failed(res)) {
            return res;
          }
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
          times[i] = static_cast<uint64_t>(duration.count());
        }

        { // measure insts
          ctx->setNumThreads(1);
          perf.begin();
          auto res = impl(args...);
          perf.end();
          if (failed(res)) {
            return res;
          }
          ctx->setNumThreads(4);
          insts[i] = perf.getResult();
          ctx->setNumThreads(16);
        }
      }
      return success();
    }

    [[nodiscard]] std::vector<uint64_t> getInsts() const { return insts; }
    [[nodiscard]] std::vector<uint64_t> getTimes() const { return times; }

    [[nodiscard]] double getAvgTime() const {
      double sum = 0;
      for (auto val : times) {
        sum += static_cast<double>(val);
      }
      return sum / static_cast<double>(nums);
    }

    [[nodiscard]] double getAvgInst() const {
      double sum = 0;
      for (auto val : insts) {
        sum += static_cast<double>(val);
      }
      return sum / static_cast<double>(nums) / 1000000000;
    }

    void record() {
      of << "kernel cost time(ms):\n";
      for (auto time : times) {
        of << time << std::endl;
      }
      of << "Run kernel " << nums << " times, the avg time cost: " << getAvgTime() << "ms" << std::endl;
      of << "##############" << std::endl;
      of << "kernel insts count:\n";
      for (auto inst : insts) {
        of << inst << std::endl;
      }
      of << "Run kernel " << nums << " times, the avg inst sampled: " << getAvgInst() << " giga insts" << std::endl;
    }

    ~KernelStat() { of.close(); }

  private:
    Perf perf;
    std::vector<uint64_t> times;
    std::vector<uint64_t> insts;
    std::ofstream of;
  };
}
HBTL_NAMESPACE_END
