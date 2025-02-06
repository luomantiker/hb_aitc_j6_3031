#pragma once

#include "hbtl/Support/Compiler.h"
#include <asm/unistd_64.h>
#include <cassert>
#ifndef __linux__
#error "This file only works under linux"
#endif // __linux__

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <asm/unistd.h>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

HBTL_NAMESPACE_BEGIN {

  /// Requires CPU with hardware performance counter
  ///
  /// NOTE(hehaoqian):
  /// Some of machines in bsub pool are from AMD,
  /// which does not have hw perf counter
  ///
  /// Set the following code in CMake to tell test frame about this
  ///
  /// ```cmake
  /// ENV_NAMES
  /// HBDK_TEST_FRAME_NEED_HW_PERF
  /// ENV_VALUES
  /// 1
  /// ENV_MODES
  /// OVERRIDE
  /// ```
  ///
  /// See `hbtl/unittests/Benchmark/CMakeLists.txt`
  class Perf {
  public:
    Perf() { initPerfEvent(instCount, PERF_COUNT_HW_INSTRUCTIONS); }

    void begin() const {
      [[maybe_unused]] int ret = ioctl(instCount.fd, PERF_EVENT_IOC_RESET, 0);
      ret = ioctl(instCount.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    void end() const { (void)ioctl(instCount.fd, PERF_EVENT_IOC_DISABLE, 0); }

    uint64_t getResult() {
      uint64_t ret = 0;
      __sync_synchronize();
      ret = samplePerfEvent(instCount);
      __sync_synchronize();
      return ret;
    }

  private:
    struct PerfResult {
      uint64_t instCount = 0;
    };

    struct PerfEvent {
      int fd = -1;
      struct perf_event_attr attr = {};
      PerfEvent() = default;
      PerfEvent(const PerfResult &) = delete;
      ~PerfEvent() {
        if (fd > 0) {
          close(fd);
        }
      }
    };

    static void initPerfEvent(PerfEvent &pe, enum perf_hw_id hwId) {
      memset(&pe.attr, 0, sizeof(struct perf_event_attr));
      pe.attr.type = PERF_TYPE_HARDWARE;
      pe.attr.size = sizeof(struct perf_event_attr);
      pe.attr.config = hwId;
      pe.attr.disabled = 1;
      pe.attr.exclude_kernel = 1;
      pe.attr.exclude_hv = 1;

      pe.fd = perf_event_open(&pe.attr, 0, -1, -1, 0);
      assert(pe.fd >= 0 && "perf_event_open failed");
    }

    static uint64_t samplePerfEvent(PerfEvent &pe) {
      uint64_t result = 0;
      [[maybe_unused]] ssize_t retSize = read(pe.fd, &result, sizeof(result));
      assert(retSize == sizeof(result) && "Read perf result failed");
      return result;
    }

    static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, uint64_t flags) {
      return static_cast<int>(syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags));
    }

    PerfEvent instCount;
    PerfResult result;
  };
}
HBTL_NAMESPACE_END
