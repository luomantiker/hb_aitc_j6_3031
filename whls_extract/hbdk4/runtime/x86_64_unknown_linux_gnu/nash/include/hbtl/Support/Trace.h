// HBTL TraceType and Trace Macros

#pragma once

#include "hbtl/ADT/ArrayRef.h" // IWYU pragma: keep
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/Context.h"
#include <sstream>
#include <string> // IWYU pragma: keep
#include <utility>

HBTL_NAMESPACE_BEGIN {

  template <bool traceEn_> class TraceScope {
  public:
    static constexpr bool traceEn = traceEn_;
    TraceScope() = delete;
    explicit TraceScope(ArrayRef<int64_t> point) : point(point) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        assert(!point.empty());
      } else {
        assert(point.empty());
      }
    }

  private:
    template <typename... ArgT>
    bool _update(HBTL_MAYBE_UNUSED size_t start, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord, ArgT... arg) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        auto thisP = point.drop_front(start);

        if (thisP.size() < coord.size()) { // not enough point
          return false;
        }

        for (auto i = 0U; i < coord.size(); ++i) {
          if (thisP[i] != coord[i]) {
            return false;
          }
        }
        return _update(start + coord.size(), arg...);
      }
      return false;
    }

    bool _update(HBTL_MAYBE_UNUSED size_t start, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        auto thisP = point.drop_front(start);

        if (thisP.size() != coord.size()) { // not enough point
          return false;
        }

        return thisP == coord;
      }
      return false;
    }

  public:
    /// update verbose flag if given one coord is identical to the point. e.g. if point = {1,2,3} then update({1,2,3})
    /// yeilds true
    inline void update(HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord0) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        verbose = _update(0, coord0);
      }
    }
    /// update verbose flag if given two coord is identical to the point. e.g. if point = {1,2,3} then
    /// update({1,2},{3}) yeilds true
    inline void update(HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord0, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord1) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        verbose = _update(0, coord0, coord1);
      }
    }
    /// update verbose flag if given coords is identical to target coord
    inline void update(HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord0, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord1,
                       HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord2) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        verbose = _update(0, coord0, coord1, coord2);
      }
    }
    /// update verbose flag if given coords is identical to target coord
    inline void update(HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord0, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord1,
                       HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord2, HBTL_MAYBE_UNUSED ArrayRef<int64_t> coord3) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        verbose = _update(0, coord0, coord1, coord2, coord3);
      }
    }

    /// log only if verbose is true and traceEn is true
    template <typename... Args> void trace(const char *info) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        if (verbose) {
          Context::get()->info(info);
        }
      }
    }

    /// always log if traceEn is true
    template <typename... Args> void remark(const char *info) {
      if HBTL_CONSTEXPR_IF (traceEn) {
        Context::get()->info(info);
      }
    }
    ArrayRef<int64_t> point;
    bool verbose = false;
  };

  /// run with trace. if point is not empty then func can get constexpr traceEn from value container. e.g. constexpr
  /// bool traceEn = decltype(v)::value
  template <typename Func> void runWithTrace(const Func &func) {
    auto point = Context::get()->getTracePoint();
    if (point.empty()) {
      (void)func(TraceScope<false>(point));
    } else {
      (void)func(TraceScope<true>(point));
    }
  }

  template <typename... ArgT> std::string _formatCoord(ArrayRef<int64_t> coord, const ArgT &...arg) {
    std::stringstream ss;
    ss << ArrayRef<int64_t>(join(coord, arg...));
    return ss.str();
  }
  /// format one coord to string. e.g. {1,2} -> "1,2"
  inline std::string formatCoord(ArrayRef<int64_t> coord0) { return _formatCoord(coord0); }
  /// format two coord to string. e.g. {1,2},{3,4} -> "1,2,3,4"
  inline std::string formatCoord(ArrayRef<int64_t> coord0, ArrayRef<int64_t> coord1) {
    return _formatCoord(coord0, coord1);
  }
  /// format three coords to string.  e.g. {1,2},{3},4 -> "1,2,3,4"
  inline std::string formatCoord(ArrayRef<int64_t> coord0, ArrayRef<int64_t> coord1, ArrayRef<int64_t> coord2) {
    return _formatCoord(coord0, coord1, coord2);
  }
  /// format four coords to string. e.g. {1},{2},3,4 -> "1,2,3,4"
  inline std::string formatCoord(ArrayRef<int64_t> coord0, ArrayRef<int64_t> coord1, ArrayRef<int64_t> coord2,
                                 ArrayRef<int64_t> coord3) {
    return _formatCoord(coord0, coord1, coord2, coord3);
  }
}
HBTL_NAMESPACE_END
