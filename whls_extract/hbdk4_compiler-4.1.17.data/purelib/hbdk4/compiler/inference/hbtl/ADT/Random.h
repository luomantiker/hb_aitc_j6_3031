#pragma once

#include "hbtl/ADT/string_view.h"
#include <type_traits>

#include "hbtl/ADT/ArrayRef.h"
#include "hbtl/ADT/Distribution.h"
#include "hbtl/ADT/STLExtras.h"
#include "hbtl/Support/Compiler.h"
#include "hbtl/Support/MathExtras.h"

#include <random>
#include <system_error>

HBTL_NAMESPACE_BEGIN {

  template <typename T> class UniformRNG {

  public:
    explicit UniformRNG(hbtl::string_view salt, T lower = std::numeric_limits<T>::lowest(),
                        T upper = std::numeric_limits<T>::max())
        : lower(lower), upper(upper) {
      std::vector<uint32_t> data(salt.size(), 0);
      copy(salt, data.begin());
      // std::seed_seq has fixed output defined by C++11 standard
      std::seed_seq seed(data.begin(), data.end());
      rng.seed(seed);
    }

    /// get a random value
    T operator()() {
      // The values of std::xxx_distribution() is stdlib dependent
      // Use hbdk version instead
      if HBTL_CONSTEXPR_IF (std::is_integral<T>::value) {
        hbtl::uniform_int_distribution<T> dist(lower, upper); // dist for integral, [a,b]
        return dist(rng);
      } else {
        hbtl::uniform_real_distribution<T> dist(lower, upper); // dist for float, [a,b)
        return dist(rng);
      }
    }

    /// fill uniform random data to a mutable array
    void fill(hbtl::MutableArrayRef<T> array) {
      std::generate(array.begin(), array.end(), [&]() { return operator()(); });
    }

    /// generate an array of uniform random data
    std::vector<T> generate(size_t size) {
      std::vector<T> array(size);
      fill(array);
      return array;
    }

    /// shuffle given array according to uniform random data
    void shuffle(hbtl::MutableArrayRef<T> array) {
      // std :: shuffle (In libc++ of LLVM12) calls std :: uniform_int_distribution, so stdlib dependent
      hbtl::shuffle(array.begin(), array.end(), rng);
    }

  private:
    std::mt19937_64 rng;
    T lower = 0;
    T upper = 0;
  };

  /**
   * @brief Fill uniform random data into given array.
   *
   * @tparam T
   * @param array
   * @param salt salt appending to random number generator
   * @param lower values should below lower
   * @param upper values should under upper
   */
  template <typename T>
  void fillUniform(hbtl::MutableArrayRef<T> array, hbtl::string_view salt, T lower = std::numeric_limits<T>::lowest(),
                   T upper = std::numeric_limits<T>::max()) {
    UniformRNG<T> rng(salt, lower, upper);
    std::generate(array.begin(), array.end(), [&]() { return rng(); });
  }

  /**
   * @brief Generate uniform random data of given size.
   *
   * @tparam T
   * @param size
   * @param salt salt appending to random number generator
   * @param lower values should below lower
   * @param upper values should under upper
   * @return std::vector<T>
   */
  template <typename T>
  std::vector<T> genUniform(size_t size, hbtl::string_view salt, T lower = std::numeric_limits<T>::lowest(),
                            T upper = std::numeric_limits<T>::max()) {
    std::vector<T> array(size);
    fillUniform<T>(array, salt, lower, upper);
    return array;
  }

  template <typename T> class NormalRNG {

  public:
    explicit NormalRNG(double mean, double std, hbtl::string_view salt, T lower = std::numeric_limits<T>::lowest(),
                       T upper = std::numeric_limits<T>::max())
        : mean(mean), std(std), lower(lower), upper(upper) {
      std::vector<uint32_t> data(salt.size(), 0);
      copy(salt, data.begin());
      std::seed_seq seed(data.begin(), data.end());
      rng.seed(seed);
    }

    /// get one normalize data.
    T operator()() {
      // The values of std::xxx_distribution() is stdlib dependent
      // Use hbdk version instead
      hbtl::normal_distribution<double> dist(mean, std);
      return clamp(static_cast<T>(dist(rng)), lower, upper);
    }

    /// fill normal distributed data into an array
    void fill(hbtl::MutableArrayRef<T> array) {
      std::generate(array.begin(), array.end(), [&]() { return operator()(); });
    }

    /// generate an array of normal distributed data
    std::vector<T> generate(size_t size) {
      std::vector<T> array(size);
      fill(array);
      return array;
    }

  private:
    std::mt19937_64 rng;
    double mean = 0;
    double std = 0;
    T lower = 0;
    T upper = 0;
  };

  /**
   * @brief Fill normal random data into given array.
   *
   * @tparam T
   * @param array
   * @param mean average value
   * @param std standard deviation
   * @param salt salt appending to random number generator
   * @param lower values should below lower
   * @param upper values should under upper
   */
  template <typename T>
  void fillNormal(hbtl::MutableArrayRef<T> array, T mean, T std, hbtl::string_view salt = "hbdk",
                  T lower = std::numeric_limits<T>::lowest(), T upper = std::numeric_limits<T>::max()) {
    NormalRNG<T> rng(mean, std, salt, lower, upper);
    std::generate(array.begin(), array.end(), [&]() { return rng(); });
  }

  /**
   * @brief  Generate normal random data of given size.
   *
   * @tparam T
   * @param size
   * @param mean average value
   * @param std standard deviation
   * @param salt salt appending to random number generator
   * @param lower values should below lower
   * @param upper values should under upper
   * @return std::vector<T>
   */
  template <typename T>
  std::vector<T> genNormal(size_t size, T mean, T std, hbtl::string_view salt = "hbdk",
                           T lower = std::numeric_limits<T>::lowest(), T upper = std::numeric_limits<T>::max()) {
    std::vector<T> array(size);
    fillNormal<T>(array, mean, std, salt, lower, upper);
    return array;
  }
}
HBTL_NAMESPACE_END
