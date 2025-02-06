#pragma once

#include <iostream>
#include <string>

namespace hbrt4 {
namespace testing {

static inline std::string getUnittestsDir() {
  const auto env = std::getenv("HBRT4_PUBLIC_UNITTESTS_DIR");
  if (env == nullptr) {
    std::cerr << "Fail to get unittests dir. Should set it in env variable HBRT4_PUBLIC_UNITTESTS_DIR\n";
    abort();
  }
  return {env};
}

static inline std::string getFilenameInUnittestsDir(std::string filename) { return getUnittestsDir() + "/" + filename; }

} // namespace testing
} // namespace hbrt4
