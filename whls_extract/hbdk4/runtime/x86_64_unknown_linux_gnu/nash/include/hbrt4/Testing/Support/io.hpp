#pragma once

#include <fstream>
#include <iostream>
#include <vector>

namespace hbrt4 {
namespace testing {

static inline std::vector<char> readFileIntoVector(const std::string &filename) {
  std::vector<char> data;
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << " " << std::strerror(errno) << std::endl;
    return {}; // return an empty vector if the file cannot be opened
  }

  // Get the size of the file
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read the file into the vector
  data.resize(fileSize);
  file.read(data.data(), fileSize);
  if (!file.good()) {
    std::cerr << "Error reading file: " << filename << " " << std::strerror(errno) << std::endl;
    return {}; // return an empty vector if the file cannot be read
  }
  file.close();
  return data;
}

} // namespace testing
} // namespace hbrt4
