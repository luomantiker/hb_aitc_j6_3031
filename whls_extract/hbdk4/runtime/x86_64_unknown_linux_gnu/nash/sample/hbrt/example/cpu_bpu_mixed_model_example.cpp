#include "cpu_bpu_mixed_model.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>

// Examples:
// HBRT4PublicExample DDR T0000_conv_k3_s1_p0_1x23x37x17.hbm T0000_conv_k3_s1_p0_1x23x37x17 input.bin
// HBRT4PublicExample Pyramid load_pyramid_2.hbm load_pyramid_2 input.bin
// HBRT4PublicExample Resizer roiResizeNV12.hbm roiResizeNV12 input.bin
int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << R"(Please enter: HBRT4PublicExample DDR/Pyramid/Resizer hbm_name graph_name input_data_name)"
              << std::endl;
    return -1;
  }

  // print user command parameters
  std::cout << "example_category: " << argv[1] << std::endl;
  std::cout << "hbm_name: " << argv[2] << std::endl;
  std::cout << "graph_name: " << argv[3] << std::endl;
  std::cout << "input_data_name: " << argv[4] << std::endl;

  // load input_data from `input_data_name`
  std::ifstream input_file(argv[4], std::ios::binary | std::ios::in);
  if (!input_file.is_open()) {
    std::cout << "Cannot open " << argv[4] << std::endl;
    return -2;
  }

  // read file length
  input_file.seekg(0, std::ios::end);
  size_t input_len = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  std::cout << "input length: " << input_len << std::endl;

  if (strcmp(argv[1], "DDR") == 0) {
    // load file content
    char *input_buffer = new char[input_len];
    input_file.read(input_buffer, static_cast<long>(input_len));
    input_file.close();

    std::cout << "DDR example running" << std::endl;
    cpu_bpu_mixed_model_normal_example(argv[2], argv[3], input_buffer, "");
    std::cout << "DDR example success" << std::endl;
    delete[] input_buffer;
  } else if (strcmp(argv[1], "Pyramid") == 0) {
    // load file content
    char *input_buffer = new char[input_len];
    input_file.read(input_buffer, static_cast<long>(input_len));
    input_file.close();

    std::cout << "Pyramid example running" << std::endl;
    cpu_bpu_mixed_model_pyramid_example(argv[2], argv[3], input_buffer, 1280, "");
    std::cout << "Pyramid example success" << std::endl;
    delete[] input_buffer;
  } else if (strcmp(argv[1], "Resizer") == 0) {
    // load file content
    char *input_buffer = new char[input_len + 16];
    input_file.read(input_buffer, static_cast<long>(input_len));
    input_file.close();

    // set ROI
    std::array<uint32_t, 4> roi = {0, 0, 146, 95};
    memcpy(input_buffer + input_len, roi.data(), 16);

    std::vector<std::vector<size_t>> dims({{1, 360, 640, 1}, {1, 180, 320, 2}, {1, 4}});
    std::vector<std::vector<size_t>> strides({{360 * 640, 640, 1, 1}, {180 * 320 * 2, 320, 2, 1}, {16, 4}});

    std::cout << "Resizer example running" << std::endl;
    cpu_bpu_mixed_model_resizer_example(argv[2], argv[3], input_buffer, "./", dims, strides);
    std::cout << "Resizer example success" << std::endl;
    delete[] input_buffer;
  } else {
    std::cout << "Please select correct example category(DDR/Pyramid/Resizer)." << std::endl;
    return -3;
  }

  return 0;
}
