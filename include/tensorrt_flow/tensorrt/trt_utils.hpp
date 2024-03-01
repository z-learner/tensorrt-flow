#pragma once

#include "tensorrt_flow/def/bmp_def.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>


#define DEFINE_HAS_MEMBER_FUNCTION(FunctionName)                                       \
  template<typename T, typename = void> struct Has##FunctionName : std::false_type {}; \
  template<typename T> struct Has##FunctionName<T, std::void_t<decltype(&T::FunctionName)>> : std::true_type {}


namespace tensorrt_flow {


namespace utils {



class TimeStatisic {
public:
  TimeStatisic()  = default;
  ~TimeStatisic() = default;

  void set_value(float value);
  void reset();

  size_t count() { return count_; }
  float  min() { return min_; }
  float  max() { return max_; }
  float  avg() { return avg_; }

private:
  size_t count_{0};
  float  min_{std::numeric_limits<float>::max()};
  float  max_{std::numeric_limits<float>::min()};
  float  avg_{0};
};


static bool SaveWHCBmpFile(const std::string& file_name, uint8_t* data, size_t size, int width, int height) {
  bmp_def::BMPFileHeader fileHeader;
  bmp_def::BMPInfoHeader infoHeader;

  fileHeader.type      = 0x4D42;   // "BM"
  fileHeader.size      = sizeof(bmp_def::BMPFileHeader) + sizeof(bmp_def::BMPInfoHeader) + size;
  fileHeader.reserved1 = 0;
  fileHeader.reserved2 = 0;
  fileHeader.offset    = sizeof(bmp_def::BMPFileHeader) + sizeof(bmp_def::BMPInfoHeader);

  infoHeader.size            = sizeof(bmp_def::BMPInfoHeader);
  infoHeader.width           = width;
  infoHeader.height          = height;
  infoHeader.planes          = 1;
  infoHeader.bitCount        = 24;   // 24位色深
  infoHeader.compression     = 0;
  infoHeader.imageSize       = size;
  infoHeader.xPixelsPerMeter = 0;
  infoHeader.yPixelsPerMeter = 0;
  infoHeader.colorsUsed      = 0;
  infoHeader.colorsImportant = 0;

  std::vector<uint8_t> data_tmp(size);

  size_t channel_offset = width * height;

  for (size_t index = 0; index < channel_offset; ++index) {
    data_tmp[index]     = static_cast<uint8_t>(data[index]);
    data_tmp[index + 1] = static_cast<uint8_t>(data[index + channel_offset]);
    data_tmp[index + 2] = static_cast<uint8_t>(data[index + channel_offset * 2]);
  }


  std::ofstream file(file_name, std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(bmp_def::BMPFileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(bmp_def::BMPInfoHeader));
    file.write(reinterpret_cast<const char*>(data_tmp.data()), data_tmp.size());
    file.close();
  } else {
    LOG_ERROR("Error: Unable to create BMP file in %s", file_name.c_str());
    return false;
  }
  return true;
}
}   // namespace utils


}   // namespace tensorrt_flow