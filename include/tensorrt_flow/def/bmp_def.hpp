
#pragma once
#include <iostream>



namespace tensorrt_flow {

namespace bmp_def {

#pragma pack(push, 1)   // 确保结构体按字节对齐
struct BMPFileHeader {
  uint16_t type;        // 文件类型，"BM"（0x42, 0x4D）
  uint32_t size;        // 文件大小
  uint16_t reserved1;   // 保留字段
  uint16_t reserved2;   // 保留字段
  uint32_t offset;      // 数据偏移量
};

struct BMPInfoHeader {
  uint32_t size;              // 信息头大小
  int32_t  width;             // 图像宽度
  int32_t  height;            // 图像高度
  uint16_t planes;            // 色彩平面数
  uint16_t bitCount;          // 位深度
  uint32_t compression;       // 压缩方式
  uint32_t imageSize;         // 图像大小
  int32_t  xPixelsPerMeter;   // 水平分辨率
  int32_t  yPixelsPerMeter;   // 垂直分辨率
  uint32_t colorsUsed;        // 使用的颜色数
  uint32_t colorsImportant;   // 重要的颜色数
};
#pragma pack(pop)

}   // namespace bmp_def
}   // namespace tensorrt_flow