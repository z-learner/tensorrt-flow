#include "tensorrt_flow/image_process/image_resize.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"

namespace tensorrt_flow {

namespace image_process {

void resize_bgr2rgb_cpu(cv::Mat& src, int target_width, int target_height, const std::array<float, 3>& mean, const std::array<float, 3>& std, ResizeTactics tactics) {

  int source_height = src.rows;
  int source_width  = src.cols;

  float dim = std::max(source_height, source_width);   // int -> float

  // int y_offset = (target_width - (source_width / dim) * target_width) / 2;
  // int x_offset = (target_height - (source_height / dim) * target_height) / 2;

  // convert color
  cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
  // resize
  switch (tactics) {
  case ResizeTactics::CPU_NEAREST: cv::resize(src, src, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR); break;
  case ResizeTactics::CPU_BILINEAR: cv::resize(src, src, cv::Size(target_width, target_height), 0, 0, cv::INTER_NEAREST); break;
  default:
    LOG_ERROR("ERROR: Wrong CPU resize tactics selected. Program terminated");
    exit(1);
    break;
  }
  return;
}


}   // namespace image_process
}   // namespace tensorrt_flow