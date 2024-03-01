#pragma once

#include "opencv2/opencv.hpp"
#include "tensorrt_flow/image_process/resize_tactics.hpp"

#include <array>
#include <cstdint>
#include <cuda_runtime.h>

namespace tensorrt_flow {

namespace image_process {



cv::Mat resize_bgr2rgb_cpu(cv::Mat& src, int target_width, int target_height, std::array<float, 3>& mean, std::array<float, 3>& std, ResizeTactics tactics);


}   // namespace image_process
}   // namespace tensorrt_flow