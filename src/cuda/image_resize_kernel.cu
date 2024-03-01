#include "tensorrt_flow/cuda/image_resize_kernel.hpp"
#include "tensorrt_flow/image_process/resize_tactics.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"

#include <cuda_runtime.h>

namespace tensorrt_flow {


namespace cuda {


__global__ void nearest_bgr2rgb_nhwc2nchw_norm_kernal(float* target, uint8_t* source, int target_width, int target_height, int source_width, int source_height, float scale_width, float scale_height,
                                                      float mean_channel0, float mean_channel1, float mean_channel2, float std_channel0, float std_channel1, float std_channel2) {

  // target index
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nearest neighbour
  int src_y = floor(y * scale_height);
  int src_x = floor(x * scale_width);

  // Do not compute for out-of-bounds data
  if (src_x < 0 || src_x >= source_width || src_y < 0 || src_y >= source_height) { return; }

  int tar_idx  = y * target_width + x;
  int tar_area = target_height * target_width;
  int src_idx  = (src_y * source_width + src_x) * 3;

  // resize and hwc2chw and bgr2rgb
  target[tar_idx]                = (source[src_idx + 2] / 255.0f - mean_channel2) / std_channel2;
  target[tar_idx + tar_area * 1] = (source[src_idx + 1] / 255.0f - mean_channel1) / std_channel1;
  target[tar_idx + tar_area * 2] = (source[src_idx] / 255.0f - mean_channel0) / std_channel0;
  return;
}


__global__ void bilinear_bgr2rgb_nhwc2nchw_norm_kernal(float* target, uint8_t* source, int target_width, int target_height, int source_width, int source_height, float scale_width, float scale_height,
                                                       float mean_channel0, float mean_channel1, float mean_channel2, float std_channel0, float std_channel1, float std_channel2) {

  // target index
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nearest neighbour
  int src_y1 = floor((y + 0.5) * scale_height - 0.5);
  int src_x1 = floor((x + 0.5) * scale_width - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_x1 < 0 || src_x2 >= source_width || src_y1 < 0 || src_y2 >= source_height) { return; }

  // index in source image to 0 ~ 1
  float th = ((y + 0.5) * scale_height - 0.5) - src_y1;
  float tw = ((x + 0.5) * scale_width - 0.5) - src_x1;

  // calculate area
  float a1_1 = (1.0f - tw) * (1.0f - th);   // right and down
  float a1_2 = tw * (1.0f - th);            // left and down
  float a2_1 = (1.0 - tw) * th;             // right and up
  float a2_2 = tw * th;                     // left and up

  // calculate index for four coordinates
  int src_idx1_1 = (src_y1 * source_width + src_x1) * 3;   // left and up
  int src_idx1_2 = (src_y1 * source_width + src_x2) * 3;   // right and up
  int src_idx2_1 = (src_y2 * source_width + src_x1) * 3;   // left and down
  int src_idx2_2 = (src_y2 * source_width + src_x2) * 3;   // left and up

  // target index
  int tar_idx  = y * target_width + x;
  int tar_area = target_width * target_height;

  // resize, bgr2rgb, hwc2chw
  target[tar_idx] = (round(a1_1 * source[src_idx1_1 + 2] + a1_2 * source[src_idx1_2 + 2] + a2_1 * source[src_idx2_1 + 2] + a2_2 * source[src_idx2_2 + 2]) / 255.0f - mean_channel2) / std_channel2;

  target[tar_idx + tar_area] =
    (round(a1_1 * source[src_idx1_1 + 1] + a1_2 * source[src_idx1_2 + 1] + a2_1 * source[src_idx2_1 + 1] + a2_2 * source[src_idx2_2 + 1]) / 255.0f - mean_channel1) / std_channel1;

  target[tar_idx + tar_area * 2] = (round(a1_1 * source[src_idx1_1] + a1_2 * source[src_idx1_2] + a2_1 * source[src_idx2_1] + a2_2 * source[src_idx2_2]) / 255.0f - mean_channel0) / std_channel0;

  return;
}


__global__ void bilinear_bgr2rgb_nhwc2nchw_shift_norm_kernel(float* target, uint8_t* source, int target_width, int target_height, int source_width, int source_height, float scale_width,
                                                             float scale_height, float mean_channel0, float mean_channel1, float mean_channel2, float std_channel0, float std_channel1,
                                                             float std_channel2) {

  // target index
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nearest neighbour
  int src_y1 = floor((y + 0.5) * scale_height - 0.5);
  int src_x1 = floor((x + 0.5) * scale_width - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_x1 < 0 || src_x2 >= source_width || src_y1 < 0 || src_y2 >= source_height) { return; }

  // index in source image to 0 ~ 1
  float th = (float)y * scale_height - src_y1;
  float tw = (float)x * scale_width - src_x1;

  // calculate area
  float a1_1 = (1.0f - tw) * (1.0f - th);   // right and down
  float a1_2 = tw * (1.0f - th);            // left and down
  float a2_1 = (1.0 - tw) * th;             // right and up
  float a2_2 = tw * th;                     // left and up

  // calculate index for four coordinates
  int src_idx1_1 = (src_y1 * source_width + src_x1) * 3;   // left and up
  int src_idx1_2 = (src_y1 * source_width + src_x2) * 3;   // right and up
  int src_idx2_1 = (src_y2 * source_width + src_x1) * 3;   // left and down
  int src_idx2_2 = (src_y2 * source_width + src_x2) * 3;   // left and up

  // shift
  y = y - int(source_height / (scale_height * 2)) + int(target_height / 2);
  x = x - int(source_width / (source_width * 2)) + int(target_width / 2);

  // target index
  int tar_idx  = y * target_width + x;
  int tar_area = target_width * target_height;

  // resize, bgr2rgb, hwc2chw
  target[tar_idx] = (round(a1_1 * source[src_idx1_1 + 2] + a1_2 * source[src_idx1_2 + 2] + a2_1 * source[src_idx2_1 + 2] + a2_2 * source[src_idx2_2 + 2]) / 255.0f - mean_channel2) / std_channel2;
  target[tar_idx + tar_area] =
    (round(a1_1 * source[src_idx1_1 + 1] + a1_2 * source[src_idx1_2 + 1] + a2_1 * source[src_idx2_1 + 1] + a2_2 * source[src_idx2_2 + 1]) / 255.0f - mean_channel1) / std_channel1;
  target[tar_idx + tar_area * 2] = (round(a1_1 * source[src_idx1_1] + a1_2 * source[src_idx1_2] + a2_1 * source[src_idx2_1] + a2_2 * source[src_idx2_2]) / 255.0f - mean_channel0) / std_channel0;

  return;
}



void resize_bgr2rgb_nhwc2nchw_gpu(float* target, uint8_t* source, int target_width, int target_height, int source_width, int source_height, float mean_channel0, float mean_channel1,
                                  float mean_channel2, float std_channel0, float std_channel1, float std_channel2, image_process::ResizeTactics tactics, cudaStream_t* stream_ptr) {
  dim3 dim_block(32, 32, 1);
  dim3 dim_grid(target_width / 32 + 1, target_height / 32 + 1, 1);

  // LOG_INFO(" dim_grid  : (%d, %d) ", dim_grid.x, dim_grid.y);
  // LOG_INFO(" dim_block : (%d, %d) ", dim_block.x, dim_block.y);

  // scaled resize
  float scaled_height = static_cast<float>(source_height) / target_height;
  float scaled_width  = static_cast<float>(source_width) / target_width;
  // LOG_INFO("source_height : %d", source_height);
  // LOG_INFO("source_width  : %d", source_width);
  // LOG_INFO("target_height  : %d", target_height);
  // LOG_INFO("target_width  : %d", target_width);
  // LOG_INFO("scaled_height : %f", scaled_height);
  // LOG_INFO("scaled_width  : %f", scaled_width);

  float scale = scaled_height > scaled_width ? scaled_height : scaled_width;
  // LOG_INFO("scale  : %f", scale);


  switch (tactics) {
  case image_process::ResizeTactics::GPU_NEAREST:
    if (stream_ptr != nullptr) {
      nearest_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block, 0, *stream_ptr>>>(target, source, target_width, target_height, source_width, source_height, scaled_width, scaled_height,
                                                                                     mean_channel0, mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    } else {
      nearest_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block>>>(target, source, target_width, target_height, source_width, source_height, scaled_width, scaled_height, mean_channel0,
                                                                     mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    }
    break;
  case image_process::ResizeTactics::GPU_NEAREST_CENTER:
    if (stream_ptr != nullptr) {
      nearest_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block, 0, *stream_ptr>>>(target, source, target_width, target_height, source_width, source_height, scale, scale, mean_channel0,
                                                                                     mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    } else {
      nearest_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block>>>(target, source, target_width, target_height, source_width, source_height, scale, scale, mean_channel0, mean_channel1,
                                                                     mean_channel2, std_channel0, std_channel1, std_channel2);
    }
    break;
  case image_process::ResizeTactics::GPU_BILINEAR:
    if (stream_ptr != nullptr) {
      bilinear_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block, 0, *stream_ptr>>>(target, source, target_width, target_height, source_width, source_height, scaled_width, scaled_height,
                                                                                      mean_channel0, mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    } else {
      bilinear_bgr2rgb_nhwc2nchw_norm_kernal<<<dim_grid, dim_block>>>(target, source, target_width, target_height, source_width, source_height, scaled_width, scaled_height, mean_channel0,
                                                                      mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    }
    break;
  case image_process::ResizeTactics::GPU_BILINEAR_CENTER:
    if (stream_ptr != nullptr) {
      bilinear_bgr2rgb_nhwc2nchw_shift_norm_kernel<<<dim_grid, dim_block, 0, *stream_ptr>>>(target, source, target_width, target_height, source_width, source_height, scale, scale, mean_channel0,
                                                                                            mean_channel1, mean_channel2, std_channel0, std_channel1, std_channel2);
    } else {
      bilinear_bgr2rgb_nhwc2nchw_shift_norm_kernel<<<dim_grid, dim_block>>>(target, source, target_width, target_height, source_width, source_height, scale, scale, mean_channel0, mean_channel1,
                                                                            mean_channel2, std_channel0, std_channel1, std_channel2);
    }
    break;
  default: LOG_ERROR("ERROR: Wrong GPU resize tactics selected. Program terminated"); exit(1);
  }
  return;
}


}   // namespace cuda




}   // namespace tensorrt_flow