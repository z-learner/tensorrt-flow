#include "affine.hpp"
#include "tensorrt_flow/cuda/cuda_helper.hpp"

namespace yolov8 {

namespace affine {


__host__ __device__ void affine_transformation(float trans_matrix[6], int src_x, int src_y, float& tar_x, float& tar_y) {
  tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
  tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}


__global__ void resize_warpaffine_bgr2rgb_kernel(float* target, uint8_t* source, TransInfo trans_info, AffineMatrix affine) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  float src_x = 0;
  float src_y = 0;

  affine_transformation(affine.reverse, x, y, src_x, src_y);

  // bilinear interpolation
  int src_x1 = floor(src_x - 0.5);
  int src_y1 = floor(src_y - 0.5);
  int src_x2 = src_x1 + 1;
  int src_y2 = src_y1 + 1;

  if (src_y1 < 0 || src_x1 < 0 || src_y2 >= trans_info.src_h || src_x2 >= trans_info.src_w) { return; }

  float tw = src_x - src_x1;
  float th = src_y - src_y1;

  float a1_1 = (1.0 - tw) * (1.0 - th);
  float a1_2 = tw * (1.0 - th);
  float a2_1 = (1.0 - tw) * th;
  float a2_2 = tw * th;

  int srcIdx1_1 = (src_y1 * trans_info.src_w + src_x1) * 3;
  int srcIdx1_2 = (src_y1 * trans_info.src_w + src_x2) * 3;
  int srcIdx2_1 = (src_y2 * trans_info.src_w + src_x1) * 3;
  int srcIdx2_2 = (src_y2 * trans_info.src_w + src_x2) * 3;

  // bilinear interpolation -- 计算resized之后的图的索引
  int tarIdx  = y * trans_info.tar_w + x;
  int tarArea = trans_info.tar_w * trans_info.tar_h;


  // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB
  target[tarIdx + tarArea * 0] = round((a1_1 * source[srcIdx1_1 + 2] + a1_2 * source[srcIdx1_2 + 2] + a2_1 * source[srcIdx2_1 + 2] + a2_2 * source[srcIdx2_2 + 2])) / 255.0f;

  target[tarIdx + tarArea * 1] = round((a1_1 * source[srcIdx1_1 + 1] + a1_2 * source[srcIdx1_2 + 1] + a2_1 * source[srcIdx2_1 + 1] + a2_2 * source[srcIdx2_2 + 1])) / 255.0f;

  target[tarIdx + tarArea * 2] = round((a1_1 * source[srcIdx1_1 + 0] + a1_2 * source[srcIdx1_2 + 0] + a2_1 * source[srcIdx2_1 + 0] + a2_2 * source[srcIdx2_2 + 0])) / 255.0f;
  return;
}


void launch_resize_affine(float* target, uint8_t* source, TransInfo trans_info, AffineMatrix affine, cudaStream_t* stream_ptr) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(trans_info.tar_w / 16 + 1, trans_info.tar_h / 16 + 1, 1);

  if (stream_ptr != nullptr) {
    resize_warpaffine_bgr2rgb_kernel<<<dimGrid, dimBlock, 0, *stream_ptr>>>(target, source, trans_info, affine);
  } else {
    resize_warpaffine_bgr2rgb_kernel<<<dimGrid, dimBlock>>>(target, source, trans_info, affine);
  }
  return;
}

}   // namespace affine
}   // namespace yolov8