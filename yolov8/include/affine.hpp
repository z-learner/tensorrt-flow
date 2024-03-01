#pragma once

#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <iostream>
namespace yolov8 {

namespace affine {


struct TransInfo {
  int src_w;
  int src_h;
  int tar_w;
  int tar_h;
  TransInfo(int srcW, int srcH, int tarW, int tarH)
    : src_w(srcW)
    , src_h(srcH)
    , tar_w(tarW)
    , tar_h(tarH) {}
};

struct AffineMatrix {
  float forward[6];
  float reverse[6];
  float forward_scale;
  float reverse_scale;

  void calc_forward_matrix(TransInfo trans) {
    forward[0] = forward_scale;
    forward[1] = 0;
    forward[2] = -forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
    forward[3] = 0;
    forward[4] = forward_scale;
    forward[5] = -forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
  };

  void calc_reverse_matrix(TransInfo trans) {
    reverse[0] = reverse_scale;
    reverse[1] = 0;
    reverse[2] = -reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
    reverse[3] = 0;
    reverse[4] = reverse_scale;
    reverse[5] = -reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
  };

  void init(TransInfo trans) {
    float scaled_w = (float)trans.tar_w / trans.src_w;
    float scaled_h = (float)trans.tar_h / trans.src_h;
    forward_scale  = (scaled_w < scaled_h ? scaled_w : scaled_h);
    reverse_scale  = 1 / forward_scale;

    // 计算src->tar和tar->src的仿射矩阵
    calc_forward_matrix(trans);
    calc_reverse_matrix(trans);
  }
};

__host__ __device__ void affine_transformation(float trans_matrix[6], int src_x, int src_y, float& tar_x, float& tar_y);
void                     launch_resize_affine(float* target, uint8_t* source, TransInfo trans_info, AffineMatrix affine, cudaStream_t* stream_ptr = nullptr);

}   // namespace affine

}   // namespace yolov8