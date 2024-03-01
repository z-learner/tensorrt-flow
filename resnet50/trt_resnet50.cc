#include "trt_resnet50.hpp"

#include "fmt/format.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "tensorrt_flow/cuda/cuda_helper.hpp"
#include "tensorrt_flow/cuda/image_resize_kernel.hpp"
#include "tensorrt_flow/image_process/image_resize.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_utils.hpp"

#include <assert.h>
#include <thread>
namespace tensorrt_flow {

namespace model {

namespace resnet50 {

Resnet50Plugin::Resnet50Plugin(const Resnet50Parameter& parameter)
  : parameter_(parameter) {
  image_memory_size_ = parameter.img_info.c * parameter.img_info.w * parameter.img_info.h * sizeof(float);
  LOG_DEBUG("Resnet50Plugin image_memory_size : %d", image_memory_size_);
  CUDA_CHECK(cudaMallocHost((void**)&input_data_, image_memory_size_));
  CUDA_CHECK(cudaMallocHost((void**)&output_data_, 1000 * sizeof(float)));
}


Resnet50Plugin::~Resnet50Plugin() {
  if (input_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(input_data_)); }
  if (output_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(output_data_)); }
  if (origin_image_memory_device_ != nullptr) { CUDA_CHECK(cudaFree(origin_image_memory_device_)); }
}


bool Resnet50Plugin::PreProcess(const std::any& input_raw_data, void** input_data, size_t input_size, std::vector<nvinfer1::Dims>& inputs_dims, cudaStream_t& stream) {
  assert(input_size == 1);

  const std::string& input_image_file = *(std::any_cast<std::string>(&input_raw_data));
  cv::Mat            input_image      = cv::imread(input_image_file);

  if (input_image.data == nullptr) { std::__throw_runtime_error(fmt::format("can't load {}", input_image_file).c_str()); }

  // LOG_INFO("input image origin size (%d, %d)", input_image.rows, input_image.cols);

  // use cpu to pre process
#if 0
  cv::resize(input_image, input_image, cv::Size(parameter_.img_info.w, parameter_.img_info.h), 0, 0, cv::INTER_LINEAR);
  // normalization and BGR2RGB
  int index;
  int offset_ch0 = parameter_.img_info.w * parameter_.img_info.h * 0;
  int offset_ch1 = parameter_.img_info.w * parameter_.img_info.h * 1;
  int offset_ch2 = parameter_.img_info.w * parameter_.img_info.h * 2;
  for (int i = 0; i < inputs_dims[0].d[2]; i++) {
    for (int j = 0; j < inputs_dims[0].d[3]; j++) {
      index                     = i * inputs_dims[0].d[3] * inputs_dims[0].d[1] + j * inputs_dims[0].d[1];
      input_data_[offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean_[0]) / std_[0];
      input_data_[offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean_[1]) / std_[1];
      input_data_[offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean_[2]) / std_[2];
    }
  }

#  if 0

  auto image_path = std::filesystem::path(input_image_file);

  auto new_file = image_path.parent_path().string() + "/resize/cpu_resize_" + image_path.filename().string();
  cv::Mat save_mat(parameter_.img_info.h, parameter_.img_info.w, CV_8UC3);

  // back to HWC
  offset_ch0 = parameter_.img_info.w * parameter_.img_info.h * 0;
  offset_ch1 = parameter_.img_info.w * parameter_.img_info.h * 1;
  offset_ch2 = parameter_.img_info.w * parameter_.img_info.h * 2;
  for (int i = 0; i < parameter_.img_info.h; ++i) {
    for (int j = 0; j < parameter_.img_info.w; ++j) {
      index                    = i * parameter_.img_info.w * parameter_.img_info.c + j * parameter_.img_info.c;
      save_mat.data[index + 0] = (input_data_[offset_ch2++] * std_[0] + mean_[0]) * 255.0f;
      save_mat.data[index + 1] = (input_data_[offset_ch1++] * std_[1] + mean_[1]) * 255.0f;
      save_mat.data[index + 2] = (input_data_[offset_ch0++] * std_[2] + mean_[2]) * 255.0f;
    }
  }

  cv::imwrite(new_file, save_mat);
#  endif

  CUDA_CHECK(cudaMemcpyAsync(input_data[0], input_data_, image_memory_size_, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));

#else
  // use gpu
  size_t origin_image_size = input_image.rows * input_image.cols * input_image.channels();
  if (origin_image_memory_size_ < origin_image_size) {
    // reallocate memeory
    if (origin_image_memory_device_ != nullptr) {
      CUDA_CHECK(cudaFree(origin_image_memory_device_));
      origin_image_memory_device_ = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&origin_image_memory_device_, origin_image_size))
    origin_image_memory_size_ = origin_image_size;
  }

  CUDA_CHECK(cudaMemcpyAsync(origin_image_memory_device_, input_image.ptr(), origin_image_size, cudaMemcpyHostToDevice, stream));
  image_process::ResizeTactics tac = image_process::ResizeTactics::GPU_BILINEAR;

  ::tensorrt_flow::cuda::resize_bgr2rgb_nhwc2nchw_gpu((float*)input_data[0], (uint8_t*)origin_image_memory_device_, parameter_.img_info.w, parameter_.img_info.h, input_image.cols, input_image.rows,
                                                      mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2], tac, &stream);

#  if 0
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> input_data_tmp(parameter_.img_info.w * parameter_.img_info.h * parameter_.img_info.c);

  CUDA_CHECK(cudaMemcpy(input_data_tmp.data(), input_data[0], input_data_tmp.size() * sizeof(float), cudaMemcpyDeviceToHost))

  auto image_path = std::filesystem::path(input_image_file);

  auto new_file = image_path.parent_path().string() + "/resize/gpu_resize_" + image_path.filename().string();

  cv::Mat save_mat(parameter_.img_info.h, parameter_.img_info.w, CV_8UC3);

  // back to HWC
  int offset_ch0 = parameter_.img_info.w * parameter_.img_info.h * 0;
  int offset_ch1 = parameter_.img_info.w * parameter_.img_info.h * 1;
  int offset_ch2 = parameter_.img_info.w * parameter_.img_info.h * 2;
  for (int i = 0; i < parameter_.img_info.h; ++i) {
    for (int j = 0; j < parameter_.img_info.w; ++j) {
      int index                = i * parameter_.img_info.w * parameter_.img_info.c + j * parameter_.img_info.c;
      save_mat.data[index + 0] = (input_data_tmp[offset_ch2++] * std_[0] + mean_[0]) * 255.0f;
      save_mat.data[index + 1] = (input_data_tmp[offset_ch1++] * std_[1] + mean_[1]) * 255.0f;
      save_mat.data[index + 2] = (input_data_tmp[offset_ch0++] * std_[2] + mean_[2]) * 255.0f;
    }
  }

  cv::imwrite(new_file, save_mat);
#  endif

#endif

  return true;
}

std::any Resnet50Plugin::PostProcess(void** engine_output_data, size_t output_size, std::vector<nvinfer1::Dims>& outputs_dims, cudaStream_t& stream) {
  assert(output_size == 1);
  CUDA_CHECK(cudaMemcpyAsync((void*)output_data_, engine_output_data[0], 1000 * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int pos = std::max_element(output_data_, output_data_ + 1000) - output_data_;

  assert(pos < 1000);
  std::any return_type = resnet50_label[pos];
  return std::move(return_type);
}
}   // namespace resnet50

}   // namespace model
}   // namespace tensorrt_flow