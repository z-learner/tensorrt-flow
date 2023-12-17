#include "trt_resnet50.hpp"

#include "fmt/format.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <assert.h>
namespace tensorrt_flow {

namespace model {

namespace resnet50 {

Resnet50Plugin::Resnet50Plugin(const Resnet50Parameter& parameter)
  : parameter_(parameter) {
  image_memory_size_ = parameter.img_info.c * parameter.img_info.w * parameter.img_info.h * sizeof(float);
  std::cout << "Resnet50Plugin image_memory_size : " << image_memory_size_ << std::endl;
  CUDA_CHECK(cudaMallocHost((void**)&input_data_, image_memory_size_));
  CUDA_CHECK(cudaMallocHost((void**)&output_data_, 1000 * sizeof(float)));
}


Resnet50Plugin::~Resnet50Plugin() {
  if (input_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(input_data_)); }
  if (output_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(output_data_)); }
}


bool Resnet50Plugin::PreProcess(const std::any& input_raw_data, void** input_data, size_t input_size, std::vector<nvinfer1::Dims>& inputs_dims, cudaStream_t& stream) {
  assert(input_size == 1);

  const std::string& input_image_file = *(std::any_cast<std::string>(&input_raw_data));
  cv::Mat            input_image      = cv::imread(input_image_file);

  if (input_image.data == nullptr) { std::__throw_runtime_error(fmt::format("can't load {}", input_image_file).c_str()); }

  cv::resize(input_image, input_image, cv::Size(parameter_.img_info.w, parameter_.img_info.h), 0, 0, cv::INTER_LINEAR);

  // normalization and BGR2RGB
  int index;
  int offset_ch0 = parameter_.img_info.w * 0;
  int offset_ch1 = parameter_.img_info.h * 1;
  int offset_ch2 = parameter_.img_info.h * 2;
  for (int i = 0; i < inputs_dims[0].d[2]; i++) {
    for (int j = 0; j < inputs_dims[0].d[3]; j++) {
      index                     = i * inputs_dims[0].d[3] * inputs_dims[0].d[1] + j * inputs_dims[0].d[1];
      input_data_[offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean_[0]) / std_[0];
      input_data_[offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean_[1]) / std_[1];
      input_data_[offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean_[2]) / std_[2];
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(input_data[0], input_data_, image_memory_size_, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
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