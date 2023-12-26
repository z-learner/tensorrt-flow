#pragma once

#include "NvInfer.h"
#include "resnet50_def.hpp"
#include "tensorrt_flow/cuda/cuda_helper.hpp"
#include "tensorrt_flow/tensorrt/trt_calibrator.hpp"

using IInt8EntropyCalibrator2Base = tensorrt_flow::calibrator::TrtCalibrator<nvinfer1::IInt8EntropyCalibrator2>;

class Resnet50Calibrator : public IInt8EntropyCalibrator2Base {
public:
  Resnet50Calibrator(tensorrt_flow::calibrator::TrtCalibratorParamter parameter, tensorrt_flow::model::resnet50::Resnet50Parameter resnet_parameter)
    : IInt8EntropyCalibrator2Base(parameter)
    , resnet_parameter_(resnet_parameter) {
    CUDA_CHECK(cudaMalloc(&input_data_, parameter_.batch_size * resnet_parameter.img_info.w * resnet_parameter.img_info.h * resnet_parameter.img_info.c * sizeof(float)));
  }
  virtual ~Resnet50Calibrator() { CUDA_CHECK(cudaFree(input_data_)); }

  virtual std::vector<void*> process(const std::vector<std::string>& file_path_name_list);


private:
  void*                                             input_data_{nullptr};
  tensorrt_flow::model::resnet50::Resnet50Parameter resnet_parameter_;
  std::array<float, 3>                              mean_{0.406, 0.456, 0.485};
  std::array<float, 3>                              std_{0.225, 0.224, 0.229};
};