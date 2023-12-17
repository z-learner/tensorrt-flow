#pragma once
#include "NvInfer.h"

#include <any>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_flow {

namespace model {

class ModelPlugin {
public:
  ModelPlugin()          = default;
  virtual ~ModelPlugin() = default;

  virtual bool     PreProcess(const std::any& input_raw_data, void** input_data, size_t input_size, std::vector<nvinfer1::Dims>& inputs_dims, cudaStream_t& stream) = 0;
  virtual std::any PostProcess(void** engine_output_data, size_t output_size, std::vector<nvinfer1::Dims>& outputs_dims, cudaStream_t& stream)                      = 0;
};

}   // namespace model
}   // namespace tensorrt_flow