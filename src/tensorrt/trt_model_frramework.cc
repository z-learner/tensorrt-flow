#include "tensorrt_flow/tensorrt/trt_model_frramework.hpp"

#include "NvOnnxParser.h"
#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <assert.h>
#include <exception>
#include <fmt/format.h>
namespace tensorrt_flow {

namespace model {


static size_t DimsSize(nvinfer1::Dims& dims) {
  size_t size = 1;
  for (int index = 0; index < dims.nbDims; ++index) { size *= dims.d[index]; }
  return size;
}

static size_t GetDataTypeBytes(nvinfer1::DataType data_type) {
  switch (data_type) {
  case nvinfer1::DataType::kFLOAT: return 4; break;
  case nvinfer1::DataType::kHALF: return 2;
  case nvinfer1::DataType::kINT8: return 1;
  case nvinfer1::DataType::kINT32: return 4;
  case nvinfer1::DataType::kBOOL: return 1;
  case nvinfer1::DataType::kUINT8: return 1;
  default: std::__throw_runtime_error(fmt::format("don't support nvinfer1::DataType :{}", static_cast<int>(data_type)).c_str());
  }
  return 0;
}


ModelFramework::ModelFramework(const ModelFrameworkParameter& parameter)
  : model_framework_parameter_(parameter) {
  logger_  = std::make_shared<logger::Logger>(parameter.logger_level);
  build_   = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger_), destroy_trt_ptr<nvinfer1::IBuilder>);
  network_ = std::shared_ptr<nvinfer1::INetworkDefinition>(build_->createNetworkV2(parameter.network_create_flags), destroy_trt_ptr<nvinfer1::INetworkDefinition>);

  auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(build_->createBuilderConfig(), destroy_trt_ptr<nvinfer1::IBuilderConfig>);
  auto parser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, *logger_), destroy_trt_ptr<nvonnxparser::IParser>);

  config->setMaxWorkspaceSize(parameter.max_engine_workspace_size);
  config->setProfilingVerbosity(parameter.verbosity);

  if (!parser->parseFromFile(parameter.onnx_file_path.c_str(), parameter.build_engine_from_onnx_log_level)) {
    std::__throw_runtime_error(fmt::format("prase onnx file failed from {}", parameter.onnx_file_path).c_str());
  }

  if (parameter.prec == precision::Precision::FP16) {
    if (!build_->platformHasFastFp16()) { std::__throw_runtime_error(fmt::format("don't support FP16 in this compute").c_str()); }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(parameter.build_percisin_constraints);
  }

  engine_  = std::shared_ptr<nvinfer1::ICudaEngine>(build_->buildEngineWithConfig(*network_, *config), destroy_trt_ptr<nvinfer1::ICudaEngine>);
  runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_), destroy_trt_ptr<nvinfer1::IRuntime>);

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), destroy_trt_ptr<nvinfer1::IExecutionContext>);

  input_count_  = network_->getNbInputs();
  output_count_ = network_->getNbOutputs();

  assert(input_count_ > 0);
  assert(output_count_ > 0);

  binding_data_.resize(input_count_ + output_count_);
  inputs_dims_.resize(input_count_);
  for (int32_t index = 0; index < input_count_; ++index) {
    inputs_dims_[index] = engine_->getBindingDimensions(index);
    assert(inputs_dims_[index].nbDims > 0);
    nvinfer1::DataType data_type = engine_->getBindingDataType(index);
    CUDA_CHECK(cudaMalloc(&(binding_data_[index]), DimsSize(inputs_dims_[index]) * GetDataTypeBytes(data_type)));
  }

  outputs_dims_.resize(output_count_);
  for (int32_t index = 0; index < output_count_; ++index) {
    outputs_dims_[index] = engine_->getBindingDimensions(input_count_ + index);
    assert(outputs_dims_[index].nbDims > 0);
    nvinfer1::DataType data_type = engine_->getBindingDataType(input_count_ + index);
    CUDA_CHECK(cudaMalloc(&(binding_data_[input_count_ + index]), DimsSize(outputs_dims_[index]) * GetDataTypeBytes(data_type)));
  }
  CUDA_CHECK(cudaStreamCreate(&stream_));
}


ModelFramework::~ModelFramework() {
  for (auto& ptr : binding_data_) { CUDA_CHECK(cudaFree(ptr)); }
}



}   // namespace model
}   // namespace tensorrt_flow