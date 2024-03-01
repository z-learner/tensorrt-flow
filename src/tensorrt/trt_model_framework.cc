#include "tensorrt_flow/tensorrt/trt_model_framework.hpp"

#include "NvOnnxParser.h"
#include "tensorrt_flow/cuda/cuda_helper.hpp"
#include "tensorrt_flow/tensorrt/trt_utils.hpp"

#include <assert.h>
#include <exception>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
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

  logger_ = std::make_shared<logger::Logger>(parameter.logger_level);

  if (parameter.is_use_engine_cache && std::filesystem::exists(parameter.engine_file_path)) {
    std::uintmax_t    file_size = std::filesystem::file_size(parameter.engine_file_path);
    std::vector<char> tmp_buff(file_size);
    std::ifstream     file(parameter.engine_file_path, std::ios::binary);
    file.read(tmp_buff.data(), file_size);
    file.close();

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_), destroy_trt_ptr<nvinfer1::IRuntime>);
    engine_  = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(tmp_buff.data(), file_size), destroy_trt_ptr<nvinfer1::ICudaEngine>);

  } else {
    build_      = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger_), destroy_trt_ptr<nvinfer1::IBuilder>);
    network_    = std::shared_ptr<nvinfer1::INetworkDefinition>(build_->createNetworkV2(parameter.network_create_flags), destroy_trt_ptr<nvinfer1::INetworkDefinition>);
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
    } else if (parameter.prec == precision::Precision::INT8) {
      if (!build_->platformHasFastInt8()) { std::__throw_runtime_error(fmt::format("don't support INT8 in this compute").c_str()); }
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      config->setFlag(parameter.build_percisin_constraints);
    }

    if (parameter.iint8_calibration != nullptr) {
      LOG_INFO("Use Calibration...");
      config->setInt8Calibrator(parameter.iint8_calibration.get());
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(build_->buildEngineWithConfig(*network_, *config), destroy_trt_ptr<nvinfer1::ICudaEngine>);

    if (parameter.is_use_engine_cache && !std::filesystem::exists(parameter.engine_file_path)) {
      std::filesystem::path fd_file_path = parameter.engine_file_path;
      if (!std::filesystem::exists(fd_file_path.parent_path())) { std::filesystem::create_directories(fd_file_path.parent_path()); }
      auto          plan = build_->buildSerializedNetwork(*network_, *config);
      std::ofstream output_file(parameter.engine_file_path);
      output_file.write(reinterpret_cast<char*>(plan->data()), plan->size());
      output_file.close();
    }
  }


  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), destroy_trt_ptr<nvinfer1::IExecutionContext>);

  assert(engine_->getNbBindings() == (parameter.engine_input_count + parameter.engine_output_count));

  assert(parameter.engine_input_count > 0);
  assert(parameter.engine_output_count > 0);

  binding_data_.resize(parameter.engine_input_count + parameter.engine_output_count);
  inputs_dims_.resize(parameter.engine_input_count);
  for (int32_t index = 0; index < parameter.engine_input_count; ++index) {
    inputs_dims_[index] = engine_->getBindingDimensions(index);
    assert(inputs_dims_[index].nbDims > 0);
    nvinfer1::DataType data_type = engine_->getBindingDataType(index);
    CUDA_CHECK(cudaMalloc(&(binding_data_[index]), DimsSize(inputs_dims_[index]) * GetDataTypeBytes(data_type)));
  }

  outputs_dims_.resize(parameter.engine_output_count);
  for (int32_t index = 0; index < parameter.engine_output_count; ++index) {
    outputs_dims_[index] = engine_->getBindingDimensions(parameter.engine_input_count + index);
    assert(outputs_dims_[index].nbDims > 0);
    nvinfer1::DataType data_type = engine_->getBindingDataType(parameter.engine_input_count + index);
    CUDA_CHECK(cudaMalloc(&(binding_data_[parameter.engine_input_count + index]), DimsSize(outputs_dims_[index]) * GetDataTypeBytes(data_type)));
  }
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

tensorrt_flow::utils::TimeStatisic ModelFramework::GetPreProcessTimeStatistic(bool reset) {
  tensorrt_flow::utils::TimeStatisic time_static = pre_process_time_statistic_;
  if (reset) { pre_process_time_statistic_.reset(); }
  return time_static;
}
tensorrt_flow::utils::TimeStatisic ModelFramework::GetInferProcessTimeStatistic(bool reset) {
  tensorrt_flow::utils::TimeStatisic time_static = infer_process_time_statistic_;
  if (reset) { infer_process_time_statistic_.reset(); }
  return time_static;
}
tensorrt_flow::utils::TimeStatisic ModelFramework::GetPostProcessTimeStatistic(bool reset) {
  tensorrt_flow::utils::TimeStatisic time_static = post_process_time_statistic_;
  if (reset) { post_process_time_statistic_.reset(); }
  return time_static;
}


ModelFramework::~ModelFramework() {
  for (auto& ptr : binding_data_) { CUDA_CHECK(cudaFree(ptr)); }
}



}   // namespace model
}   // namespace tensorrt_flow