#pragma once
#include "NvInfer.h"
#include "tensorrt_flow/cuda/cuda_helper.hpp"
#include "tensorrt_flow/def/precision.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_model_plugin.hpp"
#include "tensorrt_flow/tensorrt/trt_utils.hpp"

#include <any>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace tensorrt_flow {

namespace model {

template<typename T> void destroy_trt_ptr(T* ptr) {
  if (ptr) {
    std::string type_name = typeid(T).name();
    ptr->destroy();
  };
}

struct ModelFrameworkParameter {
  std::string onnx_file_path{""};
  std::string engine_file_path{""};
  bool        is_use_engine_cache{true};

  size_t engine_input_count{1};
  size_t engine_output_count{1};

  logger::Level        logger_level{logger::Level::VERB};
  precision::Precision prec{precision::Precision::FP32};

  nvinfer1::NetworkDefinitionCreationFlags network_create_flags{1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

  std::shared_ptr<nvinfer1::IInt8EntropyCalibrator2> iint8_calibration{nullptr};

  int                          max_engine_workspace_size{1 << 28};
  nvinfer1::ProfilingVerbosity verbosity{nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY};

  nvinfer1::BuilderFlag build_percisin_constraints{nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS};

  int build_engine_from_onnx_log_level{1};
};



class ModelFramework {
public:
  ModelFramework(const ModelFrameworkParameter& parameter);
  ~ModelFramework();

  template<typename Plugin, typename... Args> bool InitPlugin(Args... args) {
    plugin_ptr_ = std::make_unique<Plugin>(std::forward<Args>(args)...);
    return true;
  }


  template<typename InputRawDataType> inline std::any Infer(InputRawDataType&& input_raw) {
    std::any any_input_data = std::make_any<InputRawDataType>(std::move(input_raw));

    cudaEvent_t start_pre_process, start_infer, start_post_process, end_post_process;
    CUDA_CHECK(cudaEventCreate(&start_pre_process));
    CUDA_CHECK(cudaEventRecord(start_pre_process, stream_));
    plugin_ptr_->PreProcess(any_input_data, binding_data_.data(), model_framework_parameter_.engine_input_count, inputs_dims_, stream_);
    CUDA_CHECK(cudaEventCreate(&start_infer));
    CUDA_CHECK(cudaEventRecord(start_infer, stream_));

    context_->enqueueV2(binding_data_.data(), stream_, nullptr);
    CUDA_CHECK(cudaEventCreate(&start_post_process));
    CUDA_CHECK(cudaEventRecord(start_post_process, stream_));

    auto result = std::move(plugin_ptr_->PostProcess(&(binding_data_[model_framework_parameter_.engine_input_count]), model_framework_parameter_.engine_output_count, outputs_dims_, stream_));
    CUDA_CHECK(cudaEventCreate(&end_post_process));
    CUDA_CHECK(cudaEventRecord(end_post_process, stream_));
    CUDA_CHECK(cudaEventSynchronize(end_post_process));

    float process_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&process_time, start_pre_process, start_infer));
    pre_process_time_statistic_.set_value(process_time);
    CUDA_CHECK(cudaEventElapsedTime(&process_time, start_infer, start_post_process));
    infer_process_time_statistic_.set_value(process_time);
    CUDA_CHECK(cudaEventElapsedTime(&process_time, start_post_process, end_post_process));
    post_process_time_statistic_.set_value(process_time);
    return result;
  }

  utils::TimeStatisic GetPreProcessTimeStatistic(bool reset = true);
  utils::TimeStatisic GetInferProcessTimeStatistic(bool reset = true);
  utils::TimeStatisic GetPostProcessTimeStatistic(bool reset = true);

private:
  std::shared_ptr<logger::Logger>               logger_{nullptr};
  std::shared_ptr<nvinfer1::IBuilder>           build_{nullptr};
  std::shared_ptr<nvinfer1::INetworkDefinition> network_{nullptr};
  std::shared_ptr<nvinfer1::ICudaEngine>        engine_{nullptr};
  std::shared_ptr<nvinfer1::IRuntime>           runtime_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext>  context_{nullptr};
  const ModelFrameworkParameter                 model_framework_parameter_{};

  std::vector<nvinfer1::Dims> inputs_dims_{};
  std::vector<nvinfer1::Dims> outputs_dims_{};
  cudaStream_t                stream_;

  std::vector<void*> binding_data_{};

  std::unique_ptr<ModelPlugin> plugin_ptr_{nullptr};


  // time statistic
  utils::TimeStatisic pre_process_time_statistic_{};
  utils::TimeStatisic infer_process_time_statistic_{};
  utils::TimeStatisic post_process_time_statistic_{};
};





}   // namespace model



}   // namespace tensorrt_flow
