#pragma once
#include "NvInfer.h"
#include "tensorrt_flow/def/precision.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_model_plugin.hpp"

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
    LOGD("Destroy %s", type_name.c_str());
    ptr->destroy();
  };
}


struct ModelFrameworkParameter {
  std::string          onnx_file_path{""};
  std::string          engine_file_path{""};   // TODO
  logger::Level        logger_level{logger::Level::VERB};
  precision::Precision prec{precision::Precision::FP32};

  nvinfer1::NetworkDefinitionCreationFlags network_create_flags{1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

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
    std::any          any_input_data = std::make_any<InputRawDataType>(std::move(input_raw));
    plugin_ptr_->PreProcess(any_input_data, binding_data_.data(), input_count_, inputs_dims_, stream_);

    context_->enqueueV2(binding_data_.data(), stream_, nullptr);

    return plugin_ptr_->PostProcess(&(binding_data_[input_count_]), output_count_, outputs_dims_, stream_);
  }

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
  int32_t            input_count_{0};
  int32_t            output_count_{0};

  std::unique_ptr<ModelPlugin> plugin_ptr_{nullptr};
};





}   // namespace model



}   // namespace tensorrt_flow
