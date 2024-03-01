#pragma once


#include "affine.hpp"
#include "tensorrt_flow/def/arch.hpp"
#include "tensorrt_flow/tensorrt/trt_model_plugin.hpp"
#include "yolov8_parameter.hpp"

#include <array>
#include <string>
namespace yolov8 {

namespace plugin {


struct bbox {
  float x0, x1, y0, y1;
  float confidence;
  bool  flg_remove;
  int   label;

  std::string label_name{"Unknow"};

  bbox() = default;
  bbox(float x0, float y0, float x1, float y1, float conf, int label)
    : x0(x0)
    , y0(y0)
    , x1(x1)
    , y1(y1)
    , confidence(conf)
    , flg_remove(false)
    , label(label){};
};


class Yolov8DetectorPlugin : public tensorrt_flow::model::ModelPlugin {
public:
  Yolov8DetectorPlugin(yolov8::parameter::Parameter);
  virtual ~Yolov8DetectorPlugin();

  // input_raw_data is string
  virtual bool PreProcess(const std::any& input_raw_data, void** input_data, size_t input_size, std::vector<nvinfer1::Dims>& inputs_dims, cudaStream_t& stream) override;
  // return is ?
  virtual std::any PostProcess(void** engine_output_data, size_t output_size, std::vector<nvinfer1::Dims>& outputs_dims, cudaStream_t& stream) override;

private:
  yolov8::parameter::Parameter parameter_;
  size_t                       image_memory_size_{};
  float*                       input_data_{nullptr};
  float*                       output_data_{nullptr};

  yolov8::affine::TransInfo    trans_info_{0, 0, 0, 0};
  yolov8::affine::AffineMatrix affine_matrix_{};



  void*  origin_image_memory_device_{nullptr};
  size_t origin_image_memory_size_{0};

  size_t engine_output_bytes_size = 0;
};


}   // namespace plugin

}   // namespace yolov8