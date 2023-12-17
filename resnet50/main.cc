#include "tensorrt_flow/cxxopts/cxxopts.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_model_frramework.hpp"
#include "trt_resnet50.hpp"

#include <iostream>

int main(int argc, char** argv) {

  std::string      image_file_name{""};
  cxxopts::Options options("resnet50", "classification for image");

  options.add_options()("f,file", "File name", cxxopts::value<std::string>());

  auto prase_line_result = options.parse(argc, argv);

  image_file_name = prase_line_result["file"].as<std::string>();

  tensorrt_flow::model::ModelFrameworkParameter parameter;
  parameter.logger_level     = tensorrt_flow::logger::Level::INFO;
  parameter.onnx_file_path   = "../model/onnx/resnet50.onnx";
  parameter.engine_file_path = "../model/engine/resnet50.engine";
  tensorrt_flow::model::ModelFramework framewrok(parameter);

  tensorrt_flow::model::resnet50::Resnet50Parameter resnet50_parameter;
  framewrok.InitPlugin<tensorrt_flow::model::resnet50::Resnet50Plugin>(resnet50_parameter);

  auto result = framewrok.Infer(image_file_name);

  std::string label = std::any_cast<std::string>(result);
  LOG_INFO("result : %s", label.c_str());
}
