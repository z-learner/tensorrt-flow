#include "calibrator.hpp"
#include "tensorrt_flow/cxxopts/cxxopts.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_model_framework.hpp"
#include "trt_resnet50.hpp"

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

int main(int argc, char** argv) {

  std::string      image_parent_path{""};
  std::string      calibration_path{""};
  std::string      image_extension{".png"};
  cxxopts::Options options("resnet50", "classification for image");

  options.add_options()("p,path", "image_parent_path", cxxopts::value<std::string>())("c,cpath", "calibration path", cxxopts::value<std::string>());

  auto prase_line_result = options.parse(argc, argv);

  image_parent_path = prase_line_result["path"].as<std::string>();
  calibration_path  = prase_line_result["cpath"].as<std::string>();

  tensorrt_flow::model::ModelFrameworkParameter parameter;
  parameter.logger_level     = tensorrt_flow::logger::Level::INFO;
  parameter.onnx_file_path   = "../model/onnx/resnet50.onnx";
  parameter.engine_file_path = "../model/engine/resnet50.engine";
  parameter.prec             = tensorrt_flow::precision::Precision::INT8;

  tensorrt_flow::calibrator::TrtCalibratorParamter trt_calibration_parameter;
  trt_calibration_parameter.batch_size             = 1;
  trt_calibration_parameter.calibrator_dir_path    = calibration_path;
  trt_calibration_parameter.calibration_table_path = calibration_path + "/calibration_table.txt";
  trt_calibration_parameter.use_calibration_cache  = true;
  trt_calibration_parameter.regex_pattern          = ".*\.png$";

  tensorrt_flow::model::resnet50::Resnet50Parameter resnet50_parameter;

  parameter.iint8_calibration = std::shared_ptr<Resnet50Calibrator>(new Resnet50Calibrator(trt_calibration_parameter, resnet50_parameter));

  tensorrt_flow::model::ModelFramework framewrok(parameter);

  framewrok.InitPlugin<tensorrt_flow::model::resnet50::Resnet50Plugin>(resnet50_parameter);

  std::vector<fs::directory_entry> image_files;

  for (const auto& entry : fs::directory_iterator(image_parent_path)) {
    if (entry.path().filename().string().find(image_extension) == entry.path().filename().string().size() - 4) { image_files.emplace_back(std::move(entry)); }
  }

  LOG_INFO("Find %d images", image_files.size());
  for (auto& entry : image_files) {
    LOG_INFO("Start to infer : %s", entry.path().filename().string().c_str());
    auto        result = framewrok.Infer(entry.path().string());
    std::string label  = std::any_cast<std::string>(result);
    LOG_INFO("result : %s", label.c_str());
  }

  auto pre_process_time   = framewrok.GetPreProcessTimeStatistic();
  auto infer_process_time = framewrok.GetInferProcessTimeStatistic();
  auto post_process_time  = framewrok.GetPostProcessTimeStatistic();

  LOG_INFO("Pre process cost %f/%f/%f ms", pre_process_time.min(), pre_process_time.avg(), pre_process_time.max());
  LOG_INFO("Infer process cost %f/%f/%f ms", infer_process_time.min(), infer_process_time.avg(), infer_process_time.max());
  LOG_INFO("Post process cost %f/%f/%f ms", post_process_time.min(), post_process_time.avg(), post_process_time.max());

  return 0;
}
