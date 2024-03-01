#include "include/yolov8_parameter.hpp"
#include "include/yolov8_plugin.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "tensorrt_flow/cxxopts/cxxopts.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_model_framework.hpp"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string      image_parent_path{""};
  std::string      calibration_path{""};
  std::string      image_extension{".jpg"};
  cxxopts::Options options("yolo", "classification for image");

  options.add_options()("p,path", "image_parent_path", cxxopts::value<std::string>())("c,cpath", "calibration path", cxxopts::value<std::string>());

  auto prase_line_result = options.parse(argc, argv);

  image_parent_path = prase_line_result["path"].as<std::string>();
  calibration_path  = prase_line_result["cpath"].as<std::string>();

  tensorrt_flow::model::ModelFrameworkParameter parameter;
  parameter.logger_level     = tensorrt_flow::logger::Level::INFO;
  parameter.onnx_file_path   = "../model/onnx/yolov8n.onnx";
  parameter.engine_file_path = "../model/engine/yolov8n.engine";
  parameter.prec             = tensorrt_flow::precision::Precision::FP32;

  yolov8::parameter::Parameter yolo_parameter;
  yolo_parameter.img_info.w = 640;
  yolo_parameter.img_info.h = 640;
  yolo_parameter.img_info.c = 3;

  tensorrt_flow::model::ModelFramework framewrok(parameter);
  framewrok.InitPlugin<yolov8::plugin::Yolov8DetectorPlugin>(yolo_parameter);
  std::vector<fs::directory_entry> image_files;

  for (const auto& entry : fs::directory_iterator(image_parent_path)) {
    if (entry.path().filename().string().find(image_extension) == entry.path().filename().string().size() - 4) { image_files.emplace_back(std::move(entry)); }
  }
  cv::RNG rng(cv::getTickCount());
  for (auto& entry : image_files) {
    LOG_INFO("Start to infer : %s", entry.path().filename().string().c_str());
    auto result = framewrok.Infer(entry.path().string());

    std::vector<yolov8::plugin::bbox>& bboxs = *(std::any_cast<std::vector<yolov8::plugin::bbox>>(&result));

    // save result
    cv::Mat input_image = cv::imread(entry.path().string());
    for (auto& box : bboxs) {
      if (box.flg_remove) { continue; }
      cv::Scalar random_color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      auto       txt_color  = cv::Scalar(255 - random_color[0], 255 - random_color[1], 255 - random_color[2]);
      auto       txt        = cv::format({"%s: %.2f%%"}, box.label_name.c_str(), box.confidence * 100);
      int        base_line  = 0;
      auto       txt_size   = cv::getTextSize(txt, 0, 0.001 * std::min(input_image.cols, input_image.rows) * 0.001, 2, &base_line);
      int        txt_height = txt_size.height + base_line + 10;
      int        txt_width  = txt_size.width + 3;
      cv::Point  txt_pos(round(box.x0), round(box.y0 - (txt_size.height - base_line + 2)));
      cv::Rect   txt_rec(round(box.x0 - 2), round(box.y0 - txt_height), txt_width, txt_height);
      cv::Rect   box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

      cv::rectangle(input_image, box_rec, random_color, 3);
      cv::rectangle(input_image, txt_rec, random_color, -1);
      cv::putText(input_image, txt, txt_pos, 0, std::min(input_image.cols, input_image.rows) * 0.001, txt_color, 2, 16);
    }
    cv::imwrite(entry.path().parent_path().string() + "/result/" + entry.path().filename().string(), input_image);
  }
}