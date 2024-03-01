#include "yolov8_plugin.hpp"

#include "fmt/format.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "tensorrt_flow/cuda/cuda_helper.hpp"
#include "tensorrt_flow/cuda/image_resize_kernel.hpp"
#include "tensorrt_flow/image_process/image_resize.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_utils.hpp"

#include <algorithm>
#include <assert.h>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <thread>
namespace yolov8 {

namespace plugin {

static float iou_calc(const yolov8::plugin::bbox& bbox1, const yolov8::plugin::bbox& bbox2) {
  auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
  auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
  auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
  auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

  float inter_w = inter_x1 - inter_x0;
  float inter_h = inter_y1 - inter_y0;

  float inter_area = inter_w * inter_h;
  float union_area = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) + (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - inter_area;

  return inter_area / union_area;
}



Yolov8DetectorPlugin::Yolov8DetectorPlugin(yolov8::parameter::Parameter parameter)
  : parameter_{parameter} {
  image_memory_size_ = parameter.img_info.c * parameter.img_info.w * parameter.img_info.h * sizeof(float);
  LOG_DEBUG("Yolov8DetectorPlugin input image_memory_size : %d", image_memory_size_);
  CUDA_CHECK(cudaMallocHost((void**)&input_data_, image_memory_size_));
  engine_output_bytes_size = std::accumulate(parameter.output_dims.begin(), parameter.output_dims.end(), 1, std::multiplies<int>()) * sizeof(float);
  CUDA_CHECK(cudaMallocHost((void**)&output_data_, engine_output_bytes_size));
  // std::cout << "output_data size : " << std::accumulate(parameter.output_dims.begin(), parameter.output_dims.end(), 1, std::multiplies<int>()) * sizeof(float) << std::endl;
}

Yolov8DetectorPlugin::~Yolov8DetectorPlugin() {
  if (input_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(input_data_)); }
  if (output_data_ != nullptr) { CUDA_CHECK(cudaFreeHost(output_data_)); }
  if (origin_image_memory_device_ != nullptr) { CUDA_CHECK(cudaFree(origin_image_memory_device_)); }
}

bool Yolov8DetectorPlugin::PreProcess(const std::any& input_raw_data, void** input_data, size_t input_size, std::vector<nvinfer1::Dims>& inputs_dims, cudaStream_t& stream) {
  assert(input_size == 1);
  const std::string& input_image_file = *(std::any_cast<std::string>(&input_raw_data));
  cv::Mat            input_image      = cv::imread(input_image_file);

  size_t origin_image_size = input_image.rows * input_image.cols * input_image.channels();
  if (origin_image_memory_size_ < origin_image_size) {
    // reallocate memeory
    if (origin_image_memory_device_ != nullptr) {
      CUDA_CHECK(cudaFree(origin_image_memory_device_));
      origin_image_memory_device_ = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&origin_image_memory_device_, origin_image_size))
    origin_image_memory_size_ = origin_image_size;
  }

  CUDA_CHECK(cudaMemcpyAsync(origin_image_memory_device_, input_image.ptr(), origin_image_size, cudaMemcpyHostToDevice, stream));
  trans_info_.src_w = input_image.cols;
  trans_info_.src_h = input_image.rows;
  trans_info_.tar_w = parameter_.img_info.w;
  trans_info_.tar_h = parameter_.img_info.h;

  affine_matrix_.init(trans_info_);

  CUDA_CHECK(cudaMemset(input_data[0], 0, parameter_.img_info.w * parameter_.img_info.h * parameter_.img_info.c * sizeof(float)));
  yolov8::affine::launch_resize_affine((float*)input_data[0], (uint8_t*)origin_image_memory_device_, trans_info_, affine_matrix_, &stream);


#if 1
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> input_data_tmp(parameter_.img_info.w * parameter_.img_info.h * parameter_.img_info.c);

  CUDA_CHECK(cudaMemcpy(input_data_tmp.data(), input_data[0], input_data_tmp.size() * sizeof(float), cudaMemcpyDeviceToHost))

  auto image_path = std::filesystem::path(input_image_file);

  auto new_file = image_path.parent_path().string() + "/affine/gpu_affine_" + image_path.filename().string();

  cv::Mat save_mat(parameter_.img_info.h, parameter_.img_info.w, CV_8UC3);

  // back to HWC
  int offset_ch0 = parameter_.img_info.w * parameter_.img_info.h * 0;
  int offset_ch1 = parameter_.img_info.w * parameter_.img_info.h * 1;
  int offset_ch2 = parameter_.img_info.w * parameter_.img_info.h * 2;
  for (int i = 0; i < parameter_.img_info.h; ++i) {
    for (int j = 0; j < parameter_.img_info.w; ++j) {
      int index                = i * parameter_.img_info.w * parameter_.img_info.c + j * parameter_.img_info.c;
      save_mat.data[index + 0] = input_data_tmp[offset_ch2++] * 255.0f;
      save_mat.data[index + 1] = input_data_tmp[offset_ch1++] * 255.0f;
      save_mat.data[index + 2] = input_data_tmp[offset_ch0++] * 255.0f;
    }
  }
  cv::imwrite(new_file, save_mat);
#endif


  return true;
}


std::any Yolov8DetectorPlugin::PostProcess(void** engine_output_data, size_t output_size, std::vector<nvinfer1::Dims>& outputs_dims, cudaStream_t& stream) {
  float conf_threshold = 0.25;   //用来过滤decode时的bboxes
  float nms_threshold  = 0.45;   //用来过滤nms时的bboxes
  assert(output_size == 1);
  CUDA_CHECK(cudaMemcpyAsync((void*)output_data_, engine_output_data[0], engine_output_bytes_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // TODO
  int boxes_count = parameter_.output_dims[1];
  int class_count = parameter_.output_dims[2] - 4;

  // std::cout << "boxes_count :" << boxes_count << std::endl;
  // std::cout << "class_count :" << class_count << std::endl;

  std::any result;

  result.emplace<std::vector<yolov8::plugin::bbox>>();

  std::vector<yolov8::plugin::bbox>& bboxs = *(std::any_cast<std::vector<yolov8::plugin::bbox>>(&result));


  for (int i = 0; i < boxes_count; ++i) {
    float* tensor = output_data_ + i * (class_count + 4);
    int    label  = std::max_element(tensor + 4, tensor + 4 + class_count) - (tensor + 4);
    float  conf   = tensor[4 + label];
    if (conf < conf_threshold) { continue; }
    float cx = tensor[0];
    float cy = tensor[1];
    float w  = tensor[2];
    float h  = tensor[3];

    float x0 = cx - w / 2;
    float y0 = cy - h / 2;

    float x1 = x0 + w;
    float y1 = y0 + h;

    yolov8::affine::affine_transformation(affine_matrix_.reverse, x0, y0, x0, y0);
    yolov8::affine::affine_transformation(affine_matrix_.reverse, x1, y1, x1, y1);

    bboxs.emplace_back(x0, y0, x1, y1, conf, label);
  }

  LOG_INFO("the count of decoded bbox is %d", bboxs.size());
  std::sort(bboxs.begin(), bboxs.end(), [](bbox& box1, bbox& box2) { return box1.confidence > box2.confidence; });
  // NMS
  size_t real_box_size = 0;
  for (int i = 0; i < bboxs.size(); ++i) {
    if (bboxs[i].flg_remove) { continue; }
    real_box_size++;
    for (int j = i + 1; j < bboxs.size(); ++j) {
      if (bboxs[i].label == bboxs[j].label) {
        if (iou_calc(bboxs[i], bboxs[j]) > nms_threshold) { bboxs[j].flg_remove = true; }
      }
    }
  }
  LOG_INFO("the count of bbox after NMS is %d", real_box_size);
  for (auto& box : bboxs) {
    if (box.flg_remove) { continue; }
    box.label_name = yolov8::parameter::kCocoLabel[box.label];
  }
  return result;
}




}   // namespace plugin

}   // namespace yolov8
