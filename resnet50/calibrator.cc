#include "calibrator.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"


std::vector<void*> Resnet50Calibrator::process(const std::vector<std::string>& file_path_name_list) {

  void* input_data_dev_ptr = input_data_;

  std::vector<float> temp_buff(resnet_parameter_.img_info.w * resnet_parameter_.img_info.h * resnet_parameter_.img_info.c);

  size_t count = 0;

  for (auto file_name : file_path_name_list) {
    input_data_dev_ptr += (count++ * resnet_parameter_.img_info.w * resnet_parameter_.img_info.h * resnet_parameter_.img_info.c * sizeof(float));
    cv::Mat input_image = cv::imread(file_name);
    cv::resize(input_image, input_image, cv::Size(resnet_parameter_.img_info.w, resnet_parameter_.img_info.h), 0, 0, cv::INTER_LINEAR);
    int index;
    int offset_ch0 = resnet_parameter_.img_info.w * resnet_parameter_.img_info.h * 0;
    int offset_ch1 = resnet_parameter_.img_info.w * resnet_parameter_.img_info.h * 1;
    int offset_ch2 = resnet_parameter_.img_info.w * resnet_parameter_.img_info.h * 2;
    for (int i = 0; i < resnet_parameter_.img_info.h; i++) {
      for (int j = 0; j < resnet_parameter_.img_info.w; j++) {
        index                   = i * resnet_parameter_.img_info.w * resnet_parameter_.img_info.c + j * resnet_parameter_.img_info.c;
        temp_buff[offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean_[0]) / std_[0];
        temp_buff[offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean_[1]) / std_[1];
        temp_buff[offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean_[2]) / std_[2];
      }
    }
    CUDA_CHECK(cudaMemcpy(input_data_dev_ptr, temp_buff.data(), temp_buff.size() * sizeof(float), cudaMemcpyHostToDevice));
  }

  return {input_data_};
}
