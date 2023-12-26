#pragma once

#include "tensorrt_flow/def/arch.hpp"
#include "tensorrt_flow/tensorrt/trt_model_plugin.hpp"

#include <array>
#include <string>

namespace tensorrt_flow {

namespace model {


namespace resnet50 {

struct Resnet50Parameter {

  struct image_info {
    int h{224};
    int w{224};
    int c{3};
  };
  image_info       img_info;
  size_t           num_class{1000};
  arch::ArchDevice device{arch::ArchDevice::CPU};
};
}   // namespace resnet50
}   // namespace model

}   // namespace tensorrt_flow