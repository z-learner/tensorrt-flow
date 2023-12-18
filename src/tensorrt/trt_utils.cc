#include "tensorrt_flow/tensorrt/trt_utils.hpp"

namespace tensorrt_flow {


namespace utils {


void TimeStatisic::set_value(float value) {
  min_ = std::min(min_, value);
  max_ = std::max(max_, value);
  avg_ = (avg_ * count_ + value) / (count_ + 1);
  count_ += 1;
}


void TimeStatisic::reset() {
  count_ = 0;
  min_   = std::numeric_limits<float>::max();
  max_   = std::numeric_limits<float>::min();
  avg_   = 0;
}


}   // namespace utils
}   // namespace tensorrt_flow