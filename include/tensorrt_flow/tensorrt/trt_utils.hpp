#pragma once

#include <iostream>
#include <limits>

namespace tensorrt_flow {


namespace utils {

class TimeStatisic {
public:
  TimeStatisic()  = default;
  ~TimeStatisic() = default;

  void set_value(float value);
  void reset();

  size_t count() { return count_; }
  float  min() { return min_; }
  float  max() { return max_; }
  float  avg() { return avg_; }

private:
  size_t count_{0};
  float  min_{std::numeric_limits<float>::max()};
  float  max_{std::numeric_limits<float>::min()};
  float  avg_{0};
};
}   // namespace utils


}   // namespace tensorrt_flow