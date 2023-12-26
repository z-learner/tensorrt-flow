#pragma once

#include <iostream>
#include <limits>
#include <type_traits>


#define DEFINE_HAS_MEMBER_FUNCTION(FunctionName)                                       \
  template<typename T, typename = void> struct Has##FunctionName : std::false_type {}; \
  template<typename T> struct Has##FunctionName<T, std::void_t<decltype(&T::FunctionName)>> : std::true_type {}


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