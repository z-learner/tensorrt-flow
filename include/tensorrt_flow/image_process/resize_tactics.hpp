#pragma once

namespace tensorrt_flow {

namespace image_process {
enum class ResizeTactics : int {
  CPU_NEAREST         = 0,
  CPU_BILINEAR        = 1,
  GPU_NEAREST         = 2,
  GPU_NEAREST_CENTER  = 3,
  GPU_BILINEAR        = 4,
  GPU_BILINEAR_CENTER = 5,
};

}
}   // namespace tensorrt_flow