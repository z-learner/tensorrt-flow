#include "tensorrt_flow/image_process/resize_tactics.hpp"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"

#include <cuda_runtime.h>

namespace tensorrt_flow {


namespace cuda {


void resize_bgr2rgb_nhwc2nchw_gpu(float* target, uint8_t* source, int target_width, int target_height, int source_width, int source_height, float mean_channel0, float mean_channel1,
                                  float mean_channel2, float std_channel0, float std_channel1, float std_channel2, image_process::ResizeTactics tactics, cudaStream_t* stream_ptr);

}
}   // namespace tensorrt_flow