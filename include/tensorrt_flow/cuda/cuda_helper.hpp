/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CUDA_CHECK_LAST_ERROR                                                                                                                                                      \
  {                                                                                                                                                                                \
    const cudaError_t __error = cudaGetLastError();                                                                                                                                \
    if (__error != cudaSuccess) { tensorrt_flow::cuda::detail::throw_re(__FILE__, __LINE__, "[cuda] ", "error code : ", __error, ", error msg : ", cudaGetErrorString(__error)); } \
  }

#define CUDA_CHECK(call)                                                                                                                                                           \
  {                                                                                                                                                                                \
    const cudaError_t __error = call;                                                                                                                                              \
    if (__error != cudaSuccess) { tensorrt_flow::cuda::detail::throw_re(__FILE__, __LINE__, "[cuda] ", "error code : ", __error, ", error msg : ", cudaGetErrorString(__error)); } \
  }
namespace tensorrt_flow {

namespace cuda {
namespace detail {
// Procedure: throw_se
// Throws the system error under a given error code.
template<typename... ArgsT>
// void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&...
// args) {
void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  // ostreamize(oss, std::forward<ArgsT>(args)...);
  (oss << ... << args);
  throw std::runtime_error(oss.str());
}

static size_t get_max_thread_pre_block(int device_id) {
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
  return device_prop.maxThreadsPerBlock;
}
}   // namespace detail
}   // namespace cuda
}   // namespace tensorrt_flow
