#include "tensorrt_flow/memory/device_memory.hpp"

#include "tensorrt_flow/cuda/cuda_helper.hpp"

#include <assert.h>
#include <cstdlib>
#include <memory>
#include <string>

namespace tensorrt_flow {

namespace memory {

DeviceMemory::DeviceMemory(size_t memory_size, bool is_unified)
  : memory_size_(memory_size)
  , is_unified_(is_unified) {
  if (!is_unified) {
    CUDA_CHECK(cudaMalloc((void**)&memory_ptr_, memory_size));
    deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFree(ptr)); };
  } else {
    CUDA_CHECK(cudaMallocManaged((void**)&memory_ptr_, memory_size));
    deleter_function_ = [](void* ptr, size_t size) { CUDA_CHECK(cudaFree(ptr)); };
  }
}

DeviceMemory::DeviceMemory(void* memory_ptr, size_t memory_size, MemoryDeleteFunction deleter, bool is_unified)
  : memory_ptr_(memory_ptr)
  , memory_size_(memory_size)
  , deleter_function_(deleter)
  , is_unified_(is_unified) {}


bool DeviceMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  if (is_unified_) {
    assert(memory_size <= memory_size_);
    if (arch_device == arch::ArchDevice::CPU) {
      std::memcpy(memory_ptr_, memory_ptr, memory_size);
    } else if (arch_device == arch::ArchDevice::GPU) {
      CUDA_CHECK(cudaMemcpy(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost));
    } else {
      assert(false);
    }
    return true;
  }
  return false;
}

bool DeviceMemory::sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);
  if (arch_device == arch::ArchDevice::CPU) {
    CUDA_CHECK(cudaMemcpy(memory_ptr_, memory_ptr, memory_size, cudaMemcpyHostToDevice));
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpy(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToDevice));
  } else {
    assert(false);
  }
  return true;
}


bool DeviceMemory::sync_to_host(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  if (is_unified_) {
    assert(memory_size <= memory_size_);
    if (arch_device == arch::ArchDevice::CPU) {
      std::memcpy(memory_ptr_, memory_ptr, memory_size);
    } else if (arch_device == arch::ArchDevice::GPU) {
      CUDA_CHECK(cudaMemcpyAsync(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToHost, stream));
    } else {
      assert(false);
    }
    return true;
  }
  return false;
}
bool DeviceMemory::sync_to_device(void* memory_ptr, size_t memory_size, arch::ArchDevice arch_device, cudaStream_t& stream) {
  assert(arch_device != arch::ArchDevice::Unkonw);
  assert(memory_size <= memory_size_);
  if (arch_device == arch::ArchDevice::CPU) {
    CUDA_CHECK(cudaMemcpyAsync(memory_ptr_, memory_ptr, memory_size, cudaMemcpyHostToDevice, stream));
  } else if (arch_device == arch::ArchDevice::GPU) {
    CUDA_CHECK(cudaMemcpyAsync(memory_ptr_, memory_ptr, memory_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    assert(false);
  }
  return true;
}

DeviceMemory::~DeviceMemory() {
  if (is_last_object()) { deleter_function_(memory_ptr_, memory_size_); }
}

}   // namespace memory
}   // namespace tensorrt_flow